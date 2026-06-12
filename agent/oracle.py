"""Oracle — fleet-level evaluation engine ("State of the Matrix").

A read-only meta-intelligence layer over the session state database. Where
the curator maintains individual skills, the Oracle watches the whole fleet:
it aggregates the InsightsEngine usage report, scans recent trajectories for
failure signals, and emits a structured report of detected anomalies with
recommendations.

Strict invariants:
  - The Oracle only observes and advises. It never modifies agent
    configuration, skills, or sessions (the "Agent Smith rule" — an
    optimizer must propose, never silently self-modify).
  - All heuristics are deterministic and run entirely offline; no LLM
    calls are made.
  - The only state it persists is its own run bookkeeping (iteration
    counter + last run timestamp) in ``state_meta``.

Usage:
    from agent.oracle import OracleEngine
    engine = OracleEngine(db)
    report = engine.generate(days=7)
    print(engine.format_terminal(report))
"""

import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.insights import InsightsEngine

# state_meta keys for run bookkeeping
_META_ITERATION = "oracle_iteration"
_META_LAST_RUN = "oracle_last_run_at"
_META_ACKS = "oracle_acks"

# An acknowledged anomaly reactivates when any shared numeric metric grows
# to this multiple of its value at ack time ("the numbers materially
# changed").
ACK_STALE_MULTIPLIER = 1.5

# Severity levels, in escalation order.
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"
_SEVERITY_RANK = {SEVERITY_INFO: 0, SEVERITY_WARNING: 1, SEVERITY_CRITICAL: 2}

# Verdicts, derived from the highest anomaly severity present.
VERDICT_STABLE = "stable"
VERDICT_DEGRADED = "degraded"
VERDICT_UNSTABLE = "unstable"

# --- Heuristic thresholds (module-level so tests can reference them) --------

# cost_spike: recent half-window cost vs prior half-window
COST_SPIKE_RATIO = 1.75
COST_SPIKE_MIN_USD = 1.0

# token_outlier: session total tokens vs window median
TOKEN_OUTLIER_MULTIPLIER = 5
TOKEN_OUTLIER_FLOOR = 200_000

# model_concentration: one model's share of total estimated cost
MODEL_CONCENTRATION_SHARE = 0.85
MODEL_CONCENTRATION_MIN_SESSIONS = 10
MODEL_CONCENTRATION_MIN_USD = 1.0

# cache_unused: total input tokens with zero cache reads
CACHE_UNUSED_MIN_INPUT = 1_000_000

# tool_failures: per-tool error rate
TOOL_FAILURE_RATE = 0.25
TOOL_FAILURE_MIN_CALLS = 5

# deja_vu: identical tool failing repeatedly within one session
DEJA_VU_MIN_REPEATS = 3

# ghost_sessions: fraction of sessions abandoned after <= 1 message
GHOST_FRACTION = 0.30
GHOST_MIN_SESSIONS = 10

# runaway_session: single session's cost vs the window median session cost
RUNAWAY_COST_MULTIPLIER = 10
RUNAWAY_MIN_USD = 5.0
# ...and escalation when one session dominates total window spend
RUNAWAY_CRITICAL_SHARE = 0.25

# spawn_burst: hourly session starts per source vs that source's own
# median hourly rate. The relative threshold keeps steady high-volume
# sources (cron firing all day) from flagging; the floor keeps quiet
# sources from flagging on trivial counts.
SPAWN_BURST_MULTIPLIER = 4
SPAWN_BURST_FLOOR = 30

# How many leading characters of a tool result to scan for error markers.
_ERROR_SCAN_CHARS = 500

# Failure-ish values of a structured result's "status" field. Guardrail
# outcomes (blocked, approval_required) are deliberately excluded — they
# are the system working, not the tool failing.
_FAILURE_STATUSES = {"error", "failed", "timeout", "disabled"}

_ERROR_MARKERS = (
    "error:",
    "error -",
    '"error"',
    "traceback (most recent call last)",
    "exception:",
    "failed:",
    "permission denied",
    "command not found",
    "no such file or directory",
    "timed out",
    "timeout",
)


def _looks_like_error(head: Optional[str]) -> bool:
    """Heuristic: does the start of a tool result read like a failure?

    Only used as a fallback for unstructured (non-JSON) results — see
    :func:`result_failed`. Output text routinely *mentions* errors
    (compiler logs, grepped tracebacks) without the call having failed,
    so structured fields always win when present.
    """
    if not head:
        return False
    h = head.lower()
    if h.startswith("error"):
        return True
    return any(marker in h for marker in _ERROR_MARKERS)


def result_failed(row: Dict[str, Any]) -> bool:
    """Classify one tool-result row as failed or not.

    Structured-first: tools like terminal return JSON
    ``{"output", "exit_code", "error", "status", ...}``, and the
    ``output`` field comes first — so its text can mention errors at
    length while the authoritative verdict sits in the trailing fields.
    Precedence:

      1. ``exit_code_meaning`` present → the tool itself marked a
         nonzero exit as benign (grep=1 "no matches") → success.
      2. ``exit_code`` → failed iff nonzero.
      3. ``error`` non-null/non-empty → failed.
      4. ``status`` in a known failure state → failed.
      5. Valid JSON with none of the above → success.
      6. Not JSON → fall back to the text-marker heuristic on the head.
    """
    if row.get("is_json"):
        if row.get("exit_code_meaning"):
            return False
        exit_code = row.get("exit_code")
        if exit_code is not None:
            try:
                return int(exit_code) != 0
            except (TypeError, ValueError):
                return True
        if row.get("error"):
            return True
        status = row.get("status")
        if status is not None:
            return str(status).lower() in _FAILURE_STATUSES
        return False
    return _looks_like_error(row.get("head"))


class OracleEngine:
    """Generates the fleet evaluation report from a SessionDB."""

    def __init__(self, db):
        self.db = db
        self._conn = db._conn
        self._insights = InsightsEngine(db)

    # =========================================================================
    # Report generation
    # =========================================================================

    def generate(self, days: int = 7, source: str = None,
                 bump_iteration: bool = True) -> Dict[str, Any]:
        """Generate a complete Oracle report.

        Args:
            days: Lookback window in days (default: 7).
            source: Optional platform filter (same semantics as insights).
            bump_iteration: When True, advance the persisted iteration
                counter (at most once per UTC day). Pass False for
                read-only refreshes (e.g. dashboard polling).

        Returns:
            Dict with the insights base report plus ``anomalies``,
            ``verdict``, ``daily`` trend, and run bookkeeping.
        """
        cutoff = time.time() - (days * 86400)
        base = self._insights.generate(days=days, source=source)

        if base.get("empty"):
            return {
                **base,
                "iteration": self._current_iteration(),
                "last_run_at": self.db.get_meta(_META_LAST_RUN),
                "verdict": VERDICT_STABLE,
                "anomalies": [],
                "daily": [],
                "recorded_cost": 0,
            }

        sessions = self._insights._get_sessions(cutoff, source)
        tool_results = self._get_tool_results(cutoff, source)
        daily = self._get_daily_trend(cutoff, source)

        # Two cost pipelines exist and must not be conflated:
        #   - recorded_cost: sums the DB columns (actual_cost_usd, falling
        #     back to estimated_cost_usd) — what the provider/session
        #     accounting actually recorded. Every cost-based ANOMALY
        #     (cost_spike, runaway_session, model_concentration) uses this.
        #   - overview.estimated_cost (from insights): re-derived from
        #     token counts x the bundled pricing table, including
        #     cache-read rates. Useful when DB columns are empty, but it
        #     can diverge badly for providers with free cache reads.
        recorded_cost = sum(
            s.get("actual_cost_usd") or s.get("estimated_cost_usd") or 0
            for s in sessions
        )

        anomalies: List[Dict[str, Any]] = []
        anomalies += self._detect_cost_spike(daily)
        anomalies += self._detect_token_outliers(sessions)
        anomalies += self._detect_runaway_session(sessions)
        anomalies += self._detect_spawn_burst(sessions)
        anomalies += self._detect_model_concentration(sessions)
        anomalies += self._detect_cache_unused(sessions)
        anomalies += self._detect_tool_failures(tool_results)
        anomalies += self._detect_deja_vu(tool_results)
        anomalies += self._detect_ghost_sessions(sessions)
        self._apply_acknowledgements(anomalies)
        anomalies.sort(
            key=lambda a: _SEVERITY_RANK.get(a["severity"], 0), reverse=True
        )

        iteration = (
            self._bump_iteration() if bump_iteration
            else self._current_iteration()
        )

        return {
            **base,
            "iteration": iteration,
            "last_run_at": self.db.get_meta(_META_LAST_RUN),
            "verdict": self._verdict(anomalies),
            "anomalies": anomalies,
            "daily": daily,
            "recorded_cost": recorded_cost,
        }

    @staticmethod
    def _verdict(anomalies: List[Dict[str, Any]]) -> str:
        worst = max(
            (_SEVERITY_RANK.get(a["severity"], 0) for a in anomalies),
            default=-1,
        )
        if worst >= _SEVERITY_RANK[SEVERITY_CRITICAL]:
            return VERDICT_UNSTABLE
        if worst >= _SEVERITY_RANK[SEVERITY_WARNING]:
            return VERDICT_DEGRADED
        return VERDICT_STABLE

    # =========================================================================
    # Acknowledgements
    # =========================================================================
    #
    # An acknowledged anomaly ("yes, I know — that was a decision") is
    # demoted to info so it stops driving the verdict, but stays visible.
    # It reactivates automatically when the numbers materially change:
    # any numeric metric growing to ACK_STALE_MULTIPLIER x its value at
    # ack time. Acks are keyed by the anomaly's per-instance `key`
    # (e.g. "cost_spike", "tool_failure_rate:memory").

    def get_acknowledgements(self) -> Dict[str, Dict[str, Any]]:
        raw = self.db.get_meta(_META_ACKS)
        if not raw:
            return {}
        try:
            acks = json.loads(raw)
            return acks if isinstance(acks, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _save_acknowledgements(self, acks: Dict[str, Dict[str, Any]]) -> None:
        self.db.set_meta(_META_ACKS, json.dumps(acks))

    def acknowledge(self, key: str, note: str = None) -> Dict[str, Any]:
        """Acknowledge a currently-active anomaly by its key.

        Snapshots the anomaly's metrics so material worsening can
        reactivate it. Raises ValueError if no active anomaly matches —
        acknowledging something that isn't firing would be a silent
        no-op trap.
        """
        report = self.generate(bump_iteration=False)
        match = next(
            (a for a in report.get("anomalies", []) if a["key"] == key), None
        )
        if match is None:
            active = ", ".join(a["key"] for a in report.get("anomalies", []))
            raise ValueError(
                f"No active anomaly with key '{key}'. "
                f"Active: {active or '(none)'}"
            )
        acks = self.get_acknowledgements()
        acks[key] = {
            "acked_at": datetime.now(timezone.utc).isoformat(),
            "note": note,
            "metrics": {
                k: v for k, v in match.get("metrics", {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
        }
        self._save_acknowledgements(acks)
        return acks[key]

    def unacknowledge(self, key: str) -> bool:
        """Remove an acknowledgement. Returns False if it didn't exist."""
        acks = self.get_acknowledgements()
        if key not in acks:
            return False
        del acks[key]
        self._save_acknowledgements(acks)
        return True

    @staticmethod
    def _ack_is_stale(ack: Dict[str, Any], anomaly: Dict[str, Any]) -> bool:
        """Has the anomaly worsened materially since it was acknowledged?"""
        snapshot = ack.get("metrics") or {}
        current = anomaly.get("metrics") or {}
        for k, snap in snapshot.items():
            cur = current.get(k)
            if not isinstance(cur, (int, float)) or isinstance(cur, bool):
                continue
            if not isinstance(snap, (int, float)) or snap <= 0:
                continue
            if cur >= snap * ACK_STALE_MULTIPLIER:
                return True
        return False

    def _apply_acknowledgements(self, anomalies: List[Dict[str, Any]]) -> None:
        """Demote acknowledged anomalies in place (or mark stale acks)."""
        acks = self.get_acknowledgements()
        if not acks:
            return
        for a in anomalies:
            ack = acks.get(a["key"])
            if ack is None:
                continue
            if self._ack_is_stale(ack, a):
                a["ack_stale"] = True
                continue
            a["acknowledged"] = True
            a["acked_at"] = ack.get("acked_at")
            if ack.get("note"):
                a["ack_note"] = ack["note"]
            a["computed_severity"] = a["severity"]
            a["severity"] = SEVERITY_INFO

    # =========================================================================
    # Run bookkeeping (iteration counter in state_meta)
    # =========================================================================

    def _current_iteration(self) -> int:
        raw = self.db.get_meta(_META_ITERATION)
        try:
            return int(raw) if raw else 0
        except (TypeError, ValueError):
            return 0

    def _bump_iteration(self) -> int:
        """Advance the iteration counter, at most once per UTC day."""
        current = self._current_iteration()
        today = datetime.now(timezone.utc).date().isoformat()
        last_run = self.db.get_meta(_META_LAST_RUN) or ""
        if last_run[:10] == today and current > 0:
            return current
        current += 1
        self.db.set_meta(_META_ITERATION, str(current))
        self.db.set_meta(
            _META_LAST_RUN, datetime.now(timezone.utc).isoformat()
        )
        return current

    # =========================================================================
    # Data gathering
    # =========================================================================

    def _get_tool_results(self, cutoff: float, source: str = None) -> List[Dict]:
        """Fetch tool results with structured failure fields where present.

        For JSON results (terminal et al.) the authoritative fields
        (``exit_code``, ``error``, ``status``, ``exit_code_meaning``) are
        extracted in SQL; for plain-text results only the first
        ``_ERROR_SCAN_CHARS`` chars are pulled for the fallback marker
        scan. Ordered by session + timestamp so repeat-failure runs can
        be detected in a single pass.
        """
        query = f"""SELECT m.session_id, m.tool_name,
                           substr(m.content, 1, {_ERROR_SCAN_CHARS}) AS head,
                           json_valid(m.content) AS is_json,
                           CASE WHEN json_valid(m.content)
                                THEN json_extract(m.content, '$.exit_code')
                           END AS exit_code,
                           CASE WHEN json_valid(m.content)
                                THEN json_extract(m.content, '$.error')
                           END AS error,
                           CASE WHEN json_valid(m.content)
                                THEN json_extract(m.content, '$.status')
                           END AS status,
                           CASE WHEN json_valid(m.content)
                                THEN json_extract(m.content, '$.exit_code_meaning')
                           END AS exit_code_meaning
                    FROM messages m
                    JOIN sessions s ON s.id = m.session_id
                    WHERE s.started_at >= ? AND m.role = 'tool'
                      AND m.tool_name IS NOT NULL"""
        params: tuple = (cutoff,)
        if source:
            query += " AND s.source = ?"
            params = (cutoff, source)
        query += " ORDER BY m.session_id, m.timestamp"
        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def _get_daily_trend(self, cutoff: float, source: str = None) -> List[Dict]:
        query = """SELECT date(started_at, 'unixepoch') AS day,
                          COUNT(*) AS sessions,
                          SUM(input_tokens) AS input_tokens,
                          SUM(output_tokens) AS output_tokens,
                          COALESCE(SUM(COALESCE(actual_cost_usd,
                                                estimated_cost_usd, 0)), 0)
                              AS cost_usd
                   FROM sessions WHERE started_at >= ?"""
        params: tuple = (cutoff,)
        if source:
            query += " AND source = ?"
            params = (cutoff, source)
        query += " GROUP BY day ORDER BY day"
        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Anomaly heuristics
    # =========================================================================

    @staticmethod
    def _anomaly(code: str, severity: str, title: str, detail: str,
                 recommendation: str, **metrics) -> Dict[str, Any]:
        # Some codes fire once per tool/source; the key is the stable
        # per-instance handle used for acknowledgement.
        qualifier = metrics.get("tool") or metrics.get("source")
        return {
            "code": code,
            "key": f"{code}:{qualifier}" if qualifier else code,
            "severity": severity,
            "title": title,
            "detail": detail,
            "recommendation": recommendation,
            "metrics": metrics,
        }

    def _detect_cost_spike(self, daily: List[Dict]) -> List[Dict]:
        """Recent half of the window costs >> the prior half."""
        if len(daily) < 4:
            return []
        mid = len(daily) // 2
        prior = sum(d["cost_usd"] or 0 for d in daily[:mid])
        recent = sum(d["cost_usd"] or 0 for d in daily[mid:])
        if recent < COST_SPIKE_MIN_USD or prior <= 0:
            return []
        ratio = recent / prior
        if ratio < COST_SPIKE_RATIO:
            return []
        return [self._anomaly(
            "cost_spike",
            SEVERITY_WARNING if ratio < 3 else SEVERITY_CRITICAL,
            "Spend is accelerating",
            f"Cost in the recent half of the window (${recent:.2f}) is "
            f"{ratio:.1f}x the prior half (${prior:.2f}).",
            "Check the top sessions for runaway loops or an unintended "
            "switch to a more expensive model.",
            recent_usd=round(recent, 2),
            prior_usd=round(prior, 2),
            ratio=round(ratio, 2),
        )]

    def _detect_token_outliers(self, sessions: List[Dict]) -> List[Dict]:
        """Sessions burning far more tokens than the window median."""
        totals = [
            ((s.get("input_tokens") or 0) + (s.get("output_tokens") or 0), s)
            for s in sessions
        ]
        nonzero = [t for t, _ in totals if t > 0]
        if len(nonzero) < 5:
            return []
        median = statistics.median(nonzero)
        threshold = max(median * TOKEN_OUTLIER_MULTIPLIER, TOKEN_OUTLIER_FLOOR)
        outliers = sorted(
            (pair for pair in totals if pair[0] >= threshold),
            key=lambda p: p[0], reverse=True,
        )
        if not outliers:
            return []
        worst_tokens, worst = outliers[0]
        return [self._anomaly(
            "token_outlier",
            SEVERITY_WARNING,
            "Token-burn outlier sessions",
            f"{len(outliers)} session(s) consumed >= {threshold:,.0f} tokens "
            f"(window median: {median:,.0f}). Worst: session "
            f"{worst['id'][:8]} on {worst.get('source', '?')} with "
            f"{worst_tokens:,} tokens.",
            "Inspect these sessions for retry loops or oversized context; "
            "consider /compress or splitting the task across subagents.",
            outlier_count=len(outliers),
            threshold=int(threshold),
            median=int(median),
            worst_session_id=worst["id"],
            worst_tokens=worst_tokens,
        )]

    def _detect_runaway_session(self, sessions: List[Dict]) -> List[Dict]:
        """One session costing far more than the window median session.

        Cost concentration, not wall-clock duration, is the signal:
        gateway sessions legitimately stay open (idle) for days, but a
        single conversation quietly accumulating a large share of spend
        is a runaway regardless of how long it sat open.
        """
        def _cost(s: Dict) -> float:
            return s.get("actual_cost_usd") or s.get("estimated_cost_usd") or 0

        costs = [_cost(s) for s in sessions if _cost(s) > 0]
        if len(costs) < 5:
            return []
        median = statistics.median(costs)
        total = sum(costs)
        threshold = max(median * RUNAWAY_COST_MULTIPLIER, RUNAWAY_MIN_USD)
        runaways = sorted(
            (s for s in sessions if _cost(s) >= threshold),
            key=_cost, reverse=True,
        )
        if not runaways:
            return []
        worst = runaways[0]
        worst_cost = _cost(worst)
        share = worst_cost / total if total else 0
        start, end = worst.get("started_at"), worst.get("ended_at")
        hours = (end - start) / 3600 if start and end and end > start else None
        duration = f", open {hours:.0f}h" if hours else ", still open"
        return [self._anomaly(
            "runaway_session",
            SEVERITY_CRITICAL if share >= RUNAWAY_CRITICAL_SHARE
            else SEVERITY_WARNING,
            "Runaway session cost",
            f"{len(runaways)} session(s) cost >= ${threshold:.2f} "
            f"(median session: ${median:.2f}). Worst: session "
            f"{worst['id'][:8]} on {worst.get('source', '?')} at "
            f"${worst_cost:.2f} — {share:.0%} of window spend{duration}.",
            "Review the conversation: a long-lived chat accumulating "
            "context re-reads gets expensive — /compress it, /new it, or "
            "split the work across fresh sessions.",
            runaway_count=len(runaways),
            threshold_usd=round(threshold, 2),
            median_usd=round(median, 2),
            worst_session_id=worst["id"],
            worst_cost_usd=round(worst_cost, 2),
            share=round(share, 3),
        )]

    def _detect_spawn_burst(self, sessions: List[Dict]) -> List[Dict]:
        """Session-creation burst: one source spiking far above its own
        steady hourly rate.

        Compared per-source against that source's median active-hour
        rate, so a cron scheduler steadily firing all day never flags —
        only a deviation from the source's own normal does.
        """
        starts_by_source: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for s in sessions:
            started = s.get("started_at")
            if started:
                starts_by_source[s.get("source") or "unknown"][
                    int(started // 3600)
                ] += 1

        anomalies = []
        for src, hours in sorted(starts_by_source.items()):
            counts = list(hours.values())
            median = statistics.median(counts)
            threshold = max(median * SPAWN_BURST_MULTIPLIER, SPAWN_BURST_FLOOR)
            bursts = {h: c for h, c in hours.items() if c >= threshold}
            if not bursts:
                continue
            worst_hour, worst_count = max(bursts.items(), key=lambda kv: kv[1])
            when = datetime.fromtimestamp(
                worst_hour * 3600, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:00 UTC")
            anomalies.append(self._anomaly(
                "spawn_burst",
                SEVERITY_WARNING,
                f"Session spawn burst: {src}",
                f"{src} started {worst_count} sessions in one hour "
                f"({when}); its median active-hour rate is {median:.0f}. "
                f"{len(bursts)} hour(s) exceeded the burst threshold "
                f"({threshold:.0f}).",
                "Look for a retrigger loop — a crashing gateway "
                "reconnecting, a cron job spawning sub-sessions, or a "
                "subagent fork bomb.",
                source=src,
                worst_hour=when,
                worst_count=worst_count,
                median_per_hour=median,
                burst_hours=len(bursts),
            ))
        return anomalies

    def _detect_model_concentration(self, sessions: List[Dict]) -> List[Dict]:
        """Nearly all spend funneled through a single model."""
        if len(sessions) < MODEL_CONCENTRATION_MIN_SESSIONS:
            return []
        cost_by_model: Dict[str, float] = defaultdict(float)
        for s in sessions:
            cost = s.get("actual_cost_usd") or s.get("estimated_cost_usd") or 0
            cost_by_model[s.get("model") or "unknown"] += cost
        total = sum(cost_by_model.values())
        if total < MODEL_CONCENTRATION_MIN_USD:
            return []
        top_model, top_cost = max(cost_by_model.items(), key=lambda kv: kv[1])
        share = top_cost / total
        if share < MODEL_CONCENTRATION_SHARE or len(cost_by_model) < 2:
            return []
        return [self._anomaly(
            "model_concentration",
            SEVERITY_INFO,
            "Spend concentrated on one model",
            f"{top_model} accounts for {share:.0%} of estimated spend "
            f"(${top_cost:.2f} of ${total:.2f}).",
            "Route summarization, triage, and other simple tasks to a "
            "cheaper model (hermes model / per-agent model config).",
            model=top_model,
            share=round(share, 3),
            cost_usd=round(top_cost, 2),
        )]

    def _detect_cache_unused(self, sessions: List[Dict]) -> List[Dict]:
        """Heavy input volume with zero prompt-cache reads."""
        total_input = sum(s.get("input_tokens") or 0 for s in sessions)
        total_cache = sum(s.get("cache_read_tokens") or 0 for s in sessions)
        if total_input < CACHE_UNUSED_MIN_INPUT or total_cache > 0:
            return []
        return [self._anomaly(
            "cache_unused",
            SEVERITY_INFO,
            "Prompt caching unused",
            f"{total_input:,} input tokens in the window with zero cache "
            "reads recorded.",
            "If your provider supports prompt caching, enabling it can cut "
            "input cost substantially on long-running sessions.",
            input_tokens=total_input,
        )]

    def _detect_tool_failures(self, tool_results: List[Dict]) -> List[Dict]:
        """Tools whose results read like errors at a high rate."""
        calls: Dict[str, int] = defaultdict(int)
        failures: Dict[str, int] = defaultdict(int)
        for r in tool_results:
            name = r["tool_name"]
            calls[name] += 1
            if result_failed(r):
                failures[name] += 1
        anomalies = []
        for name, n_calls in sorted(calls.items()):
            if n_calls < TOOL_FAILURE_MIN_CALLS:
                continue
            rate = failures[name] / n_calls
            if rate < TOOL_FAILURE_RATE:
                continue
            anomalies.append(self._anomaly(
                "tool_failure_rate",
                SEVERITY_CRITICAL if rate >= 0.5 else SEVERITY_WARNING,
                f"High failure rate: {name}",
                f"{failures[name]} of {n_calls} {name} results in the "
                f"window look like errors ({rate:.0%}).",
                f"Check {name}'s configuration, credentials, or the "
                "environments it runs in — agents are burning turns on it.",
                tool=name,
                calls=n_calls,
                failures=failures[name],
                rate=round(rate, 3),
            ))
        return anomalies

    def _detect_deja_vu(self, tool_results: List[Dict]) -> List[Dict]:
        """The same tool failing repeatedly inside a single session.

        Results arrive ordered by (session, timestamp); count consecutive
        failing results of the same tool per session.
        """
        worst: Dict[str, Any] = {}
        affected = set()
        streak_tool = None
        streak_session = None
        streak = 0
        for r in tool_results + [{"session_id": None, "tool_name": None, "head": ""}]:
            failing = result_failed(r)
            same = (
                failing
                and r["session_id"] == streak_session
                and r["tool_name"] == streak_tool
            )
            if same:
                streak += 1
            else:
                if streak >= DEJA_VU_MIN_REPEATS:
                    affected.add(streak_session)
                    if streak > worst.get("repeats", 0):
                        worst = {
                            "session_id": streak_session,
                            "tool": streak_tool,
                            "repeats": streak,
                        }
                if failing:
                    streak_session = r["session_id"]
                    streak_tool = r["tool_name"]
                    streak = 1
                else:
                    streak_session = streak_tool = None
                    streak = 0
        if not worst:
            return []
        return [self._anomaly(
            "deja_vu",
            SEVERITY_WARNING,
            "Repeated failure loop (déjà vu)",
            f"Session {worst['session_id'][:8]} hit {worst['repeats']} "
            f"consecutive {worst['tool']} failures; {len(affected)} "
            "session(s) showed this pattern.",
            "An agent retrying the same failing call is a glitch worth "
            "fixing at the source — review the session transcript.",
            session_id=worst["session_id"],
            tool=worst["tool"],
            repeats=worst["repeats"],
            affected_sessions=len(affected),
        )]

    def _detect_ghost_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """A large share of sessions abandoned after <= 1 message."""
        if len(sessions) < GHOST_MIN_SESSIONS:
            return []
        ghosts = [s for s in sessions if (s.get("message_count") or 0) <= 1]
        fraction = len(ghosts) / len(sessions)
        if fraction < GHOST_FRACTION:
            return []
        return [self._anomaly(
            "ghost_sessions",
            SEVERITY_INFO,
            "Many sessions abandoned immediately",
            f"{len(ghosts)} of {len(sessions)} sessions "
            f"({fraction:.0%}) ended with at most one message.",
            "If these are crashes or misfires (cron jobs, gateway "
            "reconnects), find the trigger; if intentional, ignore.",
            ghost_count=len(ghosts),
            total=len(sessions),
            fraction=round(fraction, 3),
        )]

    # =========================================================================
    # Terminal rendering
    # =========================================================================

    # Matrix-green ANSI styling for the CLI report.
    _G = "\033[38;5;46m"      # bright phosphor green
    _DG = "\033[38;5;28m"     # dim green
    _Y = "\033[38;5;226m"     # warning yellow
    _R = "\033[38;5;196m"     # critical red
    _B = "\033[1m"
    _RST = "\033[0m"

    _VERDICT_LABELS = {
        VERDICT_STABLE: "SYSTEM STABLE — the Matrix hums along",
        VERDICT_DEGRADED: "ANOMALIES DETECTED — glitches in the Matrix",
        VERDICT_UNSTABLE: "SYSTEM UNSTABLE — intervention required",
    }

    def format_terminal(self, report: Dict[str, Any], color: bool = True) -> str:
        """Render the report for the terminal ('State of the Matrix')."""
        g, dg, y, r, b, rst = (
            (self._G, self._DG, self._Y, self._R, self._B, self._RST)
            if color else ("",) * 6
        )
        sev_color = {SEVERITY_INFO: dg, SEVERITY_WARNING: y, SEVERITY_CRITICAL: r}
        lines = []
        rule = g + "═" * 64 + rst
        lines.append(rule)
        lines.append(
            f"{g}{b}  THE ORACLE — STATE OF THE MATRIX"
            f"{rst}{dg}   iteration {report.get('iteration', 0)}{rst}"
        )
        lines.append(rule)

        verdict = report.get("verdict", VERDICT_STABLE)
        vcolor = {VERDICT_STABLE: g, VERDICT_DEGRADED: y, VERDICT_UNSTABLE: r}
        lines.append("")
        lines.append(
            f"  {vcolor.get(verdict, g)}{b}◉ "
            f"{self._VERDICT_LABELS.get(verdict, verdict)}{rst}"
        )

        if report.get("empty"):
            lines.append("")
            lines.append(f"  {dg}No sessions recorded in the last "
                         f"{report.get('days', '?')} days. "
                         f"There is nothing to see... yet.{rst}")
            lines.append(rule)
            return "\n".join(lines)

        o = report.get("overview", {})
        lines.append("")
        lines.append(f"{g}{b}  THE CONSTRUCT{rst}{dg} — last "
                     f"{report.get('days')} days{rst}")
        lines.append(
            f"  {g}sessions{rst} {o.get('total_sessions', 0):<8}"
            f"{g}messages{rst} {o.get('total_messages', 0):<10}"
            f"{g}tokens{rst} {o.get('total_tokens', 0):,}"
        )
        modeled = o.get("estimated_cost") or 0
        recorded = report.get("recorded_cost") or 0
        if recorded:
            lines.append(f"  {g}recorded cost{rst} ${recorded:,.2f}")
            # Surface the modeled figure only when it tells a different
            # story — a silent mix of the two pipelines is misleading.
            if modeled and not (0.8 <= modeled / recorded <= 1.25):
                lines.append(
                    f"  {dg}modeled cost ${modeled:,.2f} — token counts x "
                    f"pricing table; diverges from recorded (trust "
                    f"recorded; anomalies use it){rst}"
                )
        elif modeled:
            lines.append(
                f"  {g}est. cost{rst} ${modeled:,.2f} "
                f"{dg}(modeled — no recorded costs in DB){rst}"
            )

        models = report.get("models") or []
        if models:
            lines.append("")
            lines.append(f"{g}{b}  PROGRAMS{rst}{dg} — by model{rst}")
            for m in models[:6]:
                lines.append(
                    f"  {dg}▸{rst} {m['model'][:36]:<38}"
                    f"{m['sessions']:>4} sessions  "
                    f"{m['total_tokens']:>12,} tok"
                )

        anomalies = report.get("anomalies") or []
        lines.append("")
        if anomalies:
            lines.append(f"{g}{b}  ANOMALIES{rst}{dg} — "
                         f"{len(anomalies)} detected{rst}")
            for a in anomalies:
                c = sev_color.get(a["severity"], dg)
                marker = "✔" if a.get("acknowledged") else "▲"
                suffix = f"  {dg}[{a['key']}]{rst}"
                if a.get("acknowledged"):
                    acked = (a.get("acked_at") or "")[:10]
                    suffix += f" {dg}acknowledged {acked}{rst}"
                elif a.get("ack_stale"):
                    suffix += (f" {y}ack stale — numbers worsened since "
                               f"acknowledged{rst}")
                lines.append(f"  {c}{b}{marker} [{a['severity'].upper()}] "
                             f"{a['title']}{rst}{suffix}")
                lines.append(f"    {dg}{a['detail']}{rst}")
                if a.get("ack_note"):
                    lines.append(f"    {dg}ack note: {a['ack_note']}{rst}")
                lines.append(f"    {g}→ {a['recommendation']}{rst}")
        else:
            lines.append(f"  {dg}No anomalies. Déjà vu has not been "
                         f"reported.{rst}")

        lines.append("")
        lines.append(rule)
        return "\n".join(lines)

    def format_json(self, report: Dict[str, Any]) -> str:
        return json.dumps(report, indent=2, default=str)
