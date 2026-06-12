"""Tests for agent/oracle.py — OracleEngine fleet evaluation and anomalies."""

import time

import pytest

import json

from hermes_state import SessionDB
from agent.oracle import (
    OracleEngine,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    VERDICT_DEGRADED,
    VERDICT_STABLE,
    VERDICT_UNSTABLE,
    _looks_like_error,
    result_failed,
)


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_oracle.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


@pytest.fixture()
def engine(db):
    return OracleEngine(db)


def _make_session(db, sid, *, source="cli", model="test/model-a",
                  input_tokens=5000, output_tokens=1000, messages=3,
                  days_ago=0.0, cost=None):
    db.create_session(session_id=sid, source=source, model=model)
    db.update_token_counts(sid, input_tokens=input_tokens,
                           output_tokens=output_tokens)
    for i in range(messages):
        db.append_message(sid, role="user" if i % 2 == 0 else "assistant",
                          content=f"msg {i}")
    started = time.time() - days_ago * 86400
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, estimated_cost_usd = ? WHERE id = ?",
        (started, cost, sid),
    )
    db._conn.commit()


# =============================================================================
# Error-marker heuristic
# =============================================================================

def test_looks_like_error_positive():
    assert _looks_like_error("Error: connection refused")
    assert _looks_like_error("bash: foo: command not found")
    assert _looks_like_error("Traceback (most recent call last):\n  ...")
    assert _looks_like_error("Request timed out after 30s")


def test_looks_like_error_negative():
    assert not _looks_like_error(None)
    assert not _looks_like_error("")
    assert not _looks_like_error("Found 3 matches in src/")
    assert not _looks_like_error("ok")


# =============================================================================
# Structured result classification (result_failed)
# =============================================================================

def _row(content=None, **fields):
    """Build a tool-result row like _get_tool_results returns."""
    return {
        "session_id": "s", "tool_name": "terminal",
        "head": (content or "")[:500],
        "is_json": fields.pop("is_json", 1),
        "exit_code": fields.pop("exit_code", None),
        "error": fields.pop("error", None),
        "status": fields.pop("status", None),
        "exit_code_meaning": fields.pop("exit_code_meaning", None),
    }


def test_structured_success_not_failed():
    assert not result_failed(_row(exit_code=0))


def test_structured_nonzero_exit_failed():
    assert result_failed(_row(exit_code=1))
    assert result_failed(_row(exit_code=-1))


def test_benign_nonzero_exit_not_failed():
    # grep=1 "no matches" — the tool marked it benign via exit_code_meaning.
    assert not result_failed(
        _row(exit_code=1, exit_code_meaning="no matches found")
    )


def test_error_field_failed():
    assert result_failed(_row(error="Command timed out"))
    assert not result_failed(_row(error=None))


def test_status_field():
    assert result_failed(_row(status="error"))
    assert result_failed(_row(status="timeout"))
    # Guardrail outcomes are not tool failures.
    assert not result_failed(_row(status="blocked"))
    assert not result_failed(_row(status="approval_required"))


def test_error_text_in_output_not_failed_when_structured():
    """The poisoning regression: a successful terminal call whose OUTPUT
    mentions errors (grepping logs, compiler warnings) must not count."""
    content = json.dumps({
        "output": "Error: connection refused\nTraceback (most recent call last)",
        "exit_code": 0,
        "error": None,
    })
    assert not result_failed(_row(content=content, exit_code=0))


def test_plain_text_falls_back_to_marker_heuristic():
    assert result_failed({"is_json": 0, "head": "Error: no such tool"})
    assert not result_failed({"is_json": 0, "head": "Found 3 matches"})


# =============================================================================
# Empty database
# =============================================================================

def test_empty_db_report(engine):
    report = engine.generate(days=7)
    assert report["empty"] is True
    assert report["verdict"] == VERDICT_STABLE
    assert report["anomalies"] == []
    assert report["daily"] == []
    assert report["iteration"] == 0


def test_empty_report_renders(engine):
    report = engine.generate(days=7)
    text = engine.format_terminal(report, color=False)
    assert "STATE OF THE MATRIX" in text
    assert "No sessions recorded" in text


# =============================================================================
# Iteration bookkeeping
# =============================================================================

def test_iteration_bumps_once_per_day(db, engine):
    _make_session(db, "s1")
    r1 = engine.generate(days=7)
    assert r1["iteration"] == 1
    # Same day: no further advance.
    r2 = engine.generate(days=7)
    assert r2["iteration"] == 1


def test_iteration_not_bumped_when_disabled(db, engine):
    _make_session(db, "s1")
    report = engine.generate(days=7, bump_iteration=False)
    assert report["iteration"] == 0
    assert db.get_meta("oracle_iteration") is None


# =============================================================================
# Anomaly heuristics
# =============================================================================

def _codes(report):
    return {a["code"] for a in report["anomalies"]}


def test_healthy_fleet_is_stable(db, engine):
    for i in range(6):
        _make_session(db, f"s{i}", input_tokens=8000, output_tokens=2000)
    report = engine.generate(days=7)
    assert report["verdict"] == VERDICT_STABLE
    assert report["anomalies"] == []


def test_token_outlier_detected(db, engine):
    for i in range(6):
        _make_session(db, f"s{i}", input_tokens=8000, output_tokens=2000)
    _make_session(db, "whale", input_tokens=400_000, output_tokens=100_000)
    report = engine.generate(days=7)
    assert "token_outlier" in _codes(report)
    anomaly = next(a for a in report["anomalies"]
                   if a["code"] == "token_outlier")
    assert anomaly["metrics"]["worst_session_id"] == "whale"
    assert report["verdict"] == VERDICT_DEGRADED


def test_tool_failure_rate_detected(db, engine):
    _make_session(db, "s1", messages=0)
    for i in range(6):
        db.append_message("s1", role="tool",
                          content="Error: connection refused",
                          tool_name="web_fetch")
    report = engine.generate(days=7)
    assert "tool_failure_rate" in _codes(report)
    anomaly = next(a for a in report["anomalies"]
                   if a["code"] == "tool_failure_rate")
    assert anomaly["metrics"]["tool"] == "web_fetch"
    assert anomaly["severity"] == SEVERITY_CRITICAL  # 100% failure rate
    assert report["verdict"] == VERDICT_UNSTABLE


def test_tool_failures_below_threshold_ignored(db, engine):
    _make_session(db, "s1", messages=0)
    for i in range(9):
        db.append_message("s1", role="tool", content="all good",
                          tool_name="web_fetch")
    db.append_message("s1", role="tool", content="Error: once",
                      tool_name="web_fetch")
    report = engine.generate(days=7)
    assert "tool_failure_rate" not in _codes(report)


def test_deja_vu_detected(db, engine):
    _make_session(db, "s1", messages=0)
    for i in range(4):
        db.append_message("s1", role="tool",
                          content="Error: still broken",
                          tool_name="patch")
    report = engine.generate(days=7)
    assert "deja_vu" in _codes(report)
    anomaly = next(a for a in report["anomalies"] if a["code"] == "deja_vu")
    assert anomaly["metrics"]["tool"] == "patch"
    assert anomaly["metrics"]["repeats"] == 4
    assert anomaly["severity"] == SEVERITY_WARNING


def test_deja_vu_requires_consecutive_failures(db, engine):
    _make_session(db, "s1", messages=0)
    # Failures interleaved with successes — never 3 in a row.
    for i in range(4):
        db.append_message("s1", role="tool", content="Error: flaky",
                          tool_name="patch")
        db.append_message("s1", role="tool", content="recovered fine",
                          tool_name="patch")
    report = engine.generate(days=7)
    assert "deja_vu" not in _codes(report)


def test_json_failures_detected_end_to_end(db, engine):
    """Structured terminal failures stored as JSON trip both detectors."""
    _make_session(db, "s1", messages=0)
    failing = json.dumps({"output": "", "exit_code": 127,
                          "error": "zsh: command not found: foo"})
    for i in range(6):
        db.append_message("s1", role="tool", content=failing,
                          tool_name="terminal")
    report = engine.generate(days=7)
    assert "tool_failure_rate" in _codes(report)
    assert "deja_vu" in _codes(report)


def test_json_successes_with_error_text_not_flagged(db, engine):
    """End-to-end poisoning regression: successful calls whose output
    contains error text must not inflate the failure rate."""
    _make_session(db, "s1", messages=0)
    noisy_success = json.dumps({
        "output": "grep results:\nerror: handling in foo.py\nError: retry in bar.py",
        "exit_code": 0,
        "error": None,
    })
    for i in range(8):
        db.append_message("s1", role="tool", content=noisy_success,
                          tool_name="terminal")
    report = engine.generate(days=7)
    assert "tool_failure_rate" not in _codes(report)
    assert "deja_vu" not in _codes(report)


def test_runaway_session_detected(db, engine):
    for i in range(6):
        _make_session(db, f"s{i}", cost=0.50)
    _make_session(db, "whale", source="telegram", cost=8.0)
    report = engine.generate(days=7)
    assert "runaway_session" in _codes(report)
    anomaly = next(a for a in report["anomalies"]
                   if a["code"] == "runaway_session")
    assert anomaly["metrics"]["worst_session_id"] == "whale"
    # 8.0 of 11.0 total = 73% of window spend — escalates to critical.
    assert anomaly["severity"] == SEVERITY_CRITICAL


def test_no_runaway_when_costs_are_even(db, engine):
    for i in range(8):
        _make_session(db, f"s{i}", cost=2.0)
    report = engine.generate(days=7)
    assert "runaway_session" not in _codes(report)


def test_spawn_burst_detected(db, engine):
    # Background rate: one session in each of 5 prior hours.
    for h in range(2, 7):
        _make_session(db, f"bg{h}", source="telegram", days_ago=h / 24)
    # Burst: 35 sessions inside the most recent hour.
    for i in range(35):
        _make_session(db, f"burst{i}", source="telegram")
    report = engine.generate(days=7)
    assert "spawn_burst" in _codes(report)
    anomaly = next(a for a in report["anomalies"]
                   if a["code"] == "spawn_burst")
    assert anomaly["metrics"]["source"] == "telegram"
    assert anomaly["metrics"]["worst_count"] >= 35


def test_steady_high_rate_source_not_flagged(db, engine):
    # A cron-like source firing 31 sessions/hour, every hour: high but
    # steady — median equals the rate, so no burst.
    sid = 0
    for h in range(6):
        for i in range(31):
            sid += 1
            _make_session(db, f"c{sid}", source="cron", days_ago=h / 24,
                          messages=2)
    report = engine.generate(days=7)
    assert "spawn_burst" not in _codes(report)


def test_ghost_sessions_detected(db, engine):
    for i in range(6):
        _make_session(db, f"ghost{i}", messages=1)
    for i in range(6):
        _make_session(db, f"live{i}", messages=6)
    report = engine.generate(days=7)
    assert "ghost_sessions" in _codes(report)


def test_cache_unused_detected(db, engine):
    for i in range(5):
        _make_session(db, f"s{i}", input_tokens=300_000, output_tokens=1000)
    report = engine.generate(days=7)
    assert "cache_unused" in _codes(report)


def test_model_concentration_detected(db, engine):
    for i in range(10):
        _make_session(db, f"big{i}", model="test/expensive", cost=2.0)
    _make_session(db, "small", model="test/cheap", cost=0.05)
    report = engine.generate(days=7)
    assert "model_concentration" in _codes(report)
    anomaly = next(a for a in report["anomalies"]
                   if a["code"] == "model_concentration")
    assert anomaly["metrics"]["model"] == "test/expensive"


def test_cost_spike_detected(db, engine):
    # Prior half of an 8-day window: cheap days; recent half: expensive.
    for d in range(7, 3, -1):
        _make_session(db, f"old{d}", days_ago=d, cost=0.50)
    for d in range(3, -1, -1):
        _make_session(db, f"new{d}", days_ago=d, cost=4.0)
    report = engine.generate(days=8)
    assert "cost_spike" in _codes(report)


# =============================================================================
# Report shape + rendering
# =============================================================================

def test_report_contains_base_insights(db, engine):
    _make_session(db, "s1")
    report = engine.generate(days=7)
    assert report["overview"]["total_sessions"] == 1
    assert "models" in report
    assert "platforms" in report
    assert len(report["daily"]) == 1
    assert report["daily"][0]["sessions"] == 1


def test_anomalies_sorted_by_severity(db, engine):
    # Critical (tool failures) + info (cache unused) together.
    for i in range(5):
        _make_session(db, f"s{i}", input_tokens=300_000)
    for i in range(6):
        db.append_message("s0", role="tool", content="Error: nope",
                          tool_name="exec")
    report = engine.generate(days=7)
    severities = [a["severity"] for a in report["anomalies"]]
    ranks = [{"info": 0, "warning": 1, "critical": 2}[s] for s in severities]
    assert ranks == sorted(ranks, reverse=True)


def test_format_terminal_smoke(db, engine):
    _make_session(db, "s1")
    for i in range(6):
        db.append_message("s1", role="tool", content="Error: down",
                          tool_name="web_fetch")
    report = engine.generate(days=7)
    text = engine.format_terminal(report, color=False)
    assert "STATE OF THE MATRIX" in text
    assert "ANOMALIES" in text
    assert "web_fetch" in text
    # No ANSI escapes when color is off.
    assert "\033[" not in text


def test_format_json_round_trips(db, engine):
    import json
    _make_session(db, "s1")
    report = engine.generate(days=7)
    parsed = json.loads(engine.format_json(report))
    assert parsed["verdict"] == report["verdict"]
