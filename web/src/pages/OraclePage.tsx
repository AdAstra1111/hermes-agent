import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Eye, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type {
  OracleAnomaly,
  OracleReport,
  OracleSeverity,
  OracleVerdict,
} from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MatrixRain } from "@/components/MatrixRain";
import { usePageHeader } from "@/contexts/usePageHeader";

const PERIODS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
] as const;

const CHART_HEIGHT_PX = 120;

// Phosphor palette — the page keeps its Matrix identity under any
// dashboard theme, so colors are pinned rather than token-derived.
const PHOSPHOR = "#00ff66";
const PHOSPHOR_DIM = "#39b36b";
const PHOSPHOR_FAINT = "rgba(0, 255, 102, 0.18)";

const VERDICT_STYLES: Record<
  OracleVerdict,
  { color: string; label: string; quote: string }
> = {
  stable: {
    color: PHOSPHOR,
    label: "SYSTEM STABLE",
    quote:
      "All constructs operating within expected parameters. There is no spoon.",
  },
  degraded: {
    color: "#ffd700",
    label: "ANOMALIES DETECTED",
    quote: "A glitch in the Matrix. It happens when they change something.",
  },
  unstable: {
    color: "#ff2244",
    label: "SYSTEM UNSTABLE",
    quote: "The anomaly is systemic. Intervention required.",
  },
};

const SEVERITY_COLORS: Record<OracleSeverity, string> = {
  info: PHOSPHOR_DIM,
  warning: "#ffd700",
  critical: "#ff2244",
};

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function StatBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span
        className="font-mono text-[10px] uppercase tracking-[0.2em]"
        style={{ color: PHOSPHOR_DIM }}
      >
        {label}
      </span>
      <span className="font-mono text-xl" style={{ color: PHOSPHOR }}>
        {value}
      </span>
    </div>
  );
}

function VerdictBanner({ report }: { report: OracleReport }) {
  const v = VERDICT_STYLES[report.verdict] ?? VERDICT_STYLES.stable;
  return (
    <Card style={{ borderColor: v.color }}>
      <CardContent className="flex flex-col gap-2 py-6">
        <div className="flex items-center gap-3">
          <span
            className="inline-block h-3 w-3 rounded-full"
            style={{
              backgroundColor: v.color,
              boxShadow: `0 0 12px ${v.color}`,
            }}
          />
          <span
            className="font-mono text-lg tracking-[0.25em]"
            style={{ color: v.color, textShadow: `0 0 8px ${v.color}66` }}
          >
            {v.label}
          </span>
        </div>
        <p className="font-mono text-xs italic" style={{ color: PHOSPHOR_DIM }}>
          “{v.quote}”
        </p>
      </CardContent>
    </Card>
  );
}

function AnomalyCard({
  anomaly,
  days,
  onChanged,
}: {
  anomaly: OracleAnomaly;
  days: number;
  onChanged: () => void;
}) {
  const color = SEVERITY_COLORS[anomaly.severity] ?? PHOSPHOR_DIM;
  const [busy, setBusy] = useState<"ack" | "dispatch" | null>(null);
  const [dispatchedJob, setDispatchedJob] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const toggleAck = async () => {
    setBusy("ack");
    setActionError(null);
    try {
      if (anomaly.acknowledged) {
        await api.unackOracleAnomaly(anomaly.key);
      } else {
        await api.ackOracleAnomaly(anomaly.key);
      }
      onChanged();
    } catch (err) {
      setActionError(String(err));
    } finally {
      setBusy(null);
    }
  };

  const dispatchFix = async () => {
    setBusy("dispatch");
    setActionError(null);
    try {
      const res = await api.dispatchOracleFix(anomaly.key, days);
      setDispatchedJob(res.job_id);
    } catch (err) {
      setActionError(String(err));
    } finally {
      setBusy(null);
    }
  };
  return (
    <div
      className="flex flex-col gap-1.5 border-l-2 py-2 pl-4"
      style={{ borderColor: color, opacity: anomaly.acknowledged ? 0.55 : 1 }}
    >
      <div className="flex flex-wrap items-center gap-2">
        {anomaly.acknowledged ? (
          <CheckCircle2
            className="h-3.5 w-3.5 shrink-0"
            style={{ color: PHOSPHOR_DIM }}
          />
        ) : (
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" style={{ color }} />
        )}
        <span className="font-mono text-sm font-semibold" style={{ color }}>
          {anomaly.title}
        </span>
        <Badge
          tone="secondary"
          className="font-mono text-[9px] uppercase tracking-widest"
        >
          {anomaly.severity}
        </Badge>
        {anomaly.acknowledged && (
          <Badge
            tone="secondary"
            className="font-mono text-[9px] uppercase tracking-widest"
          >
            acknowledged {(anomaly.acked_at ?? "").slice(0, 10)}
          </Badge>
        )}
        {anomaly.ack_stale && (
          <Badge
            tone="secondary"
            className="font-mono text-[9px] uppercase tracking-widest"
            style={{ color: SEVERITY_COLORS.warning }}
          >
            ack stale — worsened
          </Badge>
        )}
        <span
          className="font-mono text-[10px]"
          style={{ color: PHOSPHOR_DIM }}
        >
          {anomaly.key}
        </span>
      </div>
      <p className="font-mono text-xs" style={{ color: PHOSPHOR_DIM }}>
        {anomaly.detail}
      </p>
      {anomaly.ack_note && (
        <p
          className="font-mono text-xs italic"
          style={{ color: PHOSPHOR_DIM }}
        >
          ack note: {anomaly.ack_note}
        </p>
      )}
      <p className="font-mono text-xs" style={{ color: PHOSPHOR }}>
        → {anomaly.recommendation}
      </p>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <Button
          type="button"
          size="sm"
          outlined
          onClick={toggleAck}
          disabled={busy !== null}
          prefix={busy === "ack" ? <Spinner /> : undefined}
        >
          {anomaly.acknowledged ? "Unacknowledge" : "Acknowledge"}
        </Button>
        {dispatchedJob ? (
          <span
            className="font-mono text-[11px]"
            style={{ color: PHOSPHOR }}
          >
            ✔ agent dispatched — cron job {dispatchedJob}
          </span>
        ) : (
          <Button
            type="button"
            size="sm"
            outlined
            onClick={dispatchFix}
            disabled={busy !== null}
            prefix={busy === "dispatch" ? <Spinner /> : undefined}
          >
            Dispatch agent
          </Button>
        )}
        {actionError && (
          <span className="font-mono text-[11px] text-destructive">
            {actionError}
          </span>
        )}
      </div>
    </div>
  );
}

function TrafficChart({ report }: { report: OracleReport }) {
  if (report.daily.length === 0) return null;
  const maxTokens = Math.max(
    ...report.daily.map((d) => d.input_tokens + d.output_tokens),
    1,
  );
  return (
    <Card>
      <CardHeader>
        <CardTitle
          className="font-mono text-sm tracking-[0.2em]"
          style={{ color: PHOSPHOR }}
        >
          TRAFFIC — TOKENS / DAY
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className="flex items-end gap-[3px]"
          style={{ height: CHART_HEIGHT_PX }}
        >
          {report.daily.map((d) => {
            const total = d.input_tokens + d.output_tokens;
            const h = Math.max(
              Math.round((total / maxTokens) * CHART_HEIGHT_PX),
              total > 0 ? 2 : 0,
            );
            return (
              <div
                key={d.day}
                className="flex-1"
                title={`${d.day}: ${total.toLocaleString()} tokens, ${
                  d.sessions
                } sessions`}
                style={{
                  height: h,
                  backgroundColor: PHOSPHOR,
                  opacity: 0.35 + 0.65 * (total / maxTokens),
                  boxShadow: `0 0 6px ${PHOSPHOR_FAINT}`,
                }}
              />
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

function ProgramsTable({ report }: { report: OracleReport }) {
  if (report.models.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle
          className="font-mono text-sm tracking-[0.2em]"
          style={{ color: PHOSPHOR }}
        >
          PROGRAMS — BY MODEL
        </CardTitle>
      </CardHeader>
      <CardContent>
        <table className="w-full font-mono text-xs">
          <thead>
            <tr
              className="text-left uppercase tracking-widest"
              style={{ color: PHOSPHOR_DIM }}
            >
              <th className="pb-2 font-normal">Program</th>
              <th className="pb-2 text-right font-normal">Constructs</th>
              <th className="pb-2 text-right font-normal">Tokens</th>
            </tr>
          </thead>
          <tbody>
            {report.models.slice(0, 10).map((m) => (
              <tr
                key={m.model}
                className="border-t"
                style={{ borderColor: PHOSPHOR_FAINT, color: PHOSPHOR }}
              >
                <td className="max-w-0 truncate py-1.5 pr-4">{m.model}</td>
                <td className="py-1.5 text-right">{m.sessions}</td>
                <td className="py-1.5 text-right">
                  {formatTokens(m.total_tokens)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function JackInPoints({ report }: { report: OracleReport }) {
  if (report.platforms.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle
          className="font-mono text-sm tracking-[0.2em]"
          style={{ color: PHOSPHOR }}
        >
          JACK-IN POINTS — BY PLATFORM
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {report.platforms.map((p) => (
          <div
            key={p.platform}
            className="flex items-center justify-between font-mono text-xs"
            style={{ color: PHOSPHOR }}
          >
            <span className="uppercase tracking-widest">{p.platform}</span>
            <span style={{ color: PHOSPHOR_DIM }}>
              {p.sessions} sessions · {p.messages.toLocaleString()} msgs ·{" "}
              {formatTokens(p.total_tokens)} tok
            </span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export default function OraclePage() {
  const [days, setDays] = useState(7);
  const [report, setReport] = useState<OracleReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setAfterTitle, setEnd, setTitle } = usePageHeader();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getOracleReport(days)
      .then(setReport)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [days]);

  useLayoutEffect(() => {
    setTitle("THE ORACLE — STATE OF THE MATRIX");
    setAfterTitle(
      <span className="flex items-center gap-2">
        {loading && <Spinner className="shrink-0 text-base text-primary" />}
        {report && (
          <Badge tone="secondary" className="font-mono text-[10px]">
            ITERATION {report.iteration}
          </Badge>
        )}
      </span>,
    );
    setEnd(
      <div className="flex w-full min-w-0 flex-wrap items-center justify-end gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {PERIODS.map((p) => (
            <Button
              key={p.label}
              type="button"
              size="sm"
              outlined={days !== p.days}
              onClick={() => setDays(p.days)}
            >
              {p.label}
            </Button>
          ))}
        </div>
        <Button
          type="button"
          size="sm"
          outlined
          onClick={load}
          disabled={loading}
          prefix={loading ? <Spinner /> : <RefreshCw />}
        >
          Refresh
        </Button>
      </div>,
    );
    return () => {
      setTitle(null);
      setAfterTitle(null);
      setEnd(null);
    };
  }, [days, loading, load, report, setAfterTitle, setEnd, setTitle]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="relative flex flex-col gap-6">
      <div className="absolute -inset-4 overflow-hidden rounded-lg">
        <MatrixRain />
      </div>

      <div className="relative flex flex-col gap-6">
        {loading && !report && (
          <div className="flex flex-col items-center justify-center gap-3 py-24">
            <span style={{ color: PHOSPHOR }}>
              <Spinner className="text-2xl" />
            </span>
            <span
              className="font-mono text-xs tracking-[0.3em]"
              style={{ color: PHOSPHOR_DIM }}
            >
              CONSULTING THE ORACLE…
            </span>
          </div>
        )}

        {error && (
          <Card>
            <CardContent className="py-6">
              <p className="text-center font-mono text-sm text-destructive">
                {error}
              </p>
            </CardContent>
          </Card>
        )}

        {report && report.empty && (
          <Card>
            <CardContent className="flex flex-col items-center gap-2 py-16">
              <Eye className="h-6 w-6" style={{ color: PHOSPHOR_DIM }} />
              <p
                className="font-mono text-sm"
                style={{ color: PHOSPHOR_DIM }}
              >
                No sessions in the last {report.days} days. The construct is
                empty… for now.
              </p>
            </CardContent>
          </Card>
        )}

        {report && !report.empty && (
          <>
            <VerdictBanner report={report} />

            <Card>
              <CardContent className="grid grid-cols-2 gap-6 py-6 sm:grid-cols-3 lg:grid-cols-5">
                <StatBlock
                  label="Constructs"
                  value={String(report.overview.total_sessions ?? 0)}
                />
                <StatBlock
                  label="Transmissions"
                  value={(report.overview.total_messages ?? 0).toLocaleString()}
                />
                <StatBlock
                  label="Tokens"
                  value={formatTokens(report.overview.total_tokens ?? 0)}
                />
                <StatBlock
                  label={report.recorded_cost ? "Cost" : "Est. Cost"}
                  value={`$${(
                    report.recorded_cost || report.overview.estimated_cost || 0
                  ).toFixed(2)}`}
                />
                <StatBlock
                  label="Hours Jacked In"
                  value={(report.overview.total_hours ?? 0).toFixed(1)}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle
                  className="font-mono text-sm tracking-[0.2em]"
                  style={{ color: PHOSPHOR }}
                >
                  ANOMALIES — {report.anomalies.length} DETECTED
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                {report.anomalies.length === 0 ? (
                  <p
                    className="font-mono text-xs"
                    style={{ color: PHOSPHOR_DIM }}
                  >
                    No anomalies. Déjà vu has not been reported.
                  </p>
                ) : (
                  report.anomalies.map((a, i) => (
                    <AnomalyCard
                      key={`${a.key}-${i}`}
                      anomaly={a}
                      days={days}
                      onChanged={load}
                    />
                  ))
                )}
              </CardContent>
            </Card>

            <TrafficChart report={report} />

            <div className="grid gap-6 lg:grid-cols-2">
              <ProgramsTable report={report} />
              <JackInPoints report={report} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
