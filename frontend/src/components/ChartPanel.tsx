import { useId, useState } from "react";
import type { ExecutionResult } from "../api/types";
import { extractSeries, type ChartPlan, type ChartPoint } from "../lib/chartability";
import { formatNumber } from "../lib/format";

interface Props {
  result: ExecutionResult;
  plan: ChartPlan;
}

/** A lightweight chart drawn from the very rows shown in the table (no extra request, no model). */
export function ChartPanel({ result, plan }: Props) {
  const selectId = useId();
  const [measure, setMeasure] = useState(plan.measureColumns[0]!);
  const series = extractSeries(result, plan.labelColumn, measure);
  const measureName = result.columns[measure] ?? "value";
  const labelName = result.columns[plan.labelColumn] ?? "label";
  const summary = `${plan.kind === "line" ? "Line" : "Bar"} chart of ${measureName} by ${labelName}, ${series.length} points.`;

  return (
    <div className="chart-panel">
      <div className="chart-head">
        <p className="chart-caption">
          <strong>{measureName}</strong> by <strong>{labelName}</strong>
          <span className="muted"> · drawn from the returned rows</span>
        </p>
        {plan.measureColumns.length > 1 && (
          <div className="chart-select">
            <label htmlFor={selectId} className="sr-only">
              Measure to plot
            </label>
            <select id={selectId} className="select" value={measure} onChange={(e) => setMeasure(Number(e.target.value))}>
              {plan.measureColumns.map((i) => (
                <option key={i} value={i}>
                  {result.columns[i]}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      {plan.kind === "line" ? <LineChart series={series} summary={summary} /> : <BarChart series={series} summary={summary} />}
    </div>
  );
}

function domain(series: ChartPoint[]): { min: number; max: number } {
  const vals = series.map((p) => p.value).filter((v): v is number => v !== null);
  return { min: Math.min(0, ...vals), max: Math.max(0, ...vals) };
}

export function BarChart({ series, summary }: { series: ChartPoint[]; summary: string }) {
  const { min, max } = domain(series);
  const span = max - min || 1;
  const zero = ((0 - min) / span) * 100;
  return (
    <div role="img" aria-label={summary} className="bars">
      {series.map((p, i) => {
        const pct = p.value === null ? 0 : ((p.value - min) / span) * 100;
        const left = Math.min(zero, pct);
        const width = Math.abs(pct - zero);
        return (
          <div className="bar-row" key={i}>
            <span className="bar-label" title={p.label}>
              {p.label}
            </span>
            <span className="bar-track">
              <span className={`bar-fill ${p.value !== null && p.value < 0 ? "is-neg" : ""}`} style={{ left: `${left}%`, width: `${Math.max(width, p.value === 0 ? 0 : 0.6)}%`, animationDelay: `${Math.min(i, 12) * 35}ms` }} />
            </span>
            <span className="bar-value">{p.value === null ? "—" : formatNumber(p.value)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function LineChart({ series, summary }: { series: ChartPoint[]; summary: string }) {
  const W = 640;
  const H = 240;
  const pad = { l: 44, r: 16, t: 16, b: 32 };
  const { min, max } = domain(series);
  const span = max - min || 1;
  const x = (i: number) => pad.l + (series.length === 1 ? 0 : (i / (series.length - 1)) * (W - pad.l - pad.r));
  const y = (v: number) => pad.t + (1 - (v - min) / span) * (H - pad.t - pad.b);
  const pts = series.map((p, i) => (p.value === null ? null : ([x(i), y(p.value)] as const)));
  const path = pts.reduce<string>((d, p, i) => (p ? `${d}${d === "" || pts[i - 1] === null ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)} ` : d), "");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="linechart" role="img" aria-label={summary}>
      {[0, 0.5, 1].map((t) => {
        const v = min + span * t;
        const yy = y(v);
        return (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={yy} y2={yy} className="grid" />
            <text x={pad.l - 8} y={yy + 4} textAnchor="end" className="axis">
              {formatNumber(Number(v.toFixed(2)))}
            </text>
          </g>
        );
      })}
      <path d={path} className="line" fill="none" />
      {pts.map((p, i) =>
        p ? (
          <circle key={i} cx={p[0]} cy={p[1]} r={3.5} className="dot">
            <title>{`${series[i]?.label}: ${series[i]?.value}`}</title>
          </circle>
        ) : null,
      )}
      <text x={pad.l} y={H - 8} className="axis">
        {series[0]?.label}
      </text>
      <text x={W - pad.r} y={H - 8} textAnchor="end" className="axis">
        {series[series.length - 1]?.label}
      </text>
    </svg>
  );
}
