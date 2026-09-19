import { useMemo, useState } from "react";
import type { QueryResponse } from "../api/types";
import { analyzeChartability } from "../lib/chartability";
import { ChartPanel } from "./ChartPanel";
import { ChartIcon, Check, Info, Lock, Shield, TableIcon } from "./icons";
import { MetaStrip } from "./MetaStrip";
import { ResultTable } from "./ResultTable";
import { SqlBlock } from "./SqlBlock";

/** A successful query: SQL artifact, safety/execution state, table (+ optional chart), metadata. */
export function ResultView({ response }: { response: QueryResponse }) {
  const result = response.result;
  const decision = useMemo(() => (result ? analyzeChartability(result) : null), [result]);
  const [view, setView] = useState<"table" | "chart">("table");
  if (!result) return null;

  const chartable = decision?.eligible === true ? decision : null;
  const rowsLabel = `${result.returned_row_count.toLocaleString("en-US")} ${result.returned_row_count === 1 ? "row" : "rows"}`;

  return (
    <div className="result-view">
      <div className="status-banner is-ok" role="status">
        <Check size={18} />
        <div>
          <strong>Query executed successfully</strong>
          <span>Safety verified, checked against the database schema, and run read-only.</span>
        </div>
      </div>

      <p className="trust-note" role="note" aria-label="What is and isn't verified">
        <Info size={15} />
        <span>
          <strong>Verified:</strong> safe, valid for this database, read-only. <strong>Not verified:</strong> that this answers your question. Review the SQL before relying on the result.
        </span>
      </p>

      {response.generated_sql && (
        <SqlBlock
          sql={response.generated_sql}
          badge={
            <span className="badges">
              <span className="badge badge-ok">
                <Shield size={13} /> Safety verified
              </span>
              <span className="badge badge-neutral">
                <Lock size={13} /> Read-only execution
              </span>
            </span>
          }
        />
      )}

      <section className="panel result-panel" aria-labelledby="result-heading">
        <div className="panel-head">
          <h2 id="result-heading" className="panel-title">
            Result <span className="count-chip">{result.truncated ? `first ${rowsLabel}` : rowsLabel}</span>
          </h2>
          {chartable && (
            <div className="segmented" role="group" aria-label="Result view">
              <button type="button" aria-pressed={view === "table"} onClick={() => setView("table")}>
                <TableIcon size={15} /> Table
              </button>
              <button type="button" aria-pressed={view === "chart"} onClick={() => setView("chart")}>
                <ChartIcon size={15} /> Chart
              </button>
            </div>
          )}
        </div>

        {result.truncated && (
          <div className="truncation-note" role="note" aria-label="Results truncated">
            <strong>Results truncated.</strong> Only the first {result.max_rows.toLocaleString("en-US")} rows are shown; more rows exist. The total isn't known. Add a filter or LIMIT to
            narrow the question.
          </div>
        )}

        {view === "chart" && chartable ? <ChartPanel result={result} plan={chartable} /> : <ResultTable result={result} />}

        {decision && !decision.eligible && result.rows.length > 1 && <p className="chart-note muted small">No chart offered: {decision.reason}</p>}
      </section>

      <MetaStrip response={response} />
    </div>
  );
}
