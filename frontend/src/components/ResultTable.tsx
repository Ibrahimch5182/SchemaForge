import { useState } from "react";
import type { ExecutionResult } from "../api/types";
import { formatCell } from "../lib/format";

const INITIAL_ROWS = 100;
const ROW_STEP = 200;

/**
 * The primary data view. Plain <table> semantics, sticky header and row index,
 * horizontal overflow inside a focusable region, and progressive row rendering
 * so a full max_rows result stays fast. Every cell is a text node.
 */
export function ResultTable({ result }: { result: ExecutionResult }) {
  const [visible, setVisible] = useState(INITIAL_ROWS);
  const { columns, rows } = result;

  if (rows.length === 0) {
    return (
      <div className="empty-result">
        <strong>No rows returned</strong>
        <p>The query ran successfully, but nothing matched.</p>
      </div>
    );
  }

  // A column is numeric when every non-null value is a number: its header aligns with its cells.
  const numeric = columns.map((_, c) => {
    const vals = rows.map((r) => r[c] ?? null).filter((v) => v !== null);
    return vals.length > 0 && vals.every((v) => typeof v === "number");
  });
  const shown = rows.slice(0, visible);
  const remaining = rows.length - shown.length;

  return (
    <div className="table-shell">
      {/* A scrollable region must be keyboard-focusable (axe: scrollable-region-focusable). */}
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */}
      <div className="tablewrap" role="region" aria-label="Query results table" tabIndex={0}>
        <table className="result-table">
          <thead>
            <tr>
              <th scope="col" className="idx">
                #
              </th>
              {columns.map((c, i) => (
                <th key={i} scope="col" title={c} className={numeric[i] ? "num" : undefined}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, r) => (
              <tr key={r}>
                <th scope="row" className="idx">
                  {r + 1}
                </th>
                {columns.map((_, c) => {
                  const cell = formatCell(row[c] ?? null);
                  return (
                    <td key={c} className={`cell-${cell.kind}`} title={cell.text.length > 40 ? cell.text : undefined}>
                      {cell.kind === "null" || cell.kind === "boolean" ? <span className={`pill pill-${cell.kind}`}>{cell.text}</span> : cell.text}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {remaining > 0 && (
        <div className="table-more">
          <span className="muted small">
            Showing {shown.length.toLocaleString("en-US")} of {rows.length.toLocaleString("en-US")} returned rows
          </span>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => setVisible((v) => v + ROW_STEP)}>
            Show {Math.min(ROW_STEP, remaining).toLocaleString("en-US")} more
          </button>
        </div>
      )}
    </div>
  );
}
