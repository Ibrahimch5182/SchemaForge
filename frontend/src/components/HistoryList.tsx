import { classifyOutcome } from "../lib/failure";
import { clockTime } from "../lib/format";
import type { HistoryEntry } from "../state/history";
import { History } from "./icons";

interface Props {
  entries: HistoryEntry[];
  activeId: string | null;
  onOpen: (entry: HistoryEntry) => void;
  onClear: () => void;
  disabled?: boolean;
}

const LABEL: Record<string, string> = {
  ok: "Succeeded",
  unsafe_sql: "Blocked",
  timeout: "Timed out",
  client_timeout: "Timed out",
  model_error: "Model error",
  model_unavailable: "Model unavailable",
  execution_error: "SQL error",
  backend_unavailable: "Backend offline",
  model_busy: "Model busy",
  model_timeout: "Model timed out",
  malformed_output: "Unusable output",
  invalid_for_database: "Invalid for database",
  cancelled: "Cancelled",
};

/** Session-local history: revisit any earlier question and its result. Never leaves the browser tab. */
export function HistoryList({ entries, activeId, onOpen, onClear, disabled }: Props) {
  if (entries.length === 0) return null;
  return (
    <section className="panel" aria-labelledby="history-heading">
      <div className="panel-head">
        <h2 id="history-heading" className="panel-title">
          <History size={16} /> This session
        </h2>
        <button type="button" className="link-button" onClick={onClear}>
          Clear
        </button>
      </div>
      <ul className="history-list">
        {entries.map((e) => {
          const c = classifyOutcome(e.outcome);
          const kind = c.ok ? "ok" : c.failure.kind;
          return (
            <li key={e.id}>
              <button type="button" className={`history-item ${e.id === activeId ? "is-active" : ""}`} onClick={() => onOpen(e)} disabled={disabled}>
                <span className={`hdot ${c.ok ? "ok" : `bad tone-${c.failure.tone}`}`} aria-hidden="true" />
                <span className="history-q">{e.request.question}</span>
                <span className="history-meta">
                  {e.request.database_id} · {LABEL[kind] ?? "Failed"} · {clockTime(e.at)}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
