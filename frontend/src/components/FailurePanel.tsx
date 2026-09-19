import type { QueryResponse } from "../api/types";
import type { Failure } from "../lib/failure";
import { Alert, Clock, Refresh, ShieldOff } from "./icons";
import { SqlBlock } from "./SqlBlock";

interface Props {
  failure: Failure;
  response?: QueryResponse;
  onRetry?: () => void;
}

/** Every failure mode gets intentional copy, a next step, and (where present) the SQL involved. */
export function FailurePanel({ failure, response, onRetry }: Props) {
  const Icon = failure.kind === "unsafe_sql" ? ShieldOff : failure.kind === "timeout" || failure.kind === "client_timeout" ? Clock : Alert;
  const sql = response?.generated_sql;

  return (
    <div className="failure-view">
      <div className={`failure-card tone-${failure.tone}`} role="alert" data-failure={failure.kind}>
        <div className="failure-icon">
          <Icon size={22} />
        </div>
        <div className="failure-body">
          <h2>{failure.title}</h2>
          <p>{failure.message}</p>
          {failure.details && failure.details.length > 0 && (
            <ul className="failure-details">
              {failure.details.map((d, i) => (
                <li key={i}>{d}</li>
              ))}
            </ul>
          )}
          <p className="failure-hint">{failure.hint}</p>
          <div className="failure-foot">
            {failure.retryable && onRetry && (
              <button type="button" className="btn btn-secondary btn-sm" onClick={onRetry}>
                <Refresh size={14} /> Try again
              </button>
            )}
            {(failure.code || failure.requestId) && (
              <span className="failure-ids mono">
                {failure.code && <span>{failure.code}</span>}
                {failure.requestId && <span>req {failure.requestId.slice(0, 12)}</span>}
              </span>
            )}
          </div>
        </div>
      </div>

      {sql && (
        <SqlBlock
          sql={sql}
          title={failure.kind === "unsafe_sql" ? "Blocked SQL (never executed)" : "Generated SQL"}
          variant={failure.kind === "unsafe_sql" ? "blocked" : "default"}
        />
      )}
    </div>
  );
}
