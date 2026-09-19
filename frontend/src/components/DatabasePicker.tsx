import type { DatabasesState } from "../state/useDatabases";
import { Alert, Database, Refresh } from "./icons";

interface Props {
  state: DatabasesState;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onReload: () => void;
  disabled?: boolean;
}

/** Registered logical databases only: the UI never sees or sends filesystem paths. */
export function DatabasePicker({ state, selectedId, onSelect, onReload, disabled }: Props) {
  return (
    <section className="panel" aria-labelledby="db-heading">
      <div className="panel-head">
        <h2 id="db-heading" className="panel-title">
          <Database size={16} /> Database
        </h2>
      </div>

      {state.status === "loading" && (
        <div className="skeleton-list" aria-busy="true" aria-label="Loading databases">
          <div className="skeleton" />
          <div className="skeleton short" />
        </div>
      )}

      {state.status === "error" && (
        <div className="inline-error" role="alert">
          <Alert size={18} />
          <div>
            <strong>{state.error.kind === "network" ? "Can't reach the backend" : "Couldn't load databases"}</strong>
            <p>{state.error.kind === "network" ? "Start the SchemaForge API, then try again." : "The backend returned an unexpected answer."}</p>
            <button type="button" className="btn btn-ghost btn-sm" onClick={onReload}>
              <Refresh size={14} /> Try again
            </button>
          </div>
        </div>
      )}

      {state.status === "ready" && state.databases.length === 0 && (
        <p className="muted small">No databases are registered on the backend yet. Add one to configs/backend.yaml.</p>
      )}

      {state.status === "ready" && state.databases.length > 0 && (
        <div role="radiogroup" aria-label="Registered databases" className="db-list">
          {state.databases.map((db) => {
            const selected = db.id === selectedId;
            return (
              <button
                key={db.id}
                type="button"
                role="radio"
                aria-checked={selected}
                disabled={disabled}
                className={`db-card ${selected ? "is-selected" : ""}`}
                onClick={() => onSelect(db.id)}
              >
                <span className="db-card-top">
                  <span className="db-name">{db.id}</span>
                  <span className="chip">{db.dialect}</span>
                </span>
                {db.description && <span className="db-desc">{db.description}</span>}
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}
