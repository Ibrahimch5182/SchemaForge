import { useId, useState, type FormEvent, type KeyboardEvent } from "react";
import { ArrowRight, Chevron, Close, Sparkle } from "./icons";

export const MAX_QUESTION = 2000;
export const MAX_CONTEXT = 4000;

export interface Draft {
  question: string;
  context: string;
}

export interface ExampleQuestion {
  label: string;
  question: string;
  context?: string;
}

interface Props {
  draft: Draft;
  onChange: (d: Draft) => void;
  onSubmit: () => void;
  onCancel: () => void;
  running: boolean;
  /** Why submitting is currently impossible (shown under the button), or null. */
  blockedReason: string | null;
  examples: ExampleQuestion[];
}

export function QueryForm({ draft, onChange, onSubmit, onCancel, running, blockedReason, examples }: Props) {
  const qId = useId();
  const cId = useId();
  const hintId = useId();
  const [contextOpen, setContextOpen] = useState(draft.context.length > 0);
  const trimmed = draft.question.trim();
  const tooLong = draft.question.length > MAX_QUESTION || draft.context.length > MAX_CONTEXT;
  const canSubmit = !running && trimmed.length > 0 && !tooLong && blockedReason === null;

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (canSubmit) onSubmit();
  };
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      submit();
    }
  };

  const showContext = contextOpen || draft.context.length > 0;

  return (
    <form className="panel query-form" onSubmit={submit} aria-label="Ask a question">
      <div className="field">
        <div className="field-head">
          <label htmlFor={qId} className="field-label">
            Your question
          </label>
          <span className={`counter ${draft.question.length > MAX_QUESTION ? "is-over" : draft.question.length > MAX_QUESTION * 0.9 ? "is-near" : ""}`} aria-live="off">
            {draft.question.length.toLocaleString("en-US")} / {MAX_QUESTION.toLocaleString("en-US")}
          </span>
        </div>
        <textarea
          id={qId}
          className="textarea"
          rows={3}
          value={draft.question}
          placeholder="e.g. What is the total salary of employees in the Engineering department?"
          onChange={(e) => onChange({ ...draft, question: e.target.value })}
          onKeyDown={onKeyDown}
          disabled={running}
          aria-describedby={hintId}
          spellCheck
        />
      </div>

      {examples.length > 0 && trimmed.length === 0 && !running && (
        <div className="examples" role="group" aria-label="Example questions">
          <span className="examples-label">
            <Sparkle size={14} /> Try
          </span>
          {examples.map((ex) => (
            <button key={ex.label} type="button" className="chip chip-button" onClick={() => onChange({ question: ex.question, context: ex.context ?? "" })}>
              {ex.label}
            </button>
          ))}
        </div>
      )}

      <div className="context-block">
        {!showContext ? (
          <button type="button" className="link-button" aria-expanded={false} onClick={() => setContextOpen(true)}>
            <Chevron size={14} className="rot-270" /> Add business context <span className="muted">(optional)</span>
          </button>
        ) : (
          <div className="field">
            <div className="field-head">
              <label htmlFor={cId} className="field-label">
                Business context <span className="muted">(optional)</span>
              </label>
              <span className={`counter ${draft.context.length > MAX_CONTEXT ? "is-over" : ""}`}>
                {draft.context.length.toLocaleString("en-US")} / {MAX_CONTEXT.toLocaleString("en-US")}
              </span>
            </div>
            <textarea
              id={cId}
              className="textarea"
              rows={2}
              value={draft.context}
              placeholder="Definitions or hints that clarify columns and terms, e.g. “Revenue means price × quantity.”"
              onChange={(e) => onChange({ ...draft, context: e.target.value })}
              onKeyDown={onKeyDown}
              disabled={running}
            />
            {draft.context.length === 0 && (
              <button type="button" className="link-button" onClick={() => setContextOpen(false)}>
                <Close size={13} /> Hide
              </button>
            )}
          </div>
        )}
      </div>

      <div className="form-actions">
        {running ? (
          <button type="button" className="btn btn-secondary" onClick={onCancel}>
            <Close size={16} /> Cancel
          </button>
        ) : (
          <button type="submit" className="btn btn-primary" disabled={!canSubmit}>
            Run query <ArrowRight size={16} />
          </button>
        )}
        <p id={hintId} className="hint">
          {blockedReason ?? (tooLong ? "Shorten the question or context to continue." : <><kbd>Ctrl</kbd> / <kbd>⌘</kbd> + <kbd>Enter</kbd> to run · read-only, single query</>)}
        </p>
      </div>
    </form>
  );
}
