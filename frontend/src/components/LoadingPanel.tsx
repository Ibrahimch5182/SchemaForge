import { SERVING } from "../data/serving";
import { useElapsedSeconds } from "../lib/hooks";
import { Close } from "./icons";

interface Props {
  startedAt: number;
  onCancel: () => void;
}

const STEPS = ["Read the schema", "Generate SQL on the model server", "Check safety", "Run read-only"];

const mmss = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

/**
 * Long-running CPU inference on the remote model server. There is no real progress signal, so this shows
 * an indeterminate animation and a real elapsed clock, never a fake percentage.
 */
export function LoadingPanel({ startedAt, onCancel }: Props) {
  const elapsed = useElapsedSeconds(startedAt, true);
  return (
    <section className="panel loading-panel" role="status" aria-live="polite" aria-busy="true" aria-label="Generating SQL">
      <div className="forge-orb" aria-hidden="true">
        <span className="ring r1" />
        <span className="ring r2" />
        <span className="core" />
      </div>
      <div className="loading-copy">
        <h2>Forging your SQL…</h2>
        <p>
          Qwen3-4B (Q4_K_M + LoRA) is generating SQL on the <strong>{SERVING.host}</strong> model server, a persistent llama.cpp runtime doing CPU inference. This typically takes
          <strong>tens of seconds</strong>. There's no live progress signal, so this shows elapsed time rather than a percentage.
        </p>
        <div className="indeterminate" aria-hidden="true">
          <span />
        </div>
        <div className="loading-meta">
          <span className="elapsed" aria-label={`Elapsed ${elapsed} seconds`}>
            {mmss(elapsed)}
          </span>
          <ol className="steps" aria-label="What happens in a query">
            {STEPS.map((s, i) => (
              <li key={s}>
                <span className="step-n">{i + 1}</span>
                {s}
              </li>
            ))}
          </ol>
        </div>
        <button type="button" className="btn btn-ghost btn-sm" onClick={onCancel}>
          <Close size={14} /> Cancel request
        </button>
      </div>
    </section>
  );
}
