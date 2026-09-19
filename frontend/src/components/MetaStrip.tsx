import type { QueryResponse } from "../api/types";
import { formatMs, formatRate, shortId } from "../lib/format";
import { readModelMeta } from "../lib/modelMeta";
import { CopyButton } from "./CopyButton";
import { Clock, Cpu, Layers } from "./icons";

/** Compact run metadata beneath the result; the full breakdown sits behind a disclosure. */
export function MetaStrip({ response }: { response: QueryResponse }) {
  const m = readModelMeta(response.model);
  const t = response.timings;
  const r = response.result;
  const rate = formatRate(m.tokensPerSecond);

  const stats: { label: string; value: string; strong?: boolean }[] = [
    { label: "Total", value: formatMs(t.total_ms), strong: true },
    { label: "Generation", value: formatMs(m.generationMs ?? t.model_ms) },
    { label: "Execution", value: formatMs(r?.elapsed_ms ?? t.execution_ms) },
  ];

  return (
    <div className="meta">
      <ul className="meta-strip" aria-label="Run summary">
        {stats.map((s) => (
          <li key={s.label} className={s.strong ? "is-strong" : ""}>
            <Clock size={14} />
            <span className="meta-label">{s.label}</span>
            <span className="meta-value">{s.value}</span>
          </li>
        ))}
        {r && (
          <li>
            <Layers size={14} />
            <span className="meta-label">Rows</span>
            <span className="meta-value">
              {r.returned_row_count.toLocaleString("en-US")}
              {r.truncated ? "+" : ""}
            </span>
          </li>
        )}
        {rate && (
          <li>
            <Cpu size={14} />
            <span className="meta-label">Speed</span>
            <span className="meta-value">{rate}</span>
          </li>
        )}
      </ul>

      <details className="meta-details">
        <summary>Runtime details</summary>
        <dl className="meta-grid">
          <div>
            <dt>Runtime</dt>
            <dd>{m.runtime === "llama_cpp" ? "llama.cpp · CPU" : (m.runtime ?? "—")}</dd>
          </div>
          <div>
            <dt>Deployment</dt>
            <dd>{m.deploymentMode === "hot_lora" ? "Q4_K_M base + runtime LoRA" : (m.deploymentMode ?? "—")}</dd>
          </div>
          {m.baseGguf && (
            <div>
              <dt>Base model</dt>
              <dd className="mono">{m.baseGguf}</dd>
            </div>
          )}
          {m.loraGguf && (
            <div>
              <dt>LoRA</dt>
              <dd className="mono">{m.loraGguf}</dd>
            </div>
          )}
          <div>
            <dt>Tokens in / out</dt>
            <dd>
              {m.inputTokens ?? "—"} / {m.outputTokens ?? "—"}
            </dd>
          </div>
          <div>
            <dt>Schema</dt>
            <dd>{formatMs(t.schema_ms)}</dd>
          </div>
          <div>
            <dt>Safety check</dt>
            <dd>{formatMs(t.safety_ms)}</dd>
          </div>
          <div>
            <dt>Schema validation</dt>
            <dd>{formatMs(t.preflight_ms)}</dd>
          </div>
          <div>
            <dt>Row cap</dt>
            <dd>{r ? r.max_rows.toLocaleString("en-US") : "—"}</dd>
          </div>
          <div>
            <dt>Request</dt>
            <dd className="mono request-id">
              {shortId(response.request_id, 12)}
              <CopyButton text={response.request_id} label="Copy request ID" compact />
            </dd>
          </div>
        </dl>
      </details>
    </div>
  );
}
