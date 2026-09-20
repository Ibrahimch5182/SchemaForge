import { useEffect, useId, useRef, useState } from "react";
import type { HealthState } from "../state/useHealth";
import { SERVING } from "../data/serving";
import { inferenceLabel, runtimeLabel } from "../lib/modelMeta";
import { Chevron, Refresh } from "./icons";

interface Props {
  state: HealthState;
  onRefresh: () => void;
}

type Tone = "loading" | "ready" | "warn" | "down";

function describe(state: HealthState): { tone: Tone; label: string } {
  if (state.status === "loading") return { tone: "loading", label: "Checking backend…" };
  if (state.status === "offline") return { tone: "down", label: "Backend offline" };
  const rt = state.health.model_runtime;
  if (!rt.configured) return { tone: "warn", label: "Model not configured" };
  if (rt.serving?.state === "loading") return { tone: "warn", label: "Model server loading" };
  if (rt.serving?.state === "unreachable") return { tone: "warn", label: "Model server unreachable" };
  if (rt.ready === false) return { tone: "warn", label: rt.runtime_mode === "persistent_server" ? "Model server not ready" : "Model files missing" };
  if (rt.availability && rt.availability.state !== "idle") return { tone: "warn", label: rt.availability.state === "saturated" ? "Model server saturated" : "Model server busy" };
  return { tone: "ready", label: `${SERVING.host} model server ready` };
}

/** Backend/model availability: a compact pill that opens a details popover. */
export function StatusPill({ state, onRefresh }: Props) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const root = useRef<HTMLDivElement>(null);
  const { tone, label } = describe(state);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const rt = state.status === "online" ? state.health.model_runtime : null;

  return (
    <div className="status-pill-wrap" ref={root}>
      <button type="button" className={`status-pill tone-${tone}`} aria-expanded={open} aria-controls={panelId} onClick={() => setOpen((o) => !o)}>
        <span className="status-dot" aria-hidden="true" />
        <span>{label}</span>
        <Chevron size={14} className="status-chev" />
      </button>
      {open && (
        <div id={panelId} className="status-panel" role="region" aria-label="Backend status details">
          <dl>
            <div>
              <dt>Backend</dt>
              <dd>{state.status === "online" ? "Online" : state.status === "offline" ? "Unreachable" : "Checking…"}</dd>
            </div>
            {rt && (
              <>
                <div>
                  <dt>Hosting</dt>
                  <dd>{SERVING.host} · {SERVING.api}</dd>
                </div>
                <div>
                  <dt>Runtime</dt>
                  <dd>{runtimeLabel(rt.runtime, rt.runtime_mode)}</dd>
                </div>
                {inferenceLabel(rt.n_gpu_layers) && (
                  <div>
                    <dt>Inference</dt>
                    <dd>{inferenceLabel(rt.n_gpu_layers)}</dd>
                  </div>
                )}
                {rt.deployment_mode && (
                  <div>
                    <dt>Deployment</dt>
                    <dd>{rt.deployment_mode === "hot_lora" ? `${SERVING.model.replace(" + LoRA", "")} · ${SERVING.quantization} base + runtime LoRA` : rt.deployment_mode}</dd>
                  </div>
                )}
                {rt.base_gguf && (
                  <div>
                    <dt>Base model</dt>
                    <dd className="mono">{rt.base_gguf}</dd>
                  </div>
                )}
                {rt.lora_gguf && (
                  <div>
                    <dt>LoRA</dt>
                    <dd className="mono">{rt.lora_gguf}</dd>
                  </div>
                )}
                {rt.checks && (
                  <div>
                    <dt>Artifacts</dt>
                    <dd>{Object.entries(rt.checks).map(([k, ok]) => `${k.replace("_gguf", "").replace("_", " ")} ${ok ? "✓" : "✗"}`).join(" · ")}</dd>
                  </div>
                )}
                {rt.availability && (
                  <div>
                    <dt>Availability</dt>
                    <dd>
                      {rt.availability.state} · {rt.availability.running}/{rt.availability.max_concurrent} running, {rt.availability.waiting}/{rt.availability.max_waiting} waiting
                    </dd>
                  </div>
                )}
                {rt.context_size !== undefined && (
                  <div>
                    <dt>Context</dt>
                    <dd>{rt.context_size.toLocaleString("en-US")} tokens</dd>
                  </div>
                )}
                {rt.sampling && (
                  <div>
                    <dt>Decoding</dt>
                    <dd>{rt.sampling}</dd>
                  </div>
                )}
                {!rt.configured && rt.reason && (
                  <div>
                    <dt>Reason</dt>
                    <dd>{rt.reason}</dd>
                  </div>
                )}
              </>
            )}
          </dl>
          <button type="button" className="btn btn-ghost btn-sm" onClick={onRefresh}>
            <Refresh size={14} /> Re-check
          </button>
        </div>
      )}
    </div>
  );
}
