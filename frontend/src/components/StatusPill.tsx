import { useEffect, useId, useRef, useState } from "react";
import type { HealthState } from "../state/useHealth";
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
  if (rt.ready === false) return { tone: "warn", label: "Model files missing" };
  if (rt.availability && rt.availability.state !== "idle") return { tone: "warn", label: rt.availability.state === "saturated" ? "Model saturated" : "Model busy" };
  return { tone: "ready", label: "Local model ready" };
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
                  <dt>Runtime</dt>
                  <dd>{rt.runtime === "llama_cpp" ? "llama.cpp (CPU)" : rt.runtime}</dd>
                </div>
                {rt.deployment_mode && (
                  <div>
                    <dt>Deployment</dt>
                    <dd>{rt.deployment_mode === "hot_lora" ? "Q4_K_M base + runtime LoRA" : rt.deployment_mode}</dd>
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
