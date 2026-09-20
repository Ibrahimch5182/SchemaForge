/** Typed, defensive read of the backend's free-form `model` metadata object. */
export interface ModelMeta {
  runtime: string | null;
  runtimeMode: string | null;
  nGpuLayers: number | null;
  deploymentMode: string | null;
  baseGguf: string | null;
  loraGguf: string | null;
  tokensPerSecond: number | null;
  promptTokensPerSecond: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
  generationMs: number | null;
  peakRssBytes: number | null;
  contextSize: number | null;
}

const s = (v: unknown) => (typeof v === "string" ? v : null);
const n = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function readModelMeta(model: Record<string, unknown>): ModelMeta {
  return {
    runtime: s(model.runtime),
    runtimeMode: s(model.runtime_mode),
    nGpuLayers: n(model.n_gpu_layers),
    deploymentMode: s(model.deployment_mode),
    baseGguf: s(model.base_gguf),
    loraGguf: s(model.lora_gguf),
    tokensPerSecond: n(model.generated_tokens_per_second),
    promptTokensPerSecond: n(model.prompt_tokens_per_second),
    inputTokens: n(model.input_tokens),
    outputTokens: n(model.output_tokens),
    generationMs: n(model.generation_latency_ms),
    peakRssBytes: n(model.peak_rss_bytes),
    contextSize: n(model.context_size),
  };
}

/** Runtime name for display. `llama_server` is the persistent llama.cpp HTTP server used in production. */
export function runtimeLabel(runtime: string | null | undefined, mode?: string | null): string {
  if (runtime === "llama_server" || mode === "persistent_server") return "llama.cpp · persistent server";
  if (runtime === "llama_cpp") return "llama.cpp · per-request process";
  return runtime ?? "—";
}

/** CPU vs GPU offload, only when the backend reported its layer offload count. */
export function inferenceLabel(nGpuLayers: number | null | undefined): string | null {
  if (nGpuLayers === null || nGpuLayers === undefined) return null;
  return nGpuLayers === 0 ? "CPU" : `GPU offload (${nGpuLayers} layers)`;
}
