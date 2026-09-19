/** Typed, defensive read of the backend's free-form `model` metadata object. */
export interface ModelMeta {
  runtime: string | null;
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
