/**
 * Where each part of SchemaForge runs. Product copy, not measurements: the values are the
 * deployment's fixed architecture (see docs/PHASE11.md). Live availability always comes from the
 * backend's /health, never from here.
 */
export const SERVING = {
  frontend: "Vercel",
  host: "AWS EC2",
  api: "FastAPI + HTTPS",
  runtime: "llama.cpp",
  serving: "Persistent",
  model: "Qwen3-4B + LoRA",
  quantization: "Q4_K_M",
  inference: "CPU",
} as const;

/** The request path, in order, for the compact pipeline chips. */
export const SERVING_PATH = ["Vercel", "HTTPS", "AWS EC2", "FastAPI", "llama.cpp", "Qwen3-4B + LoRA"] as const;
