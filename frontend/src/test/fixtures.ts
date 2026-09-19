import { vi } from "vitest";
import type { SchemaForgeApi } from "../api/client";
import type { DatabaseInfo, HealthResponse, QueryResponse } from "../api/types";

export const DB_DEMO: DatabaseInfo = { id: "demo", dialect: "sqlite", description: "Deterministic demo database" };
export const DB_SALES: DatabaseInfo = { id: "sales", dialect: "sqlite", description: null };

export const HEALTH_READY: HealthResponse = {
  status: "ok",
  model_runtime: {
    runtime: "llama_cpp",
    configured: true,
    deployment_mode: "hot_lora",
    base_gguf: "base-Q4_K_M.gguf",
    lora_gguf: "lora-1518-f16.gguf",
    context_size: 8192,
    sampling: "greedy",
  },
};

export const HEALTH_UNCONFIGURED: HealthResponse = {
  status: "ok",
  model_runtime: { runtime: "unavailable", configured: false, reason: "Missing environment variable(s): SCHEMAFORGE_LLAMA_EXE" },
};

export function makeResponse(over: Partial<QueryResponse> = {}): QueryResponse {
  return {
    request_id: "req-0123456789abcdef",
    database_id: "demo",
    status: "ok",
    generated_sql: "SELECT COUNT(emp_id) FROM employees",
    safety: { allowed: true, reasons: [] },
    result: { columns: ["COUNT(emp_id)"], rows: [[12]], returned_row_count: 1, truncated: false, max_rows: 500, elapsed_ms: 1.5 },
    error: null,
    timings: { schema_ms: 2, model_ms: 6400, safety_ms: 0.5, execution_ms: 1.5, total_ms: 6410 },
    model: {
      runtime: "llama_cpp",
      configured: true,
      deployment_mode: "hot_lora",
      base_gguf: "base-Q4_K_M.gguf",
      lora_gguf: "lora-1518-f16.gguf",
      input_tokens: 412,
      output_tokens: 9,
      generation_latency_ms: 6400,
      generated_tokens_per_second: 10.9,
    },
    prompt_sha256: "a".repeat(64),
    dialect: "sqlite",
    ...over,
  };
}

export function failureResponse(status: QueryResponse["status"], stage: "model" | "safety" | "execution" | "schema", code: string, over: Partial<QueryResponse> = {}): QueryResponse {
  return makeResponse({ status, result: null, error: { stage, code, message: `${code} message` }, ...over });
}

export function makeApi(over: Partial<SchemaForgeApi> = {}): SchemaForgeApi & { health: ReturnType<typeof vi.fn>; databases: ReturnType<typeof vi.fn>; query: ReturnType<typeof vi.fn> } {
  return {
    health: vi.fn().mockResolvedValue(HEALTH_READY),
    databases: vi.fn().mockResolvedValue([DB_DEMO, DB_SALES]),
    query: vi.fn().mockResolvedValue(makeResponse()),
    ...over,
  } as never;
}
