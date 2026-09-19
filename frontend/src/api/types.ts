/** Wire types. They mirror the Phase 8 backend contract (docs/PHASE8.md) exactly. */

export type QueryStatus = "ok" | "model_error" | "unsafe_sql" | "execution_error" | "schema_error";
export type Stage = "schema" | "model" | "safety" | "execution";

export const QUERY_STATUSES: readonly QueryStatus[] = ["ok", "model_error", "unsafe_sql", "execution_error", "schema_error"];
export const STAGES: readonly Stage[] = ["schema", "model", "safety", "execution"];

export interface DatabaseInfo {
  id: string;
  dialect: string;
  description: string | null;
}

export interface ModelRuntimeInfo {
  runtime: string;
  configured: boolean;
  deployment_mode?: string;
  base_gguf?: string | null;
  lora_gguf?: string | null;
  context_size?: number;
  max_new_tokens?: number;
  n_gpu_layers?: number;
  sampling?: string;
  reason?: string;
}

export interface HealthResponse {
  status: string;
  model_runtime: ModelRuntimeInfo;
}

export interface QueryRequest {
  database_id: string;
  question: string;
  business_context?: string | null;
}

export interface SafetyReason {
  code: string;
  message: string;
}

export interface SafetyDecision {
  allowed: boolean;
  reasons: SafetyReason[];
}

export type CellValue = string | number | boolean | null;

export interface ExecutionResult {
  columns: string[];
  rows: CellValue[][];
  returned_row_count: number;
  /** True iff at least one more row existed beyond `max_rows`. No total is ever known. */
  truncated: boolean;
  max_rows: number;
  elapsed_ms: number;
}

export interface ErrorInfo {
  stage: Stage;
  code: string;
  message: string;
}

export interface Timings {
  schema_ms: number | null;
  model_ms: number | null;
  safety_ms: number | null;
  execution_ms: number | null;
  total_ms: number;
}

export interface QueryResponse {
  request_id: string;
  database_id: string;
  status: QueryStatus;
  generated_sql: string | null;
  safety: SafetyDecision | null;
  result: ExecutionResult | null;
  error: ErrorInfo | null;
  timings: Timings;
  /** Free-form runtime metadata; read it through `readModelMeta`. */
  model: Record<string, unknown>;
  prompt_sha256: string | null;
  dialect: string | null;
}
