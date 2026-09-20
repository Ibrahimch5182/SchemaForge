/** Wire types. They mirror the Phase 8 backend contract (docs/PHASE8.md) exactly. */

export type QueryStatus = "ok" | "model_error" | "unsafe_sql" | "validation_error" | "execution_error" | "schema_error" | "cancelled";
export type Stage = "schema" | "model" | "safety" | "preflight" | "execution";
export type StageState = "passed" | "failed" | "not_run";

export const QUERY_STATUSES: readonly QueryStatus[] = ["ok", "model_error", "unsafe_sql", "validation_error", "execution_error", "schema_error", "cancelled"];
export const STAGES: readonly Stage[] = ["schema", "model", "safety", "preflight", "execution"];

export interface DatabaseInfo {
  id: string;
  dialect: string;
  description: string | null;
}

export interface ModelRuntimeInfo {
  runtime: string;
  configured: boolean;
  deployment_mode?: string;
  /** `persistent_server` (llama.cpp HTTP server) or absent for the per-request subprocess runtime. */
  runtime_mode?: string;
  /** Persistent-server state from the backend: ready / busy / saturated / loading / unreachable. */
  serving?: { state: string };
  parallel_slots?: number;
  base_gguf?: string | null;
  lora_gguf?: string | null;
  context_size?: number;
  max_new_tokens?: number;
  n_gpu_layers?: number;
  sampling?: string;
  reason?: string;
  /** Phase 10 readiness (additive): artifacts present so a query can run. Undefined on older backends. */
  ready?: boolean;
  checks?: Record<string, boolean>;
  availability?: Availability | null;
}

export interface Availability {
  state: "idle" | "busy" | "saturated";
  running: number;
  waiting: number;
  max_concurrent: number;
  max_waiting: number;
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
  /** Short identifier-only hint, e.g. the missing column name. */
  detail?: string | null;
}

export interface Timings {
  schema_ms: number | null;
  model_ms: number | null;
  safety_ms: number | null;
  preflight_ms?: number | null;
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
  /** What was and was NOT verified. `semantic_correctness` is always "not_verified"; there is no confidence score. */
  reliability?: Reliability;
}

export interface Reliability {
  safety: StageState;
  preflight: StageState;
  execution: StageState;
  semantic_correctness: "not_verified";
  confidence: null;
  note: string;
}
