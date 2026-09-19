/**
 * Defensive runtime validation of backend payloads. The backend is trusted to
 * be well-formed, but a proxy, a version skew or a bug must never crash the UI
 * or reach it as `undefined` deep in a component: anything off-contract
 * becomes a `malformed` ApiError at the boundary.
 */
import { ApiError } from "./errors";
import {
  QUERY_STATUSES,
  STAGES,
  type CellValue,
  type DatabaseInfo,
  type ErrorInfo,
  type ExecutionResult,
  type HealthResponse,
  type ModelRuntimeInfo,
  type QueryResponse,
  type QueryStatus,
  type Reliability,
  type StageState,
  type SafetyDecision,
  type Stage,
  type Timings,
} from "./types";

type Rec = Record<string, unknown>;

export const isRecord = (v: unknown): v is Rec => typeof v === "object" && v !== null && !Array.isArray(v);

function bad(what: string, requestId?: string | null): never {
  throw new ApiError({ kind: "malformed", message: `Unexpected response shape: ${what}`, requestId });
}

const str = (v: unknown, what: string): string => (typeof v === "string" ? v : bad(what));
const num = (v: unknown, what: string): number => (typeof v === "number" && Number.isFinite(v) ? v : bad(what));
const bool = (v: unknown, what: string): boolean => (typeof v === "boolean" ? v : bad(what));
const optStr = (v: unknown, what: string): string | null => (v == null ? null : str(v, what));
const optNum = (v: unknown, what: string): number | null => (v == null ? null : num(v, what));

function cell(v: unknown): CellValue {
  if (v === null || typeof v === "string" || typeof v === "boolean") return v;
  if (typeof v === "number") return Number.isFinite(v) ? v : String(v);
  return JSON.stringify(v) ?? null; // never let an object reach a table cell
}

export function parseDatabases(json: unknown): DatabaseInfo[] {
  if (!isRecord(json) || !Array.isArray(json.databases)) bad("databases");
  return json.databases.map((d: unknown, i: number) => {
    if (!isRecord(d)) bad(`databases[${i}]`);
    return {
      id: str(d.id, "database.id"),
      dialect: str(d.dialect, "database.dialect"),
      description: optStr(d.description, "database.description"),
    };
  });
}

export function parseHealth(json: unknown): HealthResponse {
  if (!isRecord(json) || !isRecord(json.model_runtime)) bad("health");
  const m = json.model_runtime;
  const runtime: ModelRuntimeInfo = {
    runtime: str(m.runtime, "model_runtime.runtime"),
    configured: bool(m.configured, "model_runtime.configured"),
    deployment_mode: typeof m.deployment_mode === "string" ? m.deployment_mode : undefined,
    base_gguf: typeof m.base_gguf === "string" ? m.base_gguf : null,
    lora_gguf: typeof m.lora_gguf === "string" ? m.lora_gguf : null,
    context_size: typeof m.context_size === "number" ? m.context_size : undefined,
    max_new_tokens: typeof m.max_new_tokens === "number" ? m.max_new_tokens : undefined,
    n_gpu_layers: typeof m.n_gpu_layers === "number" ? m.n_gpu_layers : undefined,
    sampling: typeof m.sampling === "string" ? m.sampling : undefined,
    reason: typeof m.reason === "string" ? m.reason : undefined,
    ready: typeof m.ready === "boolean" ? m.ready : undefined,
    checks: isRecord(m.checks) ? Object.fromEntries(Object.entries(m.checks).filter(([, v]) => typeof v === "boolean")) as Record<string, boolean> : undefined,
    availability: parseAvailability(m.availability),
  };
  return { status: str(json.status, "health.status"), model_runtime: runtime };
}

function parseAvailability(v: unknown): HealthResponse["model_runtime"]["availability"] {
  if (!isRecord(v)) return null;
  const state = v.state;
  if (state !== "idle" && state !== "busy" && state !== "saturated") return null;
  const n = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? x : 0);
  return { state, running: n(v.running), waiting: n(v.waiting), max_concurrent: n(v.max_concurrent), max_waiting: n(v.max_waiting) };
}

const STAGE_STATES: readonly string[] = ["passed", "failed", "not_run"];
const stageState = (v: unknown): StageState => (typeof v === "string" && STAGE_STATES.includes(v) ? (v as StageState) : "not_run");

function parseReliability(v: unknown): Reliability | undefined {
  if (!isRecord(v)) return undefined; // older backend: absent, and the UI simply omits the trust note
  return {
    safety: stageState(v.safety),
    preflight: stageState(v.preflight),
    execution: stageState(v.execution),
    semantic_correctness: "not_verified", // never anything else: no server claim can upgrade this
    confidence: null,
    note: typeof v.note === "string" ? v.note : "",
  };
}

function parseSafety(v: unknown): SafetyDecision | null {
  if (v == null) return null;
  if (!isRecord(v) || !Array.isArray(v.reasons)) bad("safety");
  return {
    allowed: bool(v.allowed, "safety.allowed"),
    reasons: v.reasons.map((r: unknown) => {
      if (!isRecord(r)) bad("safety.reasons[]");
      return { code: str(r.code, "safety.reason.code"), message: str(r.message, "safety.reason.message") };
    }),
  };
}

function parseResult(v: unknown): ExecutionResult | null {
  if (v == null) return null;
  if (!isRecord(v) || !Array.isArray(v.columns) || !Array.isArray(v.rows)) bad("result");
  const columns = v.columns.map((c: unknown) => str(c, "result.columns[]"));
  const rows = v.rows.map((row: unknown) => {
    if (!Array.isArray(row)) bad("result.rows[]");
    return row.map(cell);
  });
  return {
    columns,
    rows,
    returned_row_count: typeof v.returned_row_count === "number" ? v.returned_row_count : rows.length,
    truncated: bool(v.truncated, "result.truncated"),
    max_rows: num(v.max_rows, "result.max_rows"),
    elapsed_ms: num(v.elapsed_ms, "result.elapsed_ms"),
  };
}

function parseError(v: unknown): ErrorInfo | null {
  if (v == null) return null;
  if (!isRecord(v)) bad("error");
  const stage = v.stage;
  if (typeof stage !== "string" || !(STAGES as readonly string[]).includes(stage)) bad("error.stage");
  return {
    stage: stage as Stage,
    code: str(v.code, "error.code"),
    message: str(v.message, "error.message"),
    detail: typeof v.detail === "string" ? v.detail : null,
  };
}

function parseTimings(v: unknown): Timings {
  if (!isRecord(v)) bad("timings");
  return {
    schema_ms: optNum(v.schema_ms, "timings.schema_ms"),
    model_ms: optNum(v.model_ms, "timings.model_ms"),
    safety_ms: optNum(v.safety_ms, "timings.safety_ms"),
    preflight_ms: optNum(v.preflight_ms, "timings.preflight_ms"),
    execution_ms: optNum(v.execution_ms, "timings.execution_ms"),
    total_ms: num(v.total_ms, "timings.total_ms"),
  };
}

/** True when `json` looks like a pipeline `QueryResponse` (accepted on non-2xx too). */
export function looksLikeQueryResponse(json: unknown): boolean {
  return (
    isRecord(json) &&
    typeof json.status === "string" &&
    (QUERY_STATUSES as readonly string[]).includes(json.status) &&
    isRecord(json.timings)
  );
}

export function parseQueryResponse(json: unknown): QueryResponse {
  if (!isRecord(json)) bad("query response");
  const status = json.status;
  if (typeof status !== "string" || !(QUERY_STATUSES as readonly string[]).includes(status)) bad("status");
  const res: QueryResponse = {
    request_id: str(json.request_id, "request_id"),
    database_id: str(json.database_id, "database_id"),
    status: status as QueryStatus,
    generated_sql: optStr(json.generated_sql, "generated_sql"),
    safety: parseSafety(json.safety),
    result: parseResult(json.result),
    error: parseError(json.error),
    timings: parseTimings(json.timings),
    model: isRecord(json.model) ? json.model : {},
    prompt_sha256: optStr(json.prompt_sha256, "prompt_sha256"),
    dialect: optStr(json.dialect, "dialect"),
    reliability: parseReliability(json.reliability),
  };
  if (res.status === "ok" && !res.result) bad("ok response without a result", res.request_id);
  return res;
}
