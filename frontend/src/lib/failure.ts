/**
 * Turns every way a query can go wrong -- pipeline statuses from the backend,
 * transport errors, contract violations -- into one presentation model with
 * calm, intentional copy. Internal detail (paths, stack traces) never reaches
 * this layer, and generic backend messages are replaced by product copy.
 */
import { ApiError } from "../api/errors";
import type { QueryResponse } from "../api/types";

export type FailureKind =
  | "unsafe_sql"
  | "model_error"
  | "model_unavailable"
  | "model_busy"
  | "model_timeout"
  | "malformed_output"
  | "invalid_for_database"
  | "execution_error"
  | "timeout"
  | "schema_error"
  | "backend_unavailable"
  | "client_timeout"
  | "validation"
  | "unknown_database"
  | "database_unavailable"
  | "malformed_response"
  | "cancelled"
  | "unexpected";

export type Tone = "danger" | "warning" | "neutral";

export interface Failure {
  kind: FailureKind;
  tone: Tone;
  title: string;
  message: string;
  hint: string;
  retryable: boolean;
  /** Short machine code, shown small for support/debugging. */
  code?: string;
  requestId?: string | null;
  /** Extra, already-safe lines (e.g. the SQLite error text, rejected-construct reasons). */
  details?: string[];
}

export type Outcome = { kind: "response"; response: QueryResponse } | { kind: "error"; error: ApiError };

export type Classified = { ok: true; response: QueryResponse } | { ok: false; failure: Failure; response?: QueryResponse };

function fromResponse(r: QueryResponse): Failure {
  const code = r.error?.code;
  const base = { code, requestId: r.request_id };
  switch (r.status) {
    case "unsafe_sql":
      return {
        ...base,
        kind: "unsafe_sql",
        tone: "danger",
        title: "Blocked by the safety policy",
        message: "The model produced SQL that is not a single read-only query. It was never sent to your database.",
        hint: "Rephrase the question, or ask for information rather than a change.",
        retryable: true,
        details: r.safety?.reasons.map((x) => x.message),
      };
    case "model_error":
      if (code === "model_not_configured") {
        return {
          ...base,
          kind: "model_unavailable",
          tone: "warning",
          title: "The model server isn't configured",
          message: "The backend is running, but no model server is attached, so it can't generate SQL yet.",
          hint: "Set SCHEMAFORGE_LLAMA_SERVER_URL (persistent server) or the SCHEMAFORGE_* model paths, then restart the backend (see docs/PHASE11.md).",
          retryable: false,
        };
      }
      if (code === "model_busy") {
        return {
          ...base,
          kind: "model_busy",
          tone: "warning",
          title: "The model server is busy",
          message: "Another request is using the model right now, and the queue is full. Nothing was run against your database.",
          hint: "Wait a few seconds and try again. The model server handles a limited number of requests at a time.",
          retryable: true,
        };
      }
      if (code === "model_timeout") {
        return {
          ...base,
          kind: "model_timeout",
          tone: "warning",
          title: "The model took too long",
          message: "Inference on the model server hit the backend's generation time limit and was stopped. Your database was not touched.",
          hint: "Try a shorter question, or retry when the model server is less busy.",
          retryable: true,
        };
      }
      if (code === "malformed_model_output") {
        return {
          ...base,
          kind: "malformed_output",
          tone: "danger",
          title: "The model's answer wasn't usable SQL",
          message: "The model responded, but its output wasn't a SQL query, so nothing was run.",
          hint: "Rephrase the question and try again.",
          retryable: true,
        };
      }
      return {
        ...base,
        kind: "model_error",
        tone: "danger",
        title: "The model couldn't produce SQL",
        message: "Inference on the model server failed before any SQL was generated. Your database was not touched.",
        hint: "Try again. If it keeps failing, check the backend log for this request ID.",
        retryable: true,
      };
    case "execution_error":
      if (code === "timeout") {
        return {
          ...base,
          kind: "timeout",
          tone: "warning",
          title: "The query ran out of time",
          message: "The generated SQL passed safety checks but exceeded the read-only execution time limit and was stopped.",
          hint: "Ask for something narrower, such as a filter or a smaller time range.",
          retryable: true,
        };
      }
      return {
        ...base,
        kind: "execution_error",
        tone: "danger",
        title: code === "authorization_denied" ? "The database refused this query" : "The SQL couldn't be executed",
        message:
          code === "authorization_denied"
            ? "The read-only executor blocked an operation that isn't permitted."
            : "The SQL passed safety checks but failed when run against the database, often a column or table the model got wrong.",
        hint: "Adding business context about the columns you mean often helps.",
        retryable: true,
        details: code === "sql_error" && r.error ? [r.error.message] : undefined,
      };
    case "validation_error": {
      const what: Record<string, string> = {
        unknown_table: "a table that doesn't exist in this database",
        unknown_column: "a column that doesn't exist in this database",
        ambiguous_column: "a column name that matches more than one table",
        unknown_function: "a function this database doesn't provide",
        invalid_syntax: "SQL that isn't valid for this database",
      };
      return {
        ...base,
        kind: "invalid_for_database",
        tone: "danger",
        title: "The SQL doesn't fit this database",
        message: `The model referred to ${what[code ?? ""] ?? "something that isn't valid here"}. It passed the safety checks, but it was checked against the real schema and never executed.`,
        hint: "Adding business context that names the right tables or columns often helps.",
        retryable: true,
        details: r.error?.detail ? [`Not found or invalid: ${r.error.detail}`] : undefined,
      };
    }
    case "cancelled":
      return {
        ...base,
        kind: "cancelled",
        tone: "neutral",
        title: "Request cancelled",
        message: "This query was cancelled before it finished, and any running work was stopped.",
        hint: "Run it again whenever you're ready.",
        retryable: true,
      };
    case "schema_error":
      return {
        ...base,
        kind: "schema_error",
        tone: "danger",
        title: "The schema couldn't be read",
        message: "SchemaForge couldn't introspect this database, so no prompt was built.",
        hint: "Check that the database file is a valid SQLite database with at least one table.",
        retryable: false,
      };
    default:
      return { ...base, kind: "unexpected", tone: "danger", title: "Something unexpected happened", message: "The response could not be interpreted.", hint: "Try again.", retryable: true };
  }
}

function fromError(e: ApiError): Failure {
  const base = { code: e.code, requestId: e.requestId };
  switch (e.kind) {
    case "network":
      return {
        ...base,
        kind: "backend_unavailable",
        tone: "warning",
        title: "Can't reach the SchemaForge backend",
        message: "The browser couldn't connect to the API. It may not be running, or its address may be misconfigured.",
        hint: "Start it with the uvicorn command in docs/PHASE9.md, then retry.",
        retryable: true,
      };
    case "timeout":
      return {
        ...base,
        kind: "client_timeout",
        tone: "warning",
        title: "No answer within the wait limit",
        message: "Inference on the model server is taking longer than this page is willing to wait. The backend may still be working.",
        hint: "Retry, or raise VITE_QUERY_TIMEOUT_MS for slower (CPU) model hosts.",
        retryable: true,
      };
    case "aborted":
      return {
        ...base,
        kind: "cancelled",
        tone: "neutral",
        title: "Request cancelled",
        message: "You cancelled this query. The backend may finish generating, but the result is discarded.",
        hint: "Run it again whenever you're ready.",
        retryable: true,
      };
    case "malformed":
      return {
        ...base,
        kind: "malformed_response",
        tone: "danger",
        title: "The backend sent something unexpected",
        message: "The response didn't match the SchemaForge API contract, so it was not displayed.",
        hint: "This usually means the frontend and backend versions differ.",
        retryable: true,
      };
    case "http":
      if (e.code === "unknown_database")
        return { ...base, kind: "unknown_database", tone: "warning", title: "That database isn't registered", message: "The selected database is no longer available from the backend.", hint: "Reload the database list and choose another.", retryable: false };
      if (e.code === "database_unavailable")
        return { ...base, kind: "database_unavailable", tone: "warning", title: "The database is unavailable", message: "It's registered, but its file couldn't be opened right now.", hint: "Check the database on the machine running the backend.", retryable: true };
      if (e.code === "rate_limited")
        return { ...base, kind: "model_busy", tone: "warning", title: "Too many requests", message: "This public demo limits how many queries each visitor can run per minute.", hint: "Wait a moment and try again.", retryable: true };
      if (e.code === "invalid_request")
        return {
          ...base,
          kind: "validation",
          tone: "warning",
          title: "That request wasn't valid",
          message: "The backend rejected the question or context (for example, too long or empty).",
          hint: "Shorten the question or business context and try again.",
          retryable: false,
          details: e.fields?.map((f) => `${f.field}: ${f.issue}`),
        };
      return { ...base, kind: "unexpected", tone: "danger", title: "The backend hit an error", message: "It returned an unexpected error. No details are shown for safety.", hint: "Try again; check the backend log using the request ID.", retryable: true };
  }
}

export function classifyOutcome(outcome: Outcome): Classified {
  if (outcome.kind === "error") return { ok: false, failure: fromError(outcome.error) };
  if (outcome.response.status === "ok") return { ok: true, response: outcome.response };
  return { ok: false, failure: fromResponse(outcome.response), response: outcome.response };
}
