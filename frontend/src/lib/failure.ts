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
          title: "The local model isn't configured",
          message: "The backend is running, but no model runtime is attached, so it can't generate SQL yet.",
          hint: "Set the SCHEMAFORGE_* model paths and restart the backend (see docs/PHASE8.md).",
          retryable: false,
        };
      }
      return {
        ...base,
        kind: "model_error",
        tone: "danger",
        title: "The model couldn't produce SQL",
        message: "Local inference failed before any SQL was generated. Your database was not touched.",
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
        message: "Local inference is taking longer than this page is willing to wait. The backend may still be working.",
        hint: "Retry, or raise VITE_QUERY_TIMEOUT_MS for slower machines.",
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
