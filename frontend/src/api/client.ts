/**
 * The ONLY place that talks to the backend. Components receive a
 * `SchemaForgeApi` (via context), never `fetch`.
 */
import { ApiError } from "./errors";
import { isRecord, looksLikeQueryResponse, parseDatabases, parseHealth, parseQueryResponse } from "./guards";
import type { DatabaseInfo, HealthResponse, QueryRequest, QueryResponse } from "./types";

export interface SchemaForgeApi {
  health(signal?: AbortSignal): Promise<HealthResponse>;
  databases(signal?: AbortSignal): Promise<DatabaseInfo[]>;
  query(request: QueryRequest, signal?: AbortSignal): Promise<QueryResponse>;
}

export interface ClientOptions {
  baseUrl: string;
  queryTimeoutMs?: number;
  readTimeoutMs?: number;
  fetchImpl?: typeof fetch;
}

interface RawResponse {
  ok: boolean;
  status: number;
  json: unknown; // undefined when the body was not JSON
  requestId: string | null;
}

const isAbort = (e: unknown) => e instanceof DOMException && e.name === "AbortError";

export class SchemaForgeClient implements SchemaForgeApi {
  private readonly baseUrl: string;
  private readonly queryTimeoutMs: number;
  private readonly readTimeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: ClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.queryTimeoutMs = opts.queryTimeoutMs ?? 180_000;
    this.readTimeoutMs = opts.readTimeoutMs ?? 10_000;
    this.fetchImpl = opts.fetchImpl ?? ((...args) => fetch(...args));
  }

  async health(signal?: AbortSignal): Promise<HealthResponse> {
    const res = await this.send("/health", { method: "GET" }, this.readTimeoutMs, signal);
    if (!res.ok) throw this.httpError(res);
    return parseHealth(res.json);
  }

  async databases(signal?: AbortSignal): Promise<DatabaseInfo[]> {
    const res = await this.send("/databases", { method: "GET" }, this.readTimeoutMs, signal);
    if (!res.ok) throw this.httpError(res);
    return parseDatabases(res.json);
  }

  async query(request: QueryRequest, signal?: AbortSignal): Promise<QueryResponse> {
    const res = await this.send(
      "/query",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) },
      this.queryTimeoutMs,
      signal,
    );
    // Pipeline outcomes (unsafe_sql, model_error, timeout, ...) arrive as a full
    // QueryResponse body with a non-2xx status: that is a *result*, not a transport error.
    if (looksLikeQueryResponse(res.json)) return parseQueryResponse(res.json);
    if (!res.ok) throw this.httpError(res);
    return parseQueryResponse(res.json); // 2xx that is not a QueryResponse -> malformed
  }

  private httpError(res: RawResponse): ApiError {
    const env = isRecord(res.json) && isRecord(res.json.error) ? res.json.error : null;
    const code = env && typeof env.code === "string" ? env.code : undefined;
    const message = env && typeof env.message === "string" ? env.message : `Request failed (HTTP ${res.status}).`;
    const fields =
      env && Array.isArray(env.fields)
        ? env.fields.filter(isRecord).map((f) => ({ field: String(f.field ?? ""), issue: String(f.issue ?? "") }))
        : undefined;
    return new ApiError({ kind: "http", message, status: res.status, code, requestId: res.requestId, fields });
  }

  private async send(path: string, init: RequestInit, timeoutMs: number, external?: AbortSignal): Promise<RawResponse> {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
    const onExternalAbort = () => controller.abort();
    if (external?.aborted) controller.abort();
    external?.addEventListener("abort", onExternalAbort, { once: true });

    const interrupted = (e: unknown): ApiError | null => {
      if (timedOut) return new ApiError({ kind: "timeout", message: `No response within ${Math.round(timeoutMs / 1000)}s.` });
      if (controller.signal.aborted || isAbort(e)) return new ApiError({ kind: "aborted", message: "Request cancelled." });
      return null;
    };

    try {
      let response: Response;
      try {
        response = await this.fetchImpl(this.baseUrl + path, {
          ...init,
          signal: controller.signal,
          headers: { Accept: "application/json", ...(init.headers as Record<string, string> | undefined) },
        });
      } catch (e) {
        throw interrupted(e) ?? new ApiError({ kind: "network", message: "Could not reach the SchemaForge backend." });
      }
      let json: unknown;
      try {
        json = await response.json();
      } catch (e) {
        const stop = interrupted(e);
        if (stop) throw stop;
        json = undefined;
      }
      const requestId = response.headers.get("X-Request-ID");
      if (response.ok && json === undefined) {
        throw new ApiError({ kind: "malformed", message: "Unexpected response shape: body is not JSON.", status: response.status, requestId });
      }
      return { ok: response.ok, status: response.status, json, requestId };
    } finally {
      clearTimeout(timer);
      external?.removeEventListener("abort", onExternalAbort);
    }
  }
}
