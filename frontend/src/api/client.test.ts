import { describe, expect, it, vi } from "vitest";
import { SchemaForgeClient } from "./client";
import { ApiError } from "./errors";
import { failureResponse, HEALTH_READY, makeResponse } from "../test/fixtures";

const json = (body: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json", "X-Request-ID": "hdr-request-id-1" }, ...init });

function client(fetchImpl: typeof fetch, opts: Partial<ConstructorParameters<typeof SchemaForgeClient>[0]> = {}) {
  return new SchemaForgeClient({ baseUrl: "http://api.test/", fetchImpl, ...opts });
}

const rejectsWith = async (p: Promise<unknown>) => {
  try {
    await p;
  } catch (e) {
    return e as ApiError;
  }
  throw new Error("expected rejection");
};

describe("SchemaForgeClient", () => {
  it("GETs /health and /databases with a normalized base URL", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(json(HEALTH_READY))
      .mockResolvedValueOnce(json({ databases: [{ id: "demo", dialect: "sqlite", description: null }] }));
    const c = client(fetchImpl);
    expect((await c.health()).model_runtime.configured).toBe(true);
    expect(await c.databases()).toEqual([{ id: "demo", dialect: "sqlite", description: null }]);
    expect(fetchImpl.mock.calls[0]![0]).toBe("http://api.test/health");
    expect(fetchImpl.mock.calls[1]![0]).toBe("http://api.test/databases");
  });

  it("POSTs the query body exactly as given and parses the response", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(json(makeResponse()));
    const res = await client(fetchImpl).query({ database_id: "demo", question: "How many?", business_context: "ctx" });
    const [url, init] = fetchImpl.mock.calls[0]!;
    expect(url).toBe("http://api.test/query");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ database_id: "demo", question: "How many?", business_context: "ctx" });
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(res.status).toBe("ok");
    expect(res.result?.rows).toEqual([[12]]);
  });

  it("treats pipeline outcomes on non-2xx as results, not transport errors", async () => {
    const body = failureResponse("unsafe_sql", "safety", "not_read_only_query", {
      generated_sql: "DROP TABLE employees",
      safety: { allowed: false, reasons: [{ code: "not_read_only_query", message: "Only SELECT / WITH queries are allowed." }] },
    });
    const res = await client(vi.fn().mockResolvedValue(json(body, { status: 422 }))).query({ database_id: "demo", question: "q" });
    expect(res.status).toBe("unsafe_sql");
    expect(res.generated_sql).toBe("DROP TABLE employees");
  });

  it("maps the backend error envelope to a structured ApiError", async () => {
    const env = { request_id: "abc", error: { code: "unknown_database", message: "Unknown database id." } };
    const err = await rejectsWith(client(vi.fn().mockResolvedValue(json(env, { status: 404 }))).query({ database_id: "x", question: "q" }));
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ kind: "http", status: 404, code: "unknown_database", requestId: "hdr-request-id-1" });
  });

  it("keeps validation field info without echoing input", async () => {
    const env = { request_id: "abc", error: { code: "invalid_request", message: "Request validation failed.", fields: [{ field: "question", issue: "string_too_long" }] } };
    const err = await rejectsWith(client(vi.fn().mockResolvedValue(json(env, { status: 422 }))).query({ database_id: "d", question: "q" }));
    expect(err.fields).toEqual([{ field: "question", issue: "string_too_long" }]);
  });

  it("handles a non-JSON error body (e.g. a proxy 502 page)", async () => {
    const res = new Response("<html>Bad gateway</html>", { status: 502, headers: { "Content-Type": "text/html" } });
    const err = await rejectsWith(client(vi.fn().mockResolvedValue(res)).databases());
    expect(err).toMatchObject({ kind: "http", status: 502 });
    expect(err.code).toBeUndefined();
  });

  it("reports a network failure", async () => {
    const err = await rejectsWith(client(vi.fn().mockRejectedValue(new TypeError("Failed to fetch"))).health());
    expect(err).toMatchObject({ kind: "network" });
  });

  it("times out via the abort signal and reports kind=timeout", async () => {
    const hang: typeof fetch = (_url, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      });
    const err = await rejectsWith(client(hang, { queryTimeoutMs: 20 }).query({ database_id: "demo", question: "q" }));
    expect(err).toMatchObject({ kind: "timeout" });
  });

  it("reports a user cancellation as kind=aborted (not timeout)", async () => {
    const hang: typeof fetch = (_url, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      });
    const ctl = new AbortController();
    const p = client(hang).query({ database_id: "demo", question: "q" }, ctl.signal);
    ctl.abort();
    expect(await rejectsWith(p)).toMatchObject({ kind: "aborted" });
  });

  it.each([
    ["not an object", "hello"],
    ["unknown status", { ...makeResponse(), status: "weird" }],
    ["missing timings", { ...makeResponse(), timings: undefined }],
    ["ok without result", { ...makeResponse(), result: null }],
    ["rows not arrays", { ...makeResponse(), result: { ...makeResponse().result, rows: ["nope"] } }],
    ["bad error stage", { ...failureResponse("model_error", "model", "model_error"), error: { stage: "kernel", code: "x", message: "y" } }],
  ])("rejects a malformed 200 response: %s", async (_name, body) => {
    const err = await rejectsWith(client(vi.fn().mockResolvedValue(json(body))).query({ database_id: "demo", question: "q" }));
    expect(err).toMatchObject({ kind: "malformed" });
  });

  it("rejects a 200 with a non-JSON body as malformed", async () => {
    const res = new Response("ok", { status: 200 });
    expect(await rejectsWith(client(vi.fn().mockResolvedValue(res)).health())).toMatchObject({ kind: "malformed" });
  });

  it("rejects malformed /databases and /health payloads", async () => {
    expect(await rejectsWith(client(vi.fn().mockResolvedValue(json({ nope: 1 }))).databases())).toMatchObject({ kind: "malformed" });
    expect(await rejectsWith(client(vi.fn().mockResolvedValue(json({ status: "ok" }))).health())).toMatchObject({ kind: "malformed" });
  });

  it("stringifies non-primitive cells instead of letting objects reach the table", async () => {
    const body = makeResponse({ result: { columns: ["a"], rows: [[{ nested: 1 } as never]], returned_row_count: 1, truncated: false, max_rows: 10, elapsed_ms: 1 } });
    const res = await client(vi.fn().mockResolvedValue(json(body))).query({ database_id: "demo", question: "q" });
    expect(res.result?.rows[0]?.[0]).toBe('{"nested":1}');
  });
});

describe("SchemaForgeClient reliability behavior", () => {
  const hang: typeof fetch = (_url, init) =>
    new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    });

  it("sends a client-generated request id the backend can cancel by", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(json(makeResponse()));
    await client(fetchImpl).query({ database_id: "demo", question: "q" });
    const id = fetchImpl.mock.calls[0]![1].headers["X-Request-ID"];
    expect(id).toMatch(/^[A-Za-z0-9_.-]{8,64}$/);
  });

  it("tells the backend to cancel when the user aborts (and only then)", async () => {
    const calls: [string, RequestInit][] = [];
    const fetchImpl = vi.fn((url: string, init: RequestInit) => {
      calls.push([url, init]);
      return url.endsWith("/cancel") ? Promise.resolve(json({ cancelled: true })) : hang(url, init);
    });
    const ctl = new AbortController();
    const p = client(fetchImpl as never).query({ database_id: "demo", question: "q" }, ctl.signal);
    ctl.abort();
    expect(await rejectsWith(p)).toMatchObject({ kind: "aborted" });
    const id = (calls[0]![1].headers as Record<string, string>)["X-Request-ID"];
    await vi.waitFor(() => expect(calls).toHaveLength(2));
    expect(calls[1]![0]).toBe(`http://api.test/query/${id}/cancel`);
    expect(calls[1]![1].method).toBe("POST");
  });

  it("also cancels on a client-side timeout, and swallows a failing cancel call", async () => {
    const calls: string[] = [];
    const fetchImpl = vi.fn((url: string, init: RequestInit) => {
      calls.push(url);
      return url.endsWith("/cancel") ? Promise.reject(new TypeError("offline")) : hang(url, init);
    });
    const err = await rejectsWith(client(fetchImpl as never, { queryTimeoutMs: 20 }).query({ database_id: "demo", question: "q" }));
    expect(err).toMatchObject({ kind: "timeout" });
    await vi.waitFor(() => expect(calls.some((u) => u.endsWith("/cancel"))).toBe(true));
  });

  it("does not send a cancel for ordinary failures or successes", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(json(makeResponse()));
    await client(fetchImpl).query({ database_id: "demo", question: "q" });
    const bad = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    await rejectsWith(client(bad).query({ database_id: "demo", question: "q" }));
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(bad).toHaveBeenCalledTimes(1);
  });

  it("parses the additive Phase 10 fields and defaults them safely", async () => {
    const body = failureResponse("validation_error", "preflight", "unknown_column", {
      error: { stage: "preflight", code: "unknown_column", message: "m", detail: "bonus" },
      reliability: { safety: "passed", preflight: "failed", execution: "not_run", semantic_correctness: "not_verified", confidence: null, note: "n" },
      timings: { schema_ms: 1, model_ms: 2, safety_ms: 3, preflight_ms: 4, execution_ms: null, total_ms: 9 },
    });
    const res = await client(vi.fn().mockResolvedValue(json(body, { status: 422 }))).query({ database_id: "demo", question: "q" });
    expect(res.status).toBe("validation_error");
    expect(res.error?.detail).toBe("bonus");
    expect(res.timings.preflight_ms).toBe(4);
    expect(res.reliability).toMatchObject({ preflight: "failed", semantic_correctness: "not_verified", confidence: null });
  });

  it("never lets a payload upgrade semantic correctness or invent a confidence", async () => {
    const forged = { ...makeResponse(), reliability: { safety: "passed", preflight: "passed", execution: "passed", semantic_correctness: "verified", confidence: 0.99, note: "trust me" } };
    const res = await client(vi.fn().mockResolvedValue(json(forged))).query({ database_id: "demo", question: "q" });
    expect(res.reliability?.semantic_correctness).toBe("not_verified");
    expect(res.reliability?.confidence).toBeNull();
  });

  it("parses health readiness and ignores malformed availability", async () => {
    const good = { status: "ok", model_runtime: { runtime: "llama_cpp", configured: true, ready: true, checks: { executable: true, base_gguf: false, junk: "x" }, availability: { state: "busy", running: 1, waiting: 0, max_concurrent: 1, max_waiting: 2 } } };
    const h = await client(vi.fn().mockResolvedValue(json(good))).health();
    expect(h.model_runtime).toMatchObject({ ready: true, checks: { executable: true, base_gguf: false }, availability: { state: "busy", running: 1 } });
    const weird = { status: "ok", model_runtime: { runtime: "x", configured: true, availability: { state: "exploding" } } };
    expect((await client(vi.fn().mockResolvedValue(json(weird))).health()).model_runtime.availability).toBeNull();
  });
});
