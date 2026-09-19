import { describe, expect, it } from "vitest";
import { readConfig } from "../api/config";
import { ApiError } from "../api/errors";
import type { ExecutionResult } from "../api/types";
import { failureResponse, makeResponse } from "../test/fixtures";
import { analyzeChartability, extractSeries, MAX_CHART_CATEGORIES } from "./chartability";
import { classifyOutcome } from "./failure";
import { formatCell, formatMs, formatNumber } from "./format";
import { pathForRoute, routeFromPath } from "./route";
import { tokenizeSql } from "./sqlHighlight";

const res = (columns: string[], rows: ExecutionResult["rows"]): ExecutionResult => ({
  columns,
  rows,
  returned_row_count: rows.length,
  truncated: false,
  max_rows: 500,
  elapsed_ms: 1,
});

describe("tokenizeSql", () => {
  it("is lossless for any input", () => {
    const samples = [
      "SELECT SUM(T1.salary) FROM employees AS T1 INNER JOIN departments AS T2 ON T1.dept_id = T2.dept_id WHERE T2.name = 'Engineering'",
      "SELECT 'it''s' -- comment\n, \"Quoted Col\", `x`, [y] /* block */ FROM t WHERE a >= 1.5e-3;",
      "SELECT 'unterminated",
      "",
      "  \n\t ",
    ];
    for (const s of samples) expect(tokenizeSql(s).map((t) => t.text).join("")).toBe(s);
  });

  it("classifies keywords, functions, strings, numbers and comments", () => {
    const types = (s: string) => Object.fromEntries(tokenizeSql(s).filter((t) => t.type !== "space" && t.type !== "punct").map((t) => [t.text, t.type]));
    expect(types("SELECT COUNT(id) FROM t WHERE n > 10 AND s = 'a' -- hi")).toMatchObject({
      SELECT: "keyword",
      COUNT: "function",
      id: "identifier",
      FROM: "keyword",
      "10": "number",
      "'a'": "string",
      "-- hi": "comment",
    });
  });

  it("treats markup in SQL as inert text", () => {
    const html = "<img src=x onerror=alert(1)>";
    expect(tokenizeSql(`SELECT '${html}'`).map((t) => t.text).join("")).toContain(html);
  });
});

describe("analyzeChartability", () => {
  it("offers a bar chart for a text label + numeric measure", () => {
    const d = analyzeChartability(res(["dept", "total"], [["Eng", 625000], ["Sales", 275000], ["Support", 210000]]));
    expect(d).toEqual({ eligible: true, kind: "bar", labelColumn: 0, measureColumns: [1] });
  });

  it("offers several measures and prefers real measures over id columns", () => {
    const d = analyzeChartability(res(["name", "id", "salary", "bonus"], [["a", 1, 10, 1], ["b", 2, 20, 2]]));
    expect(d).toMatchObject({ eligible: true, measureColumns: [2, 3] });
  });

  it("uses an id-like numeric column as the label when there is no text column", () => {
    const d = analyzeChartability(res(["dept_id", "n"], [[1, 4], [2, 3]]));
    expect(d).toMatchObject({ eligible: true, labelColumn: 0, measureColumns: [1] });
  });

  it("uses a line chart for date-like labels", () => {
    const d = analyzeChartability(res(["month", "sales"], [["2024-01", 5], ["2024-02", 7], ["2024-03", 6]]));
    expect(d).toMatchObject({ eligible: true, kind: "line" });
  });

  it.each([
    ["a single row", res(["a", "b"], [["x", 1]])],
    ["a single column", res(["a"], [["x"], ["y"]])],
    ["no numeric column", res(["a", "b"], [["x", "p"], ["y", "q"]])],
    ["two numeric columns and no label", res(["x", "y"], [[1, 2], [3, 4]])],
    ["repeated labels", res(["a", "n"], [["x", 1], ["x", 2]])],
    ["a numeric column that is all NULL", res(["a", "n"], [["x", null], ["y", null]])],
    ["empty result", res(["a", "n"], [])],
  ])("declines %s", (_n, r) => {
    const d = analyzeChartability(r);
    expect(d.eligible).toBe(false);
    if (!d.eligible) expect(d.reason.length).toBeGreaterThan(5);
  });

  it("declines too many categories", () => {
    const rows = Array.from({ length: MAX_CHART_CATEGORIES + 1 }, (_, i) => [`k${i}`, i]);
    expect(analyzeChartability(res(["k", "v"], rows)).eligible).toBe(false);
  });

  it("is deterministic and extracts a series with nulls preserved", () => {
    const r = res(["k", "v"], [["a", 1], ["b", null], ["c", 3]]);
    expect(analyzeChartability(r)).toEqual(analyzeChartability(r));
    expect(extractSeries(r, 0, 1)).toEqual([{ label: "a", value: 1 }, { label: "b", value: null }, { label: "c", value: 3 }]);
  });
});

describe("classifyOutcome", () => {
  const resp = (r: ReturnType<typeof makeResponse>) => classifyOutcome({ kind: "response", response: r });
  const err = (e: ApiError) => classifyOutcome({ kind: "error", error: e });
  const kind = (c: ReturnType<typeof classifyOutcome>) => (c.ok ? "ok" : c.failure.kind);

  it("passes successful responses through", () => {
    expect(resp(makeResponse()).ok).toBe(true);
  });

  it("distinguishes every pipeline failure", () => {
    expect(kind(resp(failureResponse("unsafe_sql", "safety", "not_read_only_query")))).toBe("unsafe_sql");
    expect(kind(resp(failureResponse("model_error", "model", "model_error")))).toBe("model_error");
    expect(kind(resp(failureResponse("model_error", "model", "model_not_configured")))).toBe("model_unavailable");
    expect(kind(resp(failureResponse("execution_error", "execution", "timeout")))).toBe("timeout");
    expect(kind(resp(failureResponse("execution_error", "execution", "sql_error")))).toBe("execution_error");
    expect(kind(resp(failureResponse("execution_error", "execution", "authorization_denied")))).toBe("execution_error");
    expect(kind(resp(failureResponse("schema_error", "schema", "schema_error")))).toBe("schema_error");
  });

  it("distinguishes every transport failure", () => {
    expect(kind(err(new ApiError({ kind: "network", message: "x" })))).toBe("backend_unavailable");
    expect(kind(err(new ApiError({ kind: "timeout", message: "x" })))).toBe("client_timeout");
    expect(kind(err(new ApiError({ kind: "aborted", message: "x" })))).toBe("cancelled");
    expect(kind(err(new ApiError({ kind: "malformed", message: "x" })))).toBe("malformed_response");
    expect(kind(err(new ApiError({ kind: "http", message: "x", status: 404, code: "unknown_database" })))).toBe("unknown_database");
    expect(kind(err(new ApiError({ kind: "http", message: "x", status: 503, code: "database_unavailable" })))).toBe("database_unavailable");
    expect(kind(err(new ApiError({ kind: "http", message: "x", status: 422, code: "invalid_request" })))).toBe("validation");
    expect(kind(err(new ApiError({ kind: "http", message: "x", status: 500, code: "internal_error" })))).toBe("unexpected");
  });

  it("never surfaces raw internal messages for opaque errors", () => {
    const c = err(new ApiError({ kind: "http", message: "Traceback: secret /var/db/x", status: 500, code: "internal_error" }));
    expect(JSON.stringify(c)).not.toContain("secret");
  });

  it("keeps rejection reasons for unsafe SQL", () => {
    const c = resp(failureResponse("unsafe_sql", "safety", "x", { safety: { allowed: false, reasons: [{ code: "x", message: "Disallowed construct: Drop." }] } }));
    expect(!c.ok && c.failure.details).toEqual(["Disallowed construct: Drop."]);
  });
});

describe("formatting", () => {
  it("formats durations", () => {
    expect(formatMs(0.2)).toBe("<1 ms");
    expect(formatMs(120)).toBe("120 ms");
    expect(formatMs(6410)).toBe("6.41 s");
    expect(formatMs(22434.9)).toBe("22.4 s");
    expect(formatMs(null)).toBe("—");
  });
  it("formats cells without altering meaning", () => {
    expect(formatCell(null)).toEqual({ text: "NULL", kind: "null" });
    expect(formatCell(true)).toEqual({ text: "true", kind: "boolean" });
    expect(formatCell(625000)).toEqual({ text: "625,000", kind: "number" });
    expect(formatCell("<b>x</b>")).toEqual({ text: "<b>x</b>", kind: "text" });
    expect(formatNumber(0.1234567891)).toBe("0.123457");
  });
});

describe("config and routing", () => {
  it("reads and validates env", () => {
    expect(readConfig({})).toEqual({ apiBaseUrl: "http://127.0.0.1:8000", queryTimeoutMs: 180000 });
    expect(readConfig({ VITE_API_BASE_URL: "https://api.example.com//", VITE_QUERY_TIMEOUT_MS: "5000" })).toEqual({ apiBaseUrl: "https://api.example.com", queryTimeoutMs: 5000 });
    expect(() => readConfig({ VITE_API_BASE_URL: "not a url" })).toThrow();
    expect(() => readConfig({ VITE_API_BASE_URL: "file:///etc/passwd" })).toThrow();
    expect(readConfig({ VITE_QUERY_TIMEOUT_MS: "-1" }).queryTimeoutMs).toBe(180000);
  });
  it("maps paths to routes", () => {
    expect(routeFromPath("/")).toBe("landing");
    expect(routeFromPath("/workspace")).toBe("workspace");
    expect(routeFromPath("/workspace/")).toBe("workspace");
    expect(routeFromPath("/anything-else")).toBe("landing");
    expect(pathForRoute("workspace")).toBe("/workspace");
  });
});

describe("classifyOutcome: Phase 10 reliability states", () => {
  const kind = (r: ReturnType<typeof makeResponse>) => {
    const c = classifyOutcome({ kind: "response", response: r });
    return c.ok ? "ok" : c.failure.kind;
  };
  it("maps the new stable codes", () => {
    expect(kind(failureResponse("model_error", "model", "model_busy"))).toBe("model_busy");
    expect(kind(failureResponse("model_error", "model", "model_timeout"))).toBe("model_timeout");
    expect(kind(failureResponse("model_error", "model", "malformed_model_output"))).toBe("malformed_output");
    expect(kind(failureResponse("validation_error", "preflight", "unknown_table"))).toBe("invalid_for_database");
    expect(kind(failureResponse("cancelled", "model", "cancelled"))).toBe("cancelled");
  });
  it("keeps every backend failure code distinguishable from a generic error", () => {
    const codes: Record<string, [Parameters<typeof failureResponse>[0], Parameters<typeof failureResponse>[1]]> = {
      model_busy: ["model_error", "model"],
      model_timeout: ["model_error", "model"],
      malformed_model_output: ["model_error", "model"],
      model_not_configured: ["model_error", "model"],
      timeout: ["execution_error", "execution"],
      unknown_column: ["validation_error", "preflight"],
      cancelled: ["cancelled", "model"],
    };
    const kinds = Object.entries(codes).map(([code, [status, stage]]) => kind(failureResponse(status, stage, code)));
    expect(new Set(kinds).size).toBe(kinds.length);
    expect(kinds).not.toContain("model_error");
  });
  it("only invalid-for-database failures expose the identifier hint", () => {
    const c = classifyOutcome({ kind: "response", response: failureResponse("validation_error", "preflight", "unknown_column", { error: { stage: "preflight", code: "unknown_column", message: "m", detail: "bonus" } }) });
    expect(!c.ok && c.failure.details).toEqual(["Not found or invalid: bonus"]);
  });
});
