import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { HealthResponse, QueryResponse } from "../api/types";
import { failureResponse, HEALTH_READY, makeApi, makeResponse } from "../test/fixtures";
import { renderWorkspace } from "../test/render";

const Q = "How many employees are there?";

async function run(api = makeApi()) {
  const utils = renderWorkspace(api);
  await screen.findByRole("radio", { name: /demo/i });
  await utils.user.type(screen.getByLabelText(/your question/i), Q);
  await utils.user.click(screen.getByRole("button", { name: /run query/i }));
  return { api, ...utils };
}

describe("Reliability states are distinct and honest", () => {
  const cases: [string, QueryResponse, RegExp, string, RegExp?][] = [
    ["model busy", failureResponse("model_error", "model", "model_busy"), /model server is busy/i, "model_busy"],
    ["model timeout", failureResponse("model_error", "model", "model_timeout"), /model took too long/i, "model_timeout"],
    ["malformed output", failureResponse("model_error", "model", "malformed_model_output"), /wasn't usable sql/i, "malformed_output"],
    [
      "invalid for the database",
      failureResponse("validation_error", "preflight", "unknown_column", { generated_sql: "SELECT bonus FROM employees", error: { stage: "preflight", code: "unknown_column", message: "m", detail: "bonus" } }),
      /doesn't fit this database/i,
      "invalid_for_database",
      /Not found or invalid: bonus/,
    ],
    ["cancelled (server)", failureResponse("cancelled", "model", "cancelled"), /request cancelled/i, "cancelled"],
  ];

  it.each(cases)("renders %s as its own state", async (_n, response, title, kind, extra) => {
    await run(makeApi({ query: vi.fn().mockResolvedValue(response) }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(title);
    expect(alert).toHaveAttribute("data-failure", kind);
    if (extra) expect(document.body).toHaveTextContent(extra);
    expect(document.body.textContent).not.toMatch(/Traceback|sqlite3|\.py\b/);
  });

  it("shows the invalid SQL as generated-but-never-executed context", async () => {
    const r = failureResponse("validation_error", "preflight", "unknown_table", {
      generated_sql: "SELECT * FROM staff",
      error: { stage: "preflight", code: "unknown_table", message: "m", detail: "staff" },
    });
    await run(makeApi({ query: vi.fn().mockResolvedValue(r) }));
    expect(await screen.findByLabelText("Generated SQL")).toHaveTextContent("SELECT * FROM staff");
    expect(screen.getByRole("alert")).toHaveTextContent(/never executed/i);
  });

  it("never claims correctness or shows a confidence number on success", async () => {
    await run();
    await screen.findByText("Query executed successfully");
    const note = screen.getByRole("note", { name: /what is and isn't verified/i });
    expect(note).toHaveTextContent(/verified:.*safe, valid for this database, read-only/i);
    expect(note).toHaveTextContent(/not verified:.*answers your question/i);
    expect(screen.getByText("Safety verified")).toBeInTheDocument();
    expect(screen.getByText("Read-only execution")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/confidence|\d+\s?%|guaranteed|correct answer|definitely/i);
  });

  it("does not attach the trust note to failures", async () => {
    await run(makeApi({ query: vi.fn().mockResolvedValue(failureResponse("unsafe_sql", "safety", "not_read_only_query")) }));
    await screen.findByRole("alert");
    expect(screen.queryByRole("note", { name: /what is and isn't verified/i })).not.toBeInTheDocument();
  });

  it("renders a successful older-backend response (no reliability block) unchanged", async () => {
    const legacy = { ...makeResponse() } as Partial<QueryResponse>;
    delete legacy.reliability;
    await run(makeApi({ query: vi.fn().mockResolvedValue(legacy as QueryResponse) }));
    expect(await screen.findByText("Query executed successfully")).toBeInTheDocument();
  });

  it("marks a client-side cancel as cancelled and keeps it out of history", async () => {
    const query = vi.fn((_r: unknown, signal?: AbortSignal) => new Promise<QueryResponse>((_res, rej) => signal?.addEventListener("abort", () => rej(new ApiError({ kind: "aborted", message: "x" })))));
    const { user } = await run(makeApi({ query: query as never }));
    await user.click(await screen.findByRole("button", { name: /cancel request/i }));
    expect(await screen.findByRole("alert")).toHaveAttribute("data-failure", "cancelled");
    expect(screen.queryByText(/this session/i)).not.toBeInTheDocument();
  });
});

describe("Status pill reflects readiness and availability", () => {
  const health = (patch: Partial<HealthResponse["model_runtime"]>): HealthResponse => ({ ...HEALTH_READY, model_runtime: { ...HEALTH_READY.model_runtime, ...patch } });

  it("shows model files missing and blocks running", async () => {
    renderWorkspace(makeApi({ health: vi.fn().mockResolvedValue(health({ ready: false, checks: { executable: true, base_gguf: false, lora_gguf: true } })) }));
    expect(await screen.findByRole("button", { name: /model files missing/i })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: /run query/i })).toBeDisabled());
    expect(screen.getByText(/model files aren't available/i)).toBeInTheDocument();
  });

  it("shows busy and saturated availability without blocking submission", async () => {
    const av = (state: "busy" | "saturated") => ({ state, running: 1, waiting: state === "busy" ? 0 : 2, max_concurrent: 1, max_waiting: 2 });
    const { unmount } = renderWorkspace(makeApi({ health: vi.fn().mockResolvedValue(health({ ready: true, availability: av("busy") })) }));
    expect(await screen.findByRole("button", { name: /^model server busy$/i })).toBeInTheDocument();
    unmount();
    renderWorkspace(makeApi({ health: vi.fn().mockResolvedValue(health({ ready: true, availability: av("saturated") })) }));
    expect(await screen.findByRole("button", { name: /model server saturated/i })).toBeInTheDocument();
  });

  it("lists artifact checks and availability in the popover, without paths", async () => {
    const { user } = renderWorkspace(
      makeApi({ health: vi.fn().mockResolvedValue(health({ ready: true, checks: { executable: true, base_gguf: true, lora_gguf: true }, availability: { state: "idle", running: 0, waiting: 0, max_concurrent: 1, max_waiting: 2 } })) }),
    );
    await user.click(await screen.findByRole("button", { name: /model server ready/i }));
    const panel = screen.getByRole("region", { name: /backend status details/i });
    expect(panel).toHaveTextContent(/executable ✓/);
    expect(panel).toHaveTextContent(/0\/1 running, 0\/2 waiting/);
    expect(panel.textContent).not.toMatch(/[A-Z]:\\|\/home\//);
  });
});
