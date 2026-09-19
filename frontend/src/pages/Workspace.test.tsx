import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { QueryResponse } from "../api/types";
import { DB_DEMO, failureResponse, HEALTH_UNCONFIGURED, makeApi, makeResponse } from "../test/fixtures";
import { renderWorkspace } from "../test/render";

const Q = "How many employees are there?";

async function ready(api = makeApi()) {
  const utils = renderWorkspace(api);
  await screen.findByRole("radio", { name: /demo/i });
  return { api, ...utils };
}

async function ask(user: ReturnType<typeof renderWorkspace>["user"], question = Q) {
  await user.type(screen.getByLabelText(/your question/i), question);
  await user.click(screen.getByRole("button", { name: /run query/i }));
}

const deferred = <T,>() => {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

describe("Workspace: loading databases and health", () => {
  it("lists registered databases, preselects the first, and shows the model as ready", async () => {
    await ready();
    expect(screen.getByRole("radio", { name: /demo/i })).toBeChecked();
    expect(screen.getByRole("radio", { name: /sales/i })).not.toBeChecked();
    expect(await screen.findByRole("button", { name: /local model ready/i })).toBeInTheDocument();
  });

  it("lets the user switch database", async () => {
    const { user } = await ready();
    await user.click(screen.getByRole("radio", { name: /sales/i }));
    expect(screen.getByRole("radio", { name: /sales/i })).toBeChecked();
  });

  it("explains an unreachable backend with a retry instead of a raw error", async () => {
    const api = makeApi({
      health: vi.fn().mockRejectedValue(new ApiError({ kind: "network", message: "x" })),
      databases: vi.fn().mockRejectedValueOnce(new ApiError({ kind: "network", message: "x" })).mockResolvedValue([DB_DEMO]),
    });
    const { user } = renderWorkspace(api);
    expect(await screen.findByText(/can't reach the backend/i)).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /backend offline/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run query/i })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(await screen.findByRole("radio", { name: /demo/i })).toBeInTheDocument();
  });

  it("shows an empty-state when no databases are registered", async () => {
    renderWorkspace(makeApi({ databases: vi.fn().mockResolvedValue([]) }));
    expect(await screen.findByText(/no databases are registered/i)).toBeInTheDocument();
  });

  it("blocks running (with the reason) when the model is not configured", async () => {
    await ready(makeApi({ health: vi.fn().mockResolvedValue(HEALTH_UNCONFIGURED) }));
    await screen.findByRole("button", { name: /model not configured/i });
    await waitFor(() => expect(screen.getByRole("button", { name: /run query/i })).toBeDisabled());
    expect(screen.getByText(/local model isn't configured/i)).toBeInTheDocument();
  });

  it("does not poll: health is requested once on mount", async () => {
    const { api } = await ready();
    await new Promise((r) => setTimeout(r, 50));
    expect(api.health).toHaveBeenCalledTimes(1);
  });
});

describe("Workspace: running a query", () => {
  it("submits the question, then shows SQL, safety state, the result table and metadata", async () => {
    const { api, user } = await ready();
    await ask(user);

    expect(api.query).toHaveBeenCalledTimes(1);
    expect(api.query.mock.calls[0]![0]).toEqual({ database_id: "demo", question: Q }); // no context key when empty

    const sql = await screen.findByLabelText("Generated SQL");
    expect(sql).toHaveTextContent("SELECT COUNT(emp_id) FROM employees");
    expect(screen.getByText(/safety passed/i)).toBeInTheDocument();
    expect(screen.getByText("Read-only")).toBeInTheDocument();
    expect(screen.getByText("Query completed")).toBeInTheDocument();

    const table = screen.getByRole("region", { name: /query results table/i });
    expect(within(table).getByRole("columnheader", { name: "COUNT(emp_id)" })).toBeInTheDocument();
    expect(within(table).getByRole("cell", { name: "12" })).toBeInTheDocument();

    const summary = screen.getByRole("list", { name: /run summary/i });
    expect(summary).toHaveTextContent("6.41 s");
    expect(summary).toHaveTextContent("10.9 tok/s");
  });

  it("submits business context when provided", async () => {
    const { api, user } = await ready();
    await user.click(screen.getByRole("button", { name: /add business context/i }));
    await user.type(screen.getByLabelText(/business context/i), "Engineering is departments.name");
    await ask(user);
    expect(api.query.mock.calls[0]![0]).toEqual({ database_id: "demo", question: Q, business_context: "Engineering is departments.name" });
  });

  it("supports Ctrl+Enter from the keyboard", async () => {
    const { api, user } = await ready();
    await user.type(screen.getByLabelText(/your question/i), Q);
    await user.keyboard("{Control>}{Enter}{/Control}");
    await waitFor(() => expect(api.query).toHaveBeenCalledTimes(1));
  });

  it("keeps Run disabled for a blank or over-long question", async () => {
    const { user } = await ready();
    const run = screen.getByRole("button", { name: /run query/i });
    expect(run).toBeDisabled();
    await user.type(screen.getByLabelText(/your question/i), "   ");
    expect(run).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/your question/i), { target: { value: "x".repeat(2001) } });
    expect(run).toBeDisabled();
    expect(screen.getByText(/shorten the question/i)).toBeInTheDocument();
  });

  it("fills the question (and context) from a demo example", async () => {
    const { user } = await ready();
    await user.click(screen.getByRole("button", { name: /engineering payroll/i }));
    expect(screen.getByLabelText(/your question/i)).toHaveValue("What is the total salary of employees in the Engineering department?");
    expect(screen.getByLabelText(/business context/i)).toHaveValue("Engineering is a department name stored in departments.name.");
  });

  it("shows an honest loading state (real elapsed clock, no fake percentage) and can be resolved", async () => {
    const d = deferred<QueryResponse>();
    const { user } = await ready(makeApi({ query: vi.fn().mockReturnValue(d.promise) }));
    await ask(user);

    const status = await screen.findByRole("status", { name: /generating sql/i });
    expect(status).toHaveTextContent(/forging your sql/i);
    expect(status).toHaveTextContent(/10–30 seconds/);
    expect(status).toHaveTextContent(/no live progress signal/i);
    expect(status.textContent).not.toMatch(/\d+\s?%/);
    expect(screen.getByLabelText(/your question/i)).toBeDisabled();

    d.resolve(makeResponse());
    expect(await screen.findByText("Query completed")).toBeInTheDocument();
    expect(screen.queryByRole("status", { name: /generating sql/i })).not.toBeInTheDocument();
  });

  it("can cancel an in-flight request", async () => {
    const query = vi.fn((_req: unknown, signal?: AbortSignal) => new Promise<QueryResponse>((_res, rej) => signal?.addEventListener("abort", () => rej(new ApiError({ kind: "aborted", message: "cancelled" })))));
    const { user } = await ready(makeApi({ query: query as never }));
    await ask(user);
    await user.click(await screen.findByRole("button", { name: /cancel request/i }));
    expect(await screen.findByText("Request cancelled")).toBeInTheDocument();
    expect(screen.queryByText(/this session/i)).not.toBeInTheDocument(); // cancelled requests are not history
  });
});

describe("Workspace: results", () => {
  it("indicates truncation without claiming a total", async () => {
    const rows = Array.from({ length: 500 }, (_, i) => [i]);
    const response = makeResponse({ result: { columns: ["n"], rows, returned_row_count: 500, truncated: true, max_rows: 500, elapsed_ms: 3 } });
    const { user } = await ready(makeApi({ query: vi.fn().mockResolvedValue(response) }));
    await ask(user);
    const note = await screen.findByRole("note");
    expect(note).toHaveTextContent("Results truncated");
    expect(note).toHaveTextContent("500 rows");
    expect(note).toHaveTextContent("The total isn't known");
    expect(screen.getByText("first 500 rows")).toBeInTheDocument(); // count chip
    // progressive rendering keeps a full page fast
    expect(screen.getByText(/showing 100 of 500 returned rows/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /show 200 more/i }));
    expect(screen.getByText(/showing 300 of 500/i)).toBeInTheDocument();
  });

  it("renders NULL, booleans, numbers and text distinctly, all as plain text", async () => {
    const response = makeResponse({
      generated_sql: "SELECT * FROM t",
      result: { columns: ["a", "b", "c", "d"], rows: [[null, true, 1234567, "<img src=x onerror=alert(1)>"]], returned_row_count: 1, truncated: false, max_rows: 500, elapsed_ms: 1 },
    });
    const { container, user } = await ready(makeApi({ query: vi.fn().mockResolvedValue(response) }));
    await ask(user);
    await screen.findByText("Query completed");
    expect(screen.getByText("NULL")).toBeInTheDocument();
    expect(screen.getByText("true")).toBeInTheDocument();
    expect(screen.getByText("1,234,567")).toBeInTheDocument();
    expect(screen.getByText("<img src=x onerror=alert(1)>")).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
  });

  it("renders generated SQL strictly as text", async () => {
    const evil = "SELECT '</code><script>alert(1)</script>'";
    const { container, user } = await ready(makeApi({ query: vi.fn().mockResolvedValue(makeResponse({ generated_sql: evil })) }));
    await ask(user);
    expect(await screen.findByLabelText("Generated SQL")).toHaveTextContent(evil);
    expect(container.querySelector("script")).toBeNull();
  });

  it("shows a designed empty result", async () => {
    const response = makeResponse({ result: { columns: ["n"], rows: [], returned_row_count: 0, truncated: false, max_rows: 500, elapsed_ms: 1 } });
    const { user } = await ready(makeApi({ query: vi.fn().mockResolvedValue(response) }));
    await ask(user);
    expect(await screen.findByText("No rows returned")).toBeInTheDocument();
  });

  it("makes wide tables keyboard-scrollable", async () => {
    const cols = Array.from({ length: 30 }, (_, i) => `column_${i}`);
    const response = makeResponse({ result: { columns: cols, rows: [cols.map((_, i) => i)], returned_row_count: 1, truncated: false, max_rows: 500, elapsed_ms: 1 } });
    const { user } = await ready(makeApi({ query: vi.fn().mockResolvedValue(response) }));
    await ask(user);
    expect(await screen.findByRole("region", { name: /query results table/i })).toHaveAttribute("tabindex", "0");
  });

  it("copies the SQL and confirms it", async () => {
    const { user } = await ready();
    await ask(user);
    await user.click(await screen.findByRole("button", { name: "Copy SQL" }));
    expect(await navigator.clipboard.readText()).toBe("SELECT COUNT(emp_id) FROM employees");
    expect(await screen.findByText("Copied to clipboard")).toBeInTheDocument();
  });

  it("offers a chart only when the shape supports it, and switching keeps the table available", async () => {
    const chartable = makeResponse({ result: { columns: ["dept", "total"], rows: [["Eng", 625000], ["Sales", 275000]], returned_row_count: 2, truncated: false, max_rows: 500, elapsed_ms: 1 } });
    const { user } = await ready(makeApi({ query: vi.fn().mockResolvedValue(chartable) }));
    await ask(user);
    await user.click(await screen.findByRole("button", { name: /chart/i }));
    expect(screen.getByRole("img", { name: /bar chart of total by dept, 2 points/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /table/i }));
    expect(screen.getByRole("region", { name: /query results table/i })).toBeInTheDocument();
  });

  it("does not offer a chart for a single value", async () => {
    const { user } = await ready();
    await ask(user);
    await screen.findByText("Query completed");
    expect(screen.queryByRole("group", { name: /result view/i })).not.toBeInTheDocument();
  });
});

describe("Workspace: failure states are intentional", () => {
  const cases: [string, QueryResponse | ApiError, RegExp, RegExp?][] = [
    ["unsafe SQL", failureResponse("unsafe_sql", "safety", "not_read_only_query", {
      generated_sql: "DROP TABLE employees",
      safety: { allowed: false, reasons: [{ code: "not_read_only_query", message: "Only SELECT / WITH queries are allowed." }] },
    }), /blocked by the safety policy/i, /never executed/i],
    ["a timeout", failureResponse("execution_error", "execution", "timeout"), /ran out of time/i],
    ["a model failure", failureResponse("model_error", "model", "model_error"), /couldn't produce sql/i],
    ["an unconfigured model", failureResponse("model_error", "model", "model_not_configured"), /isn't configured/i],
    ["an execution error", failureResponse("execution_error", "execution", "sql_error", { generated_sql: "SELECT nope FROM t", error: { stage: "execution", code: "sql_error", message: "no such column: nope" } }), /couldn't be executed/i],
    ["a schema error", failureResponse("schema_error", "schema", "schema_error"), /schema couldn't be read/i],
    ["an unreachable backend", new ApiError({ kind: "network", message: "x" }), /can't reach the schemaforge backend/i],
    ["a client timeout", new ApiError({ kind: "timeout", message: "x" }), /no answer within the wait limit/i],
    ["a malformed response", new ApiError({ kind: "malformed", message: "Unexpected response shape: timings" }), /backend sent something unexpected/i],
    ["an unknown database", new ApiError({ kind: "http", message: "Unknown database id.", status: 404, code: "unknown_database" }), /isn't registered/i],
    ["a validation error", new ApiError({ kind: "http", message: "x", status: 422, code: "invalid_request", fields: [{ field: "question", issue: "string_too_long" }] }), /wasn't valid/i, /question: string_too_long/],
  ];

  it.each(cases)("renders %s", async (_name, outcome, title, extra) => {
    const query = outcome instanceof ApiError ? vi.fn().mockRejectedValue(outcome) : vi.fn().mockResolvedValue(outcome);
    const { user } = await ready(makeApi({ query }));
    await ask(user);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(title);
    if (extra) expect(document.body).toHaveTextContent(extra);
    expect(screen.queryByRole("region", { name: /query results table/i })).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/Traceback|Exception|at \w+\.\w+ \(/);
  });

  it("shows the blocked SQL as an unexecuted artifact, with the rejection reason", async () => {
    const blocked = failureResponse("unsafe_sql", "safety", "not_read_only_query", {
      generated_sql: "DROP TABLE employees",
      safety: { allowed: false, reasons: [{ code: "not_read_only_query", message: "Only SELECT / WITH queries are allowed." }] },
    });
    const { user } = await ready(makeApi({ query: vi.fn().mockResolvedValue(blocked) }));
    await ask(user);
    expect(await screen.findByLabelText("Blocked SQL (never executed)")).toHaveTextContent("DROP TABLE employees");
    expect(screen.getByText("Only SELECT / WITH queries are allowed.")).toBeInTheDocument();
  });

  it("retries the same request after a failure", async () => {
    const query = vi.fn().mockRejectedValueOnce(new ApiError({ kind: "network", message: "x" })).mockResolvedValue(makeResponse());
    const { user } = await ready(makeApi({ query }));
    await ask(user);
    await user.click(await screen.findByRole("button", { name: /try again/i }));
    expect(await screen.findByText("Query completed")).toBeInTheDocument();
    expect(query).toHaveBeenCalledTimes(2);
    expect(query.mock.calls[1]![0]).toEqual(query.mock.calls[0]![0]);
  });

  it("re-checks backend health after a network failure", async () => {
    const api = makeApi({ query: vi.fn().mockRejectedValue(new ApiError({ kind: "network", message: "x" })) });
    const { user } = await ready(api);
    await ask(user);
    await screen.findByRole("alert");
    await waitFor(() => expect(api.health).toHaveBeenCalledTimes(2));
  });
});

describe("Workspace: session history", () => {
  it("records queries and reopens them without calling the backend again", async () => {
    const { api, user } = await ready();
    await ask(user);
    await screen.findByText("Query completed");
    const item = await screen.findByRole("button", { name: new RegExp(Q.slice(0, 20), "i") });
    fireEvent.change(screen.getByLabelText(/your question/i), { target: { value: "something else" } });
    await user.click(item);
    expect(screen.getByLabelText(/your question/i)).toHaveValue(Q);
    expect(screen.getByLabelText("Generated SQL")).toHaveTextContent("SELECT COUNT(emp_id) FROM employees");
    expect(api.query).toHaveBeenCalledTimes(1);
  });

  it("clears history", async () => {
    const { user } = await ready();
    await ask(user);
    await screen.findByText("Query completed");
    await user.click(await screen.findByRole("button", { name: /^clear$/i }));
    expect(screen.queryByText(/this session/i)).not.toBeInTheDocument();
  });
});
