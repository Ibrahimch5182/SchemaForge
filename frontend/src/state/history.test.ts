import { describe, expect, it } from "vitest";
import { ApiError } from "../api/errors";
import { failureResponse, makeResponse } from "../test/fixtures";
import { loadHistory, MAX_HISTORY, saveHistory, type HistoryEntry } from "./history";

const entry = (i: number, outcome?: HistoryEntry["outcome"]): HistoryEntry => ({
  id: `e${i}`,
  at: 1_700_000_000_000 + i,
  request: { database_id: "demo", question: `question ${i}` },
  outcome: outcome ?? { kind: "response", response: makeResponse() },
});

describe("session history", () => {
  it("round-trips responses and errors through sessionStorage", () => {
    const entries = [
      entry(1),
      entry(2, { kind: "response", response: failureResponse("unsafe_sql", "safety", "x") }),
      entry(3, { kind: "error", error: new ApiError({ kind: "network", message: "down" }) }),
    ];
    saveHistory(entries);
    const back = loadHistory();
    expect(back.map((e) => e.id)).toEqual(["e1", "e2", "e3"]);
    expect(back[0]!.outcome.kind === "response" && back[0]!.outcome.response.result?.rows).toEqual([[12]]);
    const e3 = back[2]!.outcome;
    expect(e3.kind === "error" && e3.error).toBeInstanceOf(ApiError);
    expect(e3.kind === "error" && e3.error.kind).toBe("network");
  });

  it("caps the list", () => {
    saveHistory(Array.from({ length: MAX_HISTORY + 8 }, (_, i) => entry(i)));
    expect(loadHistory()).toHaveLength(MAX_HISTORY);
  });

  it("survives corrupt or hostile storage contents", () => {
    window.sessionStorage.setItem("schemaforge.history.v1", "{not json");
    expect(loadHistory()).toEqual([]);
    window.sessionStorage.setItem("schemaforge.history.v1", JSON.stringify([{ id: 1 }, null, "x", { id: "ok", at: 1, request: { question: "q" }, outcome: { kind: "response", response: { status: "ok" } } }]));
    expect(loadHistory()).toEqual([]);
  });

  it("degrades gracefully when storage is unavailable or full", () => {
    const full = { setItem: () => { throw new DOMException("quota", "QuotaExceededError"); }, removeItem: () => {}, getItem: () => null } as unknown as Storage;
    expect(() => saveHistory([entry(1), entry(2)], full)).not.toThrow();
    expect(loadHistory(undefined)).toBeDefined();
  });
});
