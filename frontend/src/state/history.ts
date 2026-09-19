/**
 * Session-local query history (sessionStorage only: no server-side history, no
 * accounts). Entries are revisitable: they keep the request and the outcome.
 * Storage may be unavailable or full; every access is guarded and the UI works
 * without it.
 */
import { useCallback, useState } from "react";
import { ApiError, type ApiErrorKind } from "../api/errors";
import { parseQueryResponse } from "../api/guards";
import type { QueryRequest } from "../api/types";
import type { Outcome } from "../lib/failure";

const STORAGE_KEY = "schemaforge.history.v1";
export const MAX_HISTORY = 15;

export interface HistoryEntry {
  id: string;
  at: number;
  request: QueryRequest;
  outcome: Outcome;
}

type StoredOutcome =
  | { kind: "response"; response: unknown }
  | { kind: "error"; error: { kind: ApiErrorKind; message: string; status?: number; code?: string; requestId?: string | null } };

interface StoredEntry {
  id: string;
  at: number;
  request: QueryRequest;
  outcome: StoredOutcome;
}

function serialize(e: HistoryEntry): StoredEntry {
  const outcome: StoredOutcome =
    e.outcome.kind === "response"
      ? { kind: "response", response: e.outcome.response }
      : {
          kind: "error",
          error: { kind: e.outcome.error.kind, message: e.outcome.error.message, status: e.outcome.error.status, code: e.outcome.error.code, requestId: e.outcome.error.requestId },
        };
  return { id: e.id, at: e.at, request: e.request, outcome };
}

function revive(raw: unknown): HistoryEntry | null {
  try {
    const s = raw as StoredEntry;
    if (!s || typeof s.id !== "string" || typeof s.at !== "number" || typeof s.request?.question !== "string") return null;
    if (s.outcome.kind === "response") {
      return { id: s.id, at: s.at, request: s.request, outcome: { kind: "response", response: parseQueryResponse(s.outcome.response) } };
    }
    return { id: s.id, at: s.at, request: s.request, outcome: { kind: "error", error: new ApiError(s.outcome.error) } };
  } catch {
    return null; // a corrupt entry is dropped, never thrown
  }
}

export function loadHistory(storage: Storage | undefined = safeStorage()): HistoryEntry[] {
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map(revive).filter((e): e is HistoryEntry => e !== null).slice(0, MAX_HISTORY);
  } catch {
    return [];
  }
}

export function saveHistory(entries: HistoryEntry[], storage: Storage | undefined = safeStorage()): void {
  if (!storage) return;
  let list = entries.slice(0, MAX_HISTORY);
  // Result sets can be large: on quota errors, drop the oldest entries until it fits.
  while (list.length > 0) {
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify(list.map(serialize)));
      return;
    } catch {
      list = list.slice(0, -1);
    }
  }
  try {
    storage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

function safeStorage(): Storage | undefined {
  try {
    return window.sessionStorage;
  } catch {
    return undefined;
  }
}

let counter = 0;
export const newEntryId = () => `${Date.now().toString(36)}-${(counter++).toString(36)}`;

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(() => loadHistory());

  const add = useCallback((request: QueryRequest, outcome: Outcome) => {
    // A user-cancelled request is not history.
    if (outcome.kind === "error" && outcome.error.kind === "aborted") return;
    setEntries((prev) => {
      const next = [{ id: newEntryId(), at: Date.now(), request, outcome }, ...prev].slice(0, MAX_HISTORY);
      saveHistory(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setEntries([]);
    saveHistory([]);
  }, []);

  return { entries, add, clear };
}
