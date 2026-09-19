import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/errors";
import type { QueryRequest } from "../api/types";
import type { Outcome } from "../lib/failure";
import { useApi } from "./api-context";

export type RunState =
  | { phase: "idle" }
  | { phase: "running"; startedAt: number; request: QueryRequest }
  | { phase: "settled"; request: QueryRequest; outcome: Outcome; elapsedMs: number };

interface Options {
  onSettled?: (request: QueryRequest, outcome: Outcome) => void;
}

/** Owns one in-flight query at a time, with real cancellation (AbortController). */
export function useQueryRunner({ onSettled }: Options = {}) {
  const api = useApi();
  const [state, setState] = useState<RunState>({ phase: "idle" });
  const controller = useRef<AbortController | null>(null);
  const settledCb = useRef(onSettled);
  useEffect(() => {
    settledCb.current = onSettled;
  }, [onSettled]);

  const run = useCallback(
    async (request: QueryRequest) => {
      controller.current?.abort();
      const ctl = new AbortController();
      controller.current = ctl;
      const startedAt = Date.now();
      setState({ phase: "running", startedAt, request });
      let outcome: Outcome;
      try {
        outcome = { kind: "response", response: await api.query(request, ctl.signal) };
      } catch (e) {
        const error = e instanceof ApiError ? e : new ApiError({ kind: "network", message: "Unexpected client error." });
        outcome = { kind: "error", error };
      }
      if (controller.current !== ctl) return; // superseded by a newer run
      controller.current = null;
      setState({ phase: "settled", request, outcome, elapsedMs: Date.now() - startedAt });
      settledCb.current?.(request, outcome);
    },
    [api],
  );

  const cancel = useCallback(() => controller.current?.abort(), []);
  const restore = useCallback((request: QueryRequest, outcome: Outcome) => {
    controller.current?.abort();
    controller.current = null;
    setState({ phase: "settled", request, outcome, elapsedMs: 0 });
  }, []);
  const reset = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    setState({ phase: "idle" });
  }, []);

  useEffect(() => () => controller.current?.abort(), []);
  return { state, run, cancel, restore, reset };
}
