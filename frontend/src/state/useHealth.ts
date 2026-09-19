import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/errors";
import type { HealthResponse } from "../api/types";
import { useApi } from "./api-context";

export type HealthState =
  | { status: "loading" }
  | { status: "online"; health: HealthResponse; checkedAt: number }
  | { status: "offline"; error: ApiError; checkedAt: number };

/** Re-check when the tab regains focus only if the last check is older than this. */
const STALE_MS = 60_000;

/**
 * Backend/model availability. Checked on mount, on demand (`refresh`), and when
 * the tab regains focus after going stale -- deliberately no interval polling.
 */
export function useHealth() {
  const api = useApi();
  const [state, setState] = useState<HealthState>({ status: "loading" });
  const lastChecked = useRef(0);
  const inflight = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    inflight.current?.abort();
    const controller = new AbortController();
    inflight.current = controller;
    try {
      const health = await api.health(controller.signal);
      if (controller.signal.aborted) return;
      lastChecked.current = Date.now();
      setState({ status: "online", health, checkedAt: lastChecked.current });
    } catch (e) {
      if (controller.signal.aborted) return;
      lastChecked.current = Date.now();
      const error = e instanceof ApiError ? e : new ApiError({ kind: "network", message: "Health check failed." });
      setState({ status: "offline", error, checkedAt: lastChecked.current });
    }
  }, [api]);

  useEffect(() => {
    void refresh();
    const onFocus = () => {
      if (Date.now() - lastChecked.current > STALE_MS) void refresh();
    };
    window.addEventListener("focus", onFocus);
    return () => {
      window.removeEventListener("focus", onFocus);
      inflight.current?.abort();
    };
  }, [refresh]);

  return { state, refresh };
}
