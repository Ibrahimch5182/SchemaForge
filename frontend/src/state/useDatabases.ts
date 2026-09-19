import { useCallback, useEffect, useState } from "react";
import { ApiError } from "../api/errors";
import type { DatabaseInfo } from "../api/types";
import { useApi } from "./api-context";

export type DatabasesState =
  | { status: "loading" }
  | { status: "ready"; databases: DatabaseInfo[] }
  | { status: "error"; error: ApiError };

export function useDatabases() {
  const api = useApi();
  const [state, setState] = useState<DatabasesState>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    api
      .databases(controller.signal)
      .then((databases) => {
        if (!controller.signal.aborted) setState({ status: "ready", databases });
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        const error = e instanceof ApiError ? e : new ApiError({ kind: "network", message: "Could not load databases." });
        setState({ status: "error", error });
      });
    return () => controller.abort();
  }, [api, attempt]);

  const reload = useCallback(() => {
    setState({ status: "loading" });
    setAttempt((n) => n + 1);
  }, []);
  return { state, reload };
}
