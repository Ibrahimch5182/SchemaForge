import { createContext, useContext, type ReactNode } from "react";
import type { SchemaForgeApi } from "../api/client";

const ApiContext = createContext<SchemaForgeApi | null>(null);

export function ApiProvider({ api, children }: { api: SchemaForgeApi; children: ReactNode }) {
  return <ApiContext.Provider value={api}>{children}</ApiContext.Provider>;
}

export function useApi(): SchemaForgeApi {
  const api = useContext(ApiContext);
  if (!api) throw new Error("useApi must be used inside <ApiProvider>");
  return api;
}
