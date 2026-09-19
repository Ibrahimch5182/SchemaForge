const DEFAULT_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_QUERY_TIMEOUT_MS = 180_000;

export interface FrontendConfig {
  apiBaseUrl: string;
  queryTimeoutMs: number;
}

/** Reads Vite env. Pure (takes the env object) so it is unit-testable. */
export function readConfig(env: Record<string, string | undefined> = import.meta.env): FrontendConfig {
  const raw = (env.VITE_API_BASE_URL ?? "").trim() || DEFAULT_BASE_URL;
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`VITE_API_BASE_URL is not a valid URL: ${raw}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("VITE_API_BASE_URL must be http(s)");
  const timeout = Number(env.VITE_QUERY_TIMEOUT_MS);
  return {
    apiBaseUrl: raw.replace(/\/+$/, ""),
    queryTimeoutMs: Number.isFinite(timeout) && timeout > 0 ? timeout : DEFAULT_QUERY_TIMEOUT_MS,
  };
}
