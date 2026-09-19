import type { CellValue } from "../api/types";

export function formatMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return "—";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

export function formatNumber(n: number): string {
  return Number.isInteger(n) ? n.toLocaleString("en-US") : n.toLocaleString("en-US", { maximumFractionDigits: 6 });
}

export type CellKind = "null" | "number" | "boolean" | "text";

export interface FormattedCell {
  text: string;
  kind: CellKind;
}

/** Display formatting only; never mutates the underlying value. Always plain text. */
export function formatCell(v: CellValue): FormattedCell {
  if (v === null) return { text: "NULL", kind: "null" };
  if (typeof v === "number") return { text: formatNumber(v), kind: "number" };
  if (typeof v === "boolean") return { text: v ? "true" : "false", kind: "boolean" };
  return { text: v, kind: "text" };
}

export function formatRate(v: unknown): string | null {
  return typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(1)} tok/s` : null;
}

export function shortId(id: string, n = 8): string {
  return id.length <= n ? id : id.slice(0, n);
}

export function clockTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
