/**
 * Deterministic, client-side chart eligibility. No model is involved: the
 * decision is a pure function of the returned columns and rows, so the same
 * result always yields the same (explainable) answer. The table stays the
 * source of truth; a chart is only offered when the shape clearly supports it.
 */
import type { CellValue, ExecutionResult } from "../api/types";

export const MAX_CHART_CATEGORIES = 24;
export const MIN_CHART_POINTS = 2;

export type ChartKind = "bar" | "line";

export interface ChartPlan {
  eligible: true;
  kind: ChartKind;
  labelColumn: number;
  /** Candidate measure columns (all numeric); the UI lets the user switch between them. */
  measureColumns: number[];
}

export interface ChartRejection {
  eligible: false;
  reason: string;
}

export type ChartDecision = ChartPlan | ChartRejection;

const DATE_LIKE = /^\d{4}(-\d{2}(-\d{2})?)?$/;
const ID_LIKE = /(^|_)id$|^id$/i;

const isNumber = (v: CellValue): v is number => typeof v === "number" && Number.isFinite(v);

function columnValues(rows: CellValue[][], col: number): CellValue[] {
  return rows.map((r) => r[col] ?? null);
}

function isNumericColumn(values: CellValue[]): boolean {
  const present = values.filter((v) => v !== null);
  return present.length > 0 && present.every(isNumber);
}

function isTextColumn(values: CellValue[]): boolean {
  const present = values.filter((v) => v !== null);
  return present.length === values.length && present.every((v) => typeof v === "string" && v.trim() !== "");
}

export function analyzeChartability(result: ExecutionResult): ChartDecision {
  const { columns, rows } = result;
  if (columns.length < 2) return { eligible: false, reason: "A chart needs a label column and a numeric column." };
  if (rows.length < MIN_CHART_POINTS) return { eligible: false, reason: "A single row is better read in the table." };
  if (rows.length > MAX_CHART_CATEGORIES)
    return { eligible: false, reason: `More than ${MAX_CHART_CATEGORIES} categories are hard to read as a chart.` };

  const cols = columns.map((_, i) => columnValues(rows, i));
  const numeric = cols.map(isNumericColumn);

  // Label: first text column; otherwise the first numeric id-like column (e.g. dept_id).
  let labelColumn = cols.findIndex((v, i) => !numeric[i] && isTextColumn(v));
  if (labelColumn === -1) labelColumn = columns.findIndex((name, i) => numeric[i] && ID_LIKE.test(name));
  if (labelColumn === -1) return { eligible: false, reason: "No text column to label the categories." };

  const labels = cols[labelColumn]!.map(String);
  if (new Set(labels).size !== labels.length) return { eligible: false, reason: "Category labels repeat, so bars would be ambiguous." };

  let measures = columns.map((_, i) => i).filter((i) => i !== labelColumn && numeric[i]);
  const nonId = measures.filter((i) => !ID_LIKE.test(columns[i]!));
  if (nonId.length > 0) measures = nonId; // prefer real measures over id columns
  if (measures.length === 0) return { eligible: false, reason: "No numeric column to plot." };

  const allNull = (i: number) => cols[i]!.every((v) => v === null);
  measures = measures.filter((i) => !allNull(i));
  if (measures.length === 0) return { eligible: false, reason: "The numeric columns contain no values." };

  const dateLike = labels.every((l) => DATE_LIKE.test(l));
  return { eligible: true, kind: dateLike ? "line" : "bar", labelColumn, measureColumns: measures };
}

export interface ChartPoint {
  label: string;
  value: number | null;
}

export function extractSeries(result: ExecutionResult, labelColumn: number, measureColumn: number): ChartPoint[] {
  return result.rows.map((r) => {
    const v = r[measureColumn] ?? null;
    return { label: String(r[labelColumn] ?? ""), value: isNumber(v) ? v : null };
  });
}
