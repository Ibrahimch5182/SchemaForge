/**
 * Frozen, measured project results (Phases 6-7). These are the ONLY numbers the
 * marketing surface displays; they are not computed, averaged or combined.
 * Each benchmark keeps its own metric and scope on purpose: they are three
 * different evaluations and must never be read as one score.
 */

export interface Benchmark {
  id: "seen" | "heldout" | "minidev";
  eyebrow: string;
  title: string;
  metric: string;
  scope: string;
  sampleSize: number;
  base: number;
  tuned: number;
  deltaPp: number;
  takeaway: string;
  caveat: string;
  tone: "seen" | "heldout" | "external";
}

export const BENCHMARKS: readonly Benchmark[] = [
  {
    id: "seen",
    eyebrow: "Specialization",
    title: "Seen training-set specialization",
    metric: "Normalized SQL exact match",
    scope: "Databases the model was fine-tuned on",
    sampleSize: 500,
    base: 34.2,
    tuned: 67.4,
    deltaPp: 33.2,
    takeaway: "The adapter learned the SchemaForge task on data it has seen.",
    caveat: "Seen-data result. This is specialization, not generalization to new schemas.",
    tone: "seen",
  },
  {
    id: "heldout",
    eyebrow: "Generalization",
    title: "Schema-held-out validation",
    metric: "Execution accuracy · SchemaForge SQLite comparator",
    scope: "Held-out database schemas the model never trained on",
    sampleSize: 534,
    base: 37.40,
    tuned: 60.60,
    deltaPp: 23.20,
    takeaway: "A real gain on schemas the model has never seen.",
    caveat: "Scored with our own SQLite execution comparator, not the official BIRD evaluator.",
    tone: "heldout",
  },
  {
    id: "minidev",
    eyebrow: "External benchmark",
    title: "BIRD MiniDev",
    metric: "Execution accuracy",
    scope: "Untouched external benchmark, never used in training",
    sampleSize: 500,
    base: 43.6,
    tuned: 59.03,
    deltaPp: 15.43,
    takeaway: "A modest improvement on an untouched external benchmark.",
    caveat: "Impressive margin on 500 examples; reported as measured, without spin.",
    tone: "external",
  },
];

export const DEPLOYMENT = {
  f16BaseGb: 7.498,
  q4BaseGb: 2.326,
  loraGb: 0.062,
  /** Q4_K_M base + LoRA, the effective deployed model. */
  effectiveGb: 2.388,
  reductionPct: 69,
  cpuTokensPerSecond: 10.86,
  peakRssGb: 5.7,
  sanityAgreement: { same: 4, of: 10 },
} as const;
