import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BENCHMARKS, DEPLOYMENT } from "../data/results";
import { Landing } from "./Landing";

beforeEach(() => {
  // Reduced motion: no typewriter timers, bars render at their final state.
  window.matchMedia = ((q: string) => ({ matches: q.includes("reduce"), media: q, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, onchange: null, dispatchEvent: () => false })) as never;
});

describe("Landing", () => {
  it("explains the product and offers a route into the workspace", async () => {
    const navigate = vi.fn();
    render(<Landing navigate={navigate} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/plain english in/i);
    expect(screen.getByText(/entirely on your own hardware/i, { exact: false })).toBeInTheDocument();
    const cta = screen.getAllByRole("link", { name: /open the workspace/i })[0]!;
    expect(cta).toHaveAttribute("href", "/workspace");
    await userEvent.click(cta);
    expect(navigate).toHaveBeenCalledWith("workspace");
  });

  it("shows the three frozen benchmarks with their own metrics and exact values", () => {
    render(<Landing navigate={vi.fn()} />);
    const expected: Record<string, { metric: RegExp; values: string[]; delta: string }> = {
      "Seen training-set specialization": { metric: /normalized sql exact match/i, values: ["4.20%", "37.40%"], delta: "+33.20" },
      "Schema-held-out validation": { metric: /execution accuracy.*sqlite comparator/i, values: ["39.14%", "44.57%"], delta: "+5.43" },
      "BIRD MiniDev": { metric: /^execution accuracy$/i, values: ["43.60%", "44.80%"], delta: "+1.20" },
    };
    for (const [title, want] of Object.entries(expected)) {
      const card = screen.getByRole("article", { name: title });
      expect(within(card).getByText(want.metric)).toBeInTheDocument();
      for (const v of want.values) expect(within(card).getByText(v)).toBeInTheDocument();
      expect(within(card).getByText(want.delta)).toBeInTheDocument();
      expect(within(card).getByText(/percentage points/i)).toBeInTheDocument();
    }
    expect(screen.getAllByRole("article")).toHaveLength(3);
  });

  it("keeps the three benchmarks separate and scientifically labelled", () => {
    render(<Landing navigate={vi.fn()} />);
    const seen = screen.getByRole("article", { name: "Seen training-set specialization" });
    expect(seen).toHaveTextContent(/specialization, not generalization/i);
    expect(screen.getByRole("article", { name: "Schema-held-out validation" })).toHaveTextContent(/held-out database schemas/i);
    expect(screen.getByRole("article", { name: "BIRD MiniDev" })).toHaveTextContent(/never used in training/i);
    expect(document.body).toHaveTextContent(/never averaged into one score/i);
    expect(document.body.textContent).not.toMatch(/combined score|overall accuracy|average accuracy/i);
    // data module is the single source: three distinct metrics, no derived aggregate
    expect(new Set(BENCHMARKS.map((b) => b.metric)).size).toBe(3);
    for (const b of BENCHMARKS) expect(+(b.tuned - b.base).toFixed(2)).toBe(b.deltaPp);
  });

  it("reports deployment engineering honestly, including the quantization caveat", () => {
    render(<Landing navigate={vi.fn()} />);
    const sizes = screen.getByRole("group", { name: /model size comparison/i });
    expect(sizes).toHaveTextContent("7.50 GB");
    expect(sizes).toHaveTextContent("2.33 GB");
    expect(sizes).toHaveTextContent("2.39 GB");
    expect(document.body).toHaveTextContent(String(DEPLOYMENT.cpuTokensPerSecond));
    expect(document.body).toHaveTextContent(/~69%/);
    const caveat = screen.getByLabelText(/quantization caveat/i);
    expect(caveat).toHaveTextContent(/only 4 of 10 prompts/i);
    expect(caveat).toHaveTextContent(/equivalence is not claimed/i);
  });

  it("does not fabricate social proof or traffic", () => {
    render(<Landing navigate={vi.fn()} />);
    expect(document.body.textContent).not.toMatch(/trusted by|customers|uptime|\d+\+? (users|companies)|testimonial/i);
  });

  it("presents the safety model as deterministic, two-layer", () => {
    render(<Landing navigate={vi.fn()} />);
    const safety = document.getElementById("safety")!;
    expect(safety).toHaveTextContent(/AST safety policy/i);
    expect(safety).toHaveTextContent(/independent read-only executor/i);
  });
});
