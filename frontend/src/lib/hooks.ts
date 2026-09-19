import { useEffect, useRef, useState, type RefObject } from "react";

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** True once the element has entered the viewport (one-shot). Falls back to true without IntersectionObserver. */
export function useInView<T extends Element>(threshold = 0.25): [RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === "undefined" || prefersReducedMotion());
  useEffect(() => {
    if (seen) return;
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { threshold },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen, threshold]);
  return [ref, seen];
}

/** Elapsed whole seconds since `startedAt`, ticking at 1 Hz while `active` (a real clock, not fake progress). */
export function useElapsedSeconds(startedAt: number | null, active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active || startedAt === null) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [active, startedAt]);
  return startedAt === null ? 0 : Math.max(0, Math.floor((now - startedAt) / 1000));
}

/**
 * Types `text` out character by character. Shows it all immediately with reduced
 * motion; while `enabled` is false (waiting for an earlier step) it shows nothing.
 */
export function useTypewriter(text: string, enabled: boolean, msPerChar = 18, startDelayMs = 0): string {
  const instant = prefersReducedMotion();
  // Progress is keyed by text so a new string restarts from 0 without a synchronous reset.
  const [progress, setProgress] = useState({ text, count: 0 });
  const count = progress.text === text ? progress.count : 0;
  useEffect(() => {
    if (instant || !enabled) return;
    let i = 0;
    let interval: ReturnType<typeof setInterval> | undefined;
    const start = setTimeout(() => {
      interval = setInterval(() => {
        i += 1;
        setProgress({ text, count: i });
        if (i >= text.length && interval) clearInterval(interval);
      }, msPerChar);
    }, startDelayMs);
    return () => {
      clearTimeout(start);
      if (interval) clearInterval(interval);
    };
  }, [text, instant, enabled, msPerChar, startDelayMs]);
  if (instant) return text;
  return enabled ? text.slice(0, count) : "";
}
