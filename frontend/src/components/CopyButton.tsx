import { useEffect, useRef, useState } from "react";
import { copyText } from "../lib/clipboard";
import { Check, Copy } from "./icons";

interface Props {
  text: string;
  label: string; // e.g. "Copy SQL"
  className?: string;
  compact?: boolean;
}

/** Copy-to-clipboard with visible + screen-reader feedback. */
export function CopyButton({ text, label, className = "", compact = false }: Props) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => void (timer.current && clearTimeout(timer.current)), []);

  const onClick = async () => {
    const ok = await copyText(text);
    setState(ok ? "copied" : "failed");
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setState("idle"), 1800);
  };

  const text2 = state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : compact ? "Copy" : label;
  return (
    <>
      <button type="button" className={`btn btn-ghost btn-sm copy-btn ${state !== "idle" ? `is-${state}` : ""} ${className}`} onClick={onClick} aria-label={label}>
        {state === "copied" ? <Check size={15} /> : <Copy size={15} />}
        <span aria-hidden="true">{text2}</span>
      </button>
      <span className="sr-only" role="status" aria-live="polite">
        {state === "copied" ? "Copied to clipboard" : state === "failed" ? "Copy failed" : ""}
      </span>
    </>
  );
}
