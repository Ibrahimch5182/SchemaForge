import { useMemo, type ReactNode } from "react";
import { tokenizeSql } from "../lib/sqlHighlight";
import { CopyButton } from "./CopyButton";

interface Props {
  sql: string;
  title?: string;
  variant?: "default" | "blocked";
  badge?: ReactNode;
}

/**
 * Generated SQL as a first-class artifact. Rendered strictly as text nodes
 * (each token is a <span> with a text child): never as HTML, never executable.
 */
export function SqlBlock({ sql, title = "Generated SQL", variant = "default", badge }: Props) {
  const tokens = useMemo(() => tokenizeSql(sql), [sql]);
  return (
    <figure className={`sqlblock ${variant === "blocked" ? "is-blocked" : ""}`}>
      <figcaption className="sqlblock-head">
        <span className="sqlblock-title">{title}</span>
        <span className="sqlblock-tools">
          {badge}
          <CopyButton text={sql} label="Copy SQL" />
        </span>
      </figcaption>
      {/* Scrollable code must be keyboard-focusable (axe: scrollable-region-focusable). */}
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */}
      <pre className="sqlblock-pre" tabIndex={0} aria-label={title}>
        <code>
          {tokens.map((t, i) =>
            t.type === "space" ? t.text : (
              <span key={i} className={`tok tok-${t.type}`}>
                {t.text}
              </span>
            ),
          )}
        </code>
      </pre>
    </figure>
  );
}
