import { useMemo } from "react";
import { useTypewriter } from "../lib/hooks";
import { tokenizeSql } from "../lib/sqlHighlight";
import { Check, Lock, Shield } from "./icons";

/**
 * Hero illustration: one REAL example captured from the Phase 8 smoke run on the
 * bundled demo database. It is labelled as a captured example, not a live query.
 */
const QUESTION = "What is the total salary of employees in the Engineering department?";
const SQL = `SELECT SUM(T1.salary)
FROM employees AS T1
INNER JOIN departments AS T2
  ON T1.dept_id = T2.dept_id
WHERE T2.name = 'Engineering'`;

export function ForgeDemo() {
  const typedQuestion = useTypewriter(QUESTION, true, 16, 500);
  const questionDone = typedQuestion.length === QUESTION.length;
  const typedSql = useTypewriter(SQL, questionDone, 12, 250);
  const sqlDone = typedSql.length === SQL.length;
  const tokens = useMemo(() => tokenizeSql(typedSql), [typedSql]);

  return (
    <figure className="forge-demo" aria-label="Example: a question turned into SQL and a result, captured from a local run">
      <div className="forge-window">
        <div className="forge-bar" aria-hidden="true">
          <span />
          <span />
          <span />
          <em>demo database · captured run</em>
        </div>

        <div className="forge-step">
          <span className="forge-tag">Question</span>
          <p className="forge-q">
            {typedQuestion}
            {!questionDone && <span className="caret" aria-hidden="true" />}
          </p>
        </div>

        <div className="forge-arrow" aria-hidden="true">
          <span />
        </div>

        <div className="forge-step">
          <span className="forge-tag tag-sql">SQL · generated locally</span>
          <pre className="forge-sql" aria-label="Generated SQL example">
            <code>
              {tokens.map((t, i) => (t.type === "space" ? t.text : <span key={i} className={`tok tok-${t.type}`}>{t.text}</span>))}
              {questionDone && !sqlDone && <span className="caret" aria-hidden="true" />}
            </code>
          </pre>
        </div>

        <div className={`forge-result ${sqlDone ? "is-in" : ""}`}>
          <div className="forge-badges">
            <span className="badge badge-ok">
              <Shield size={13} /> Safety passed
            </span>
            <span className="badge badge-neutral">
              <Lock size={13} /> Read-only
            </span>
          </div>
          <div className="forge-value">
            <Check size={16} />
            <span className="forge-num">625,000</span>
            <span className="forge-unit">SUM(salary)</span>
          </div>
        </div>
      </div>
      <figcaption>Captured from the Phase 8 smoke run on the bundled demo database (line breaks added for display).</figcaption>
    </figure>
  );
}
