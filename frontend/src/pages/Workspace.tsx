import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { QueryRequest } from "../api/types";
import { DatabasePicker } from "../components/DatabasePicker";
import { FailurePanel } from "../components/FailurePanel";
import { HistoryList } from "../components/HistoryList";
import { Database, Lock, Shield, Sparkle } from "../components/icons";
import { LoadingPanel } from "../components/LoadingPanel";
import { QueryForm, type Draft, type ExampleQuestion } from "../components/QueryForm";
import { ResultView } from "../components/ResultView";
import { SiteHeader } from "../components/SiteHeader";
import { StatusPill } from "../components/StatusPill";
import { classifyOutcome } from "../lib/failure";
import { prefersReducedMotion } from "../lib/hooks";
import type { Route } from "../lib/route";
import { useDatabases } from "../state/useDatabases";
import { useHealth } from "../state/useHealth";
import { useHistory, type HistoryEntry } from "../state/history";
import { useQueryRunner } from "../state/useQueryRunner";

/** Example prompts for the bundled demo database only (from the Phase 8 smoke run). */
const DEMO_EXAMPLES: ExampleQuestion[] = [
  { label: "How many employees are there?", question: "How many employees are there?" },
  {
    label: "Engineering payroll",
    question: "What is the total salary of employees in the Engineering department?",
    context: "Engineering is a department name stored in departments.name.",
  },
];

export function Workspace({ navigate }: { navigate: (r: Route) => void }) {
  const { state: health, refresh: refreshHealth } = useHealth();
  const { state: dbs, reload: reloadDbs } = useDatabases();
  const history = useHistory();
  const [pickedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft>({ question: "", context: "" });
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);
  const focusOnSettle = useRef(false);

  const runner = useQueryRunner({
    onSettled: (request, outcome) => {
      history.add(request, outcome);
      // A transport failure is a good moment to re-check the backend (no polling otherwise).
      if (outcome.kind === "error" && outcome.error.kind === "network") void refreshHealth();
    },
  });

  // Default to the first registered database; a stale pick (after a reload) falls back to it.
  const databases = dbs.status === "ready" ? dbs.databases : [];
  const selectedId = pickedId && databases.some((d) => d.id === pickedId) ? pickedId : (databases[0]?.id ?? null);

  const running = runner.state.phase === "running";

  const blockedReason = useMemo(() => {
    if (dbs.status === "loading") return "Loading databases…";
    if (dbs.status === "error") return "Databases are unavailable, so there's nothing to query yet.";
    if (!selectedId) return "Register a database on the backend to get started.";
    if (health.status === "online" && !health.health.model_runtime.configured) return "The local model isn't configured on the backend yet.";
    return null;
  }, [dbs, selectedId, health]);

  const buildRequest = useCallback(
    (): QueryRequest => ({
      database_id: selectedId ?? "",
      question: draft.question.trim(),
      ...(draft.context.trim() ? { business_context: draft.context.trim() } : {}),
    }),
    [selectedId, draft],
  );

  const submit = useCallback(() => {
    focusOnSettle.current = true;
    setActiveHistoryId(null);
    void runner.run(buildRequest());
  }, [runner, buildRequest]);

  const retry = useCallback(() => {
    if (runner.state.phase !== "settled") return;
    focusOnSettle.current = true;
    void runner.run(runner.state.request);
  }, [runner]);

  const openHistory = useCallback(
    (entry: HistoryEntry) => {
      focusOnSettle.current = true;
      setActiveHistoryId(entry.id);
      if (dbs.status === "ready" && dbs.databases.some((d) => d.id === entry.request.database_id)) setSelectedId(entry.request.database_id);
      setDraft({ question: entry.request.question, context: entry.request.business_context ?? "" });
      runner.restore(entry.request, entry.outcome);
    },
    [dbs, runner],
  );

  // Move focus to the result once it settles so keyboard/screen-reader users land on it.
  const phase = runner.state.phase;
  useEffect(() => {
    if (phase !== "settled" || !focusOnSettle.current) return;
    focusOnSettle.current = false;
    const el = outputRef.current;
    if (!el) return;
    el.focus({ preventScroll: true });
    el.scrollIntoView?.({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  }, [phase]);

  const examples = selectedId === "demo" ? DEMO_EXAMPLES : [];
  const classified = runner.state.phase === "settled" ? classifyOutcome(runner.state.outcome) : null;

  return (
    <div className="app-shell">
      <SiteHeader route="workspace" navigate={navigate} right={<StatusPill state={health} onRefresh={() => void refreshHealth()} />} />

      <main className="workspace" id="main">
        <div className="workspace-head">
          <h1>Query workspace</h1>
          <p>Ask a question in plain English. SchemaForge writes one read-only SQL query, checks it, and runs it against the database you choose.</p>
        </div>

        <div className="workspace-grid">
          <aside className="workspace-side" aria-label="Databases and history">
            <DatabasePicker state={dbs} selectedId={selectedId} onSelect={setSelectedId} onReload={reloadDbs} disabled={running} />
            <HistoryList entries={history.entries} activeId={activeHistoryId} onOpen={openHistory} onClear={history.clear} disabled={running} />
          </aside>

          <div className="workspace-main">
            <QueryForm
              draft={draft}
              onChange={setDraft}
              onSubmit={submit}
              onCancel={runner.cancel}
              running={running}
              blockedReason={blockedReason}
              examples={examples}
            />

            <div className="output" ref={outputRef} tabIndex={-1} aria-label="Query output">
              {runner.state.phase === "idle" && <EmptyState />}
              {runner.state.phase === "running" && <LoadingPanel startedAt={runner.state.startedAt} onCancel={runner.cancel} />}
              {classified && (classified.ok ? <ResultView response={classified.response} /> : <FailurePanel failure={classified.failure} response={classified.response} onRetry={retry} />)}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function EmptyState() {
  return (
    <section className="panel empty-state" aria-label="Getting started">
      <div className="empty-mark" aria-hidden="true">
        <Sparkle size={26} />
      </div>
      <h2>Ask your first question</h2>
      <p>Pick a database, describe what you want to know, and run it. Results appear here with the SQL the model wrote.</p>
      <ul className="empty-points">
        <li>
          <Database size={16} /> Only registered databases; no file paths, ever
        </li>
        <li>
          <Shield size={16} /> Every query is parsed and must be a single read-only SELECT
        </li>
        <li>
          <Lock size={16} /> Executed on a read-only connection with time and row limits
        </li>
      </ul>
    </section>
  );
}
