import { BENCHMARKS, DEPLOYMENT, type Benchmark } from "../data/results";
import { ForgeDemo } from "../components/ForgeDemo";
import { Alert, ArrowRight, Cpu, Database, Layers, Lock, Shield, ShieldOff, Sparkle } from "../components/icons";
import { LogoMark } from "../components/Logo";
import { SiteHeader } from "../components/SiteHeader";
import { useInView } from "../lib/hooks";
import { RouteLink, type Route } from "../lib/route";

export function Landing({ navigate }: { navigate: (r: Route) => void }) {
  return (
    <div className="app-shell landing">
      <SiteHeader route="landing" navigate={navigate} />
      <main id="main">
        <Hero navigate={navigate} />
        <Pipeline />
        <ResultsSection />
        <DeploymentSection />
        <SafetySection />
        <FinalCta navigate={navigate} />
      </main>
      <Footer />
    </div>
  );
}

/* ---------------------------------------------------------------- hero */
function Hero({ navigate }: { navigate: (r: Route) => void }) {
  return (
    <section className="hero" aria-labelledby="hero-title">
      <div className="hero-bg" aria-hidden="true">
        <span className="glow g1" />
        <span className="glow g2" />
        <span className="grid-lines" />
      </div>
      <div className="container hero-grid">
        <div className="hero-copy">
          <p className="eyebrow">
            <Sparkle size={14} /> Fine-tuned · Quantized · Runs on your machine
          </p>
          <h1 id="hero-title">
            Plain English in.
            <br />
            <span className="molten">Guarded SQL</span> out.
          </h1>
          <p className="lede">
            SchemaForge is a specialized Text-to-SQL model: fine-tuned, evaluated, and quantized to run entirely on your own hardware, with deterministic safety standing between the
            model and your data.
          </p>
          <div className="hero-cta">
            <RouteLink to="workspace" navigate={navigate} className="btn btn-primary btn-lg">
              Open the workspace <ArrowRight size={18} />
            </RouteLink>
            <a href="#results" className="btn btn-secondary btn-lg">
              See the measured results
            </a>
          </div>
          <ul className="hero-facts" aria-label="Key facts">
            <li>
              <strong>Qwen3-4B</strong>
              <span>QLoRA fine-tuned</span>
            </li>
            <li>
              <strong>~2.39 GB</strong>
              <span>deployed (Q4_K_M + LoRA)</span>
            </li>
            <li>
              <strong>{DEPLOYMENT.cpuTokensPerSecond} tok/s</strong>
              <span>on CPU, no GPU needed</span>
            </li>
          </ul>
        </div>
        <ForgeDemo />
      </div>
    </section>
  );
}

/* ------------------------------------------------------------ pipeline */
const STEPS = [
  { icon: Database, title: "Schema in", body: "Your registered SQLite database is introspected into the same canonical prompt the model was trained on." },
  { icon: Cpu, title: "Model writes SQL", body: "Fine-tuned Qwen3-4B with a runtime LoRA, quantized to Q4_K_M, decoding greedily on CPU." },
  { icon: Shield, title: "Safety policy", body: "The SQL is parsed. One read-only statement passes; anything else is blocked before it touches data." },
  { icon: Lock, title: "Read-only run", body: "An independent read-only connection executes it with a time limit and a row cap." },
];

function Pipeline() {
  return (
    <section id="pipeline" className="section" aria-labelledby="pipeline-title">
      <div className="container">
        <SectionHead eyebrow="How it works" id="pipeline-title" title="One question, four gates." lede="The model only ever produces text. Everything that touches your data is ordinary, deterministic code." />
        <ol className="pipeline">
          {STEPS.map((s, i) => (
            <li key={s.title} className="pipe-step">
              <span className="pipe-n">{String(i + 1).padStart(2, "0")}</span>
              <span className="pipe-icon">
                <s.icon size={20} />
              </span>
              <h3>{s.title}</h3>
              <p>{s.body}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------- results */
function ResultsSection() {
  return (
    <section
      id="results"
      className="section section-alt"
      aria-labelledby="results-title"
    >
      <div className="container">
        <SectionHead
          eyebrow="Measured results"
          id="results-title"
          title="Measured, not marketed."
          lede="Two evaluations test generalization from different angles: held-out database schemas and an untouched external benchmark. Results are reported independently and never averaged into one score."
        />

        <div className="bench-grid">
          {BENCHMARKS.map((b) => (
            <BenchmarkCard key={b.id} b={b} />
          ))}
        </div>

        <p className="fine-print">
          <Alert size={15} />
          <span>
            Base = untouched Qwen3-4B-Instruct-2507. Fine-tuned = the same
            model with the checkpoint-1518 LoRA. Both results report execution
            accuracy. Compare base to fine-tuned within each evaluation.
          </span>
        </p>
      </div>
    </section>
  );
}

const pct = (n: number) => `${n.toFixed(2)}%`;

function BenchmarkCard({ b }: { b: Benchmark }) {
  const [ref, seen] = useInView<HTMLElement>(0.3);
  return (
    <article ref={ref} className={`bench tone-${b.tone} ${seen ? "in-view" : ""}`} aria-label={b.title}>
      <header>
        <span className="bench-eyebrow">{b.eyebrow}</span>
        <h3>{b.title}</h3>
        <p className="bench-metric">{b.metric}</p>
      </header>

      <div className="bench-delta" aria-label={`Improvement of ${b.deltaPp.toFixed(2)} percentage points`}>
        <span className="delta-num">+{b.deltaPp.toFixed(2)}</span>
        <span className="delta-unit">percentage points</span>
      </div>

      <div className="bench-bars">
        <div className="bench-row">
          <span className="bench-name">Base</span>
          <span className="bench-track">
            <span className="bench-fill is-base" style={{ ["--w" as string]: `${b.base}%` }} />
          </span>
          <span className="bench-val">{pct(b.base)}</span>
        </div>
        <div className="bench-row">
          <span className="bench-name">Fine-tuned</span>
          <span className="bench-track">
            <span className="bench-fill is-tuned" style={{ ["--w" as string]: `${b.tuned}%` }} />
          </span>
          <span className="bench-val is-tuned">{pct(b.tuned)}</span>
        </div>
      </div>

      <p className="bench-takeaway">{b.takeaway}</p>
      <footer>
        <span>n = {b.sampleSize}</span>
        <span>{b.scope}</span>
        <span className="bench-caveat">{b.caveat}</span>
      </footer>
    </article>
  );
}

/* ---------------------------------------------------------- deployment */
function DeploymentSection() {
  const [ref, seen] = useInView<HTMLDivElement>(0.25);
  const max = DEPLOYMENT.f16BaseGb;
  const rows = [
    { label: "F16 base", gb: DEPLOYMENT.f16BaseGb, kind: "f16" },
    { label: "Q4_K_M base", gb: DEPLOYMENT.q4BaseGb, kind: "q4" },
    { label: "Q4_K_M + LoRA (deployed)", gb: DEPLOYMENT.effectiveGb, kind: "eff" },
  ] as const;

  return (
    <section id="deployment" className="section" aria-labelledby="deploy-title">
      <div className="container">
        <SectionHead
          eyebrow="Deployment engineering"
          id="deploy-title"
          title="From 7.5 GB to a laptop-sized model."
          lede="The base model is quantized to Q4_K_M for llama.cpp. The fine-tune stays a small separate LoRA, loaded at runtime."
        />
        <div className={`deploy-grid ${seen ? "in-view" : ""}`} ref={ref}>
          <div className="deploy-sizes panel-lg" role="group" aria-label="Model size comparison">
            <h3>On-disk size</h3>
            {rows.map((r) => (
              <div className="size-row" key={r.label}>
                <span className="size-label">{r.label}</span>
                <span className="size-track">
                  <span className={`size-fill kind-${r.kind}`} style={{ ["--w" as string]: `${(r.gb / max) * 100}%` }} />
                </span>
                <span className="size-val">{r.gb.toFixed(2)} GB</span>
              </div>
            ))}
            <p className="muted small">
              The LoRA adapter alone is {(DEPLOYMENT.loraGb * 1000).toFixed(0)} MB. The quantized base is about {DEPLOYMENT.reductionPct}% smaller than F16.
            </p>
          </div>

          <ul className="stat-tiles">
            <li>
              <span className="stat-n">~{DEPLOYMENT.reductionPct}%</span>
              <span className="stat-l">smaller base model</span>
            </li>
            <li>
              <span className="stat-n">{DEPLOYMENT.cpuTokensPerSecond}</span>
              <span className="stat-l">tokens/sec generation, CPU-only</span>
            </li>
            <li>
              <span className="stat-n">~{DEPLOYMENT.effectiveGb.toFixed(2)} GB</span>
              <span className="stat-l">effective deployed model</span>
            </li>
            <li>
              <span className="stat-n">~{DEPLOYMENT.peakRssGb.toFixed(1)} GB</span>
              <span className="stat-l">peak process memory measured</span>
            </li>
          </ul>
        </div>

        <aside className="honesty" aria-label="Quantization caveat">
          <Layers size={18} />
          <p>
            <strong>Quantization is not free, and we don&apos;t pretend it is.</strong> On a fixed 10-prompt sanity check, Q4_K_M matched the F16 output exactly on only{" "}
            {DEPLOYMENT.sanityAgreement.same} of {DEPLOYMENT.sanityAgreement.of} prompts. Output equivalence is not claimed, and that sanity set is a regression check, not an accuracy benchmark.
          </p>
        </aside>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------- safety */
function SafetySection() {
  return (
    <section id="safety" className="section section-alt" aria-labelledby="safety-title">
      <div className="container safety-grid">
        <div>
          <SectionHead
            eyebrow="Deterministic safety"
            id="safety-title"
            title="The model proposes. Code decides."
            lede="A language model can be talked into anything, so the protection is not in the prompt. It sits after the model, in two independent layers."
            align="left"
          />
          <ul className="layer-list">
            <li>
              <span className="layer-n">1</span>
              <div>
                <h3>AST safety policy</h3>
                <p>Exactly one statement, and it must be a read-only SELECT or WITH. Mutations hidden inside CTEs or subqueries are found by walking the whole parse tree.</p>
              </div>
            </li>
            <li>
              <span className="layer-n">2</span>
              <div>
                <h3>Independent read-only executor</h3>
                <p>A read-only SQLite connection, an authorizer that denies writes, PRAGMA and ATTACH, a time limit, and a row cap. It holds even if layer 1 were bypassed.</p>
              </div>
            </li>
          </ul>
        </div>

        <figure className="blocked-demo" aria-label="Example of a blocked statement">
          <div className="blocked-head">
            <ShieldOff size={16} /> Blocked before execution
          </div>
          <pre>
            <code>
              <span className="tok tok-keyword">DROP TABLE</span> <span className="tok tok-identifier">employees</span>
              <span className="tok tok-punct">;</span>
            </code>
          </pre>
          <div className="blocked-reason">
            <span className="badge badge-bad">not_read_only_query</span>
            <span>Only SELECT / WITH queries are allowed.</span>
          </div>
          <figcaption>Illustrative example of the policy&apos;s structured decision.</figcaption>
        </figure>
      </div>
    </section>
  );
}

/* --------------------------------------------------------------- cta */
function FinalCta({ navigate }: { navigate: (r: Route) => void }) {
  return (
    <section className="section final-cta" aria-labelledby="cta-title">
      <div className="container cta-card">
        <h2 id="cta-title">See it run on your own machine.</h2>
        <p>Start the local backend, open the workspace, and ask your demo database a question.</p>
        <RouteLink to="workspace" navigate={navigate} className="btn btn-primary btn-lg">
          Open the workspace <ArrowRight size={18} />
        </RouteLink>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="site-footer">
      <div className="container footer-inner">
        <span className="footer-brand">
          <LogoMark size={22} /> SchemaForge
        </span>
        <p>Qwen3-4B-Instruct-2507 · QLoRA fine-tune · llama.cpp Q4_K_M + runtime LoRA · evaluated on BIRD</p>
      </div>
    </footer>
  );
}

function SectionHead({ eyebrow, id, title, lede, align = "center" }: { eyebrow: string; id: string; title: string; lede: string; align?: "center" | "left" }) {
  return (
    <div className={`section-head align-${align}`}>
      <p className="eyebrow">{eyebrow}</p>
      <h2 id={id}>{title}</h2>
      <p className="lede">{lede}</p>
    </div>
  );
}
