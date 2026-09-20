# SchemaForge — Portfolio Material

Everything here is drawn from this repository's documented results. It contains no users,
revenue, business impact or scale claims, because there are none: this is an engineering and
research project. Authoritative numbers: [`RESULTS.md`](RESULTS.md).

## A. One sentence

SchemaForge fine-tunes an open-weight 4B model with QLoRA for read-only Text-to-SQL, evaluates
it without data leakage, quantizes it, and serves it through a deterministic safety layer as a
web application deployed on AWS and Vercel.

## B. Resume / project description

- Fine-tuned Qwen3-4B (4-bit QLoRA, completion-only loss) on BIRD with database-level splits;
  normalized SQL exact match 4.2 -> 37.4% on seen data, execution accuracy 39.1 -> 44.6% on
  held-out schemas, and BIRD Mini-Dev EX 43.6 -> 44.8% (+1.2 pp, reported as modest).
- Built a leakage-safe evaluation system around the official BIRD evaluator and a
  deterministic data/prompt pipeline shared by training, evaluation and serving.
- Quantized to GGUF Q4_K_M (~2.4 GB with LoRA) and served through a persistent llama.cpp server
  behind a FastAPI backend with deterministic AST safety, schema-aware preflight and a
  read-only SQLite executor.
- Deployed on AWS EC2 (Docker Compose, Caddy HTTPS) with a Vercel frontend and verified it from
  a phone on LTE.

## C. Technical portfolio description

SchemaForge asks a narrow research question: can a compact open-weight model, specialized with
QLoRA, generate correct SQL for database schemas it has never seen? Text-to-SQL was chosen as
the vehicle because it has an executable ground truth, so every claim about fine-tuning can be
checked against results rather than impressions. The data pipeline builds a single canonical
prompt/completion format from BIRD, splits strictly by database, and reuses the exact same
prompt builder for training, evaluation and serving. Very large schemas are handled by an
adaptive per-database compaction policy so that no training example is truncated at 4,096
tokens.

The model, Qwen3-4B-Instruct-2507, was trained with 4-bit NF4 QLoRA (rank 16 across all
attention and MLP projections) and an explicit, unit-tested completion-only loss, in resumable
sessions on Kaggle GPUs. Evaluation is reported as three separate results — seen-data
specialization, schema-held-out generalization and the untouched BIRD Mini-Dev benchmark — with
the official evaluator vendored unmodified. The held-out and external gains are real but
modest (+5.4 pp and +1.2 pp EX), Soft-F1 did not improve externally, and the write-up says so.

For deployment, the adapter and base were converted to GGUF and the base quantized to Q4_K_M,
then served by a persistent, private llama.cpp server that loads the model once. A FastAPI
backend treats the model as an untrusted text generator: SQL is parsed and checked against an
AST allow-list, compiled against the real schema in a preflight, and executed on a read-only,
authorizer-guarded SQLite connection with timeouts and row caps. The API is explicit that safe
is not correct — semantic correctness is reported as not verified and no confidence score is
invented. The stack runs in hardened containers behind Caddy on a single CPU-only AWS EC2 host,
with a Vercel frontend, and was validated end to end from a phone on LTE. The AWS host is
recreated for demonstrations rather than left running.

## D. Key technologies

Python 3.11, PyTorch / Transformers / PEFT / bitsandbytes (QLoRA), Qwen3-4B-Instruct-2507, BIRD
and its official evaluator, sqlglot, llama.cpp / GGUF (Q4_K_M, LoRA), FastAPI, SQLite, Docker
Compose, Caddy, AWS EC2, Vercel, React + TypeScript + Vite, Vitest, pytest, uv.

## E. Strongest engineering achievements

- End-to-end ownership: data -> training -> evaluation -> quantization -> serving -> safety ->
  frontend -> cloud, in one reproducible repository.
- Leakage discipline: DB-level splits with zero overlap asserted in code; evaluation-only
  benchmark with gold isolated from generation; official evaluator used unmodified.
- Deterministic trust boundary around a probabilistic model, with an independent read-only
  executor that holds even if the AST layer is bypassed (tested).
- Production reliability work: bounded concurrency, cancellation, timeouts, readiness,
  structured errors, rate/body limits, hardened containers, hash-verified artifacts.
- Real deployment with independent (mobile/LTE) validation, then deliberate teardown and a
  written recreation runbook.
- Measured, honest reporting: modest and negative results are stated; observations are not
  presented as benchmarks.

## F. Scientifically honest results summary

| Evaluation | Base | Fine-tuned | Δ |
|---|---|---|---|
| Seen-training normalized exact match (n=500) | 4.20% | 37.40% | +33.20 pp — specialization, not generalization |
| Schema-held-out execution accuracy (n=534, 7 unseen DBs; own comparator) | 39.14% | 44.57% | +5.43 pp |
| BIRD Mini-Dev EX (n=500, untouched, official) | 43.6% | 44.8% | +1.2 pp — modest |

Mini-Dev Soft-F1 went 47.70 -> 47.18 (no improvement); parse success 0.942 -> 0.984; execution
success 0.82 -> 0.87. No state-of-the-art claim. The quantized deployment model was not
re-scored. One cloud request (16.0 s, 3.4 tok/s on 2 vCPU) is an observation, not a benchmark.

## G. Suggested interview talking points

- **Why Text-to-SQL?** Executable ground truth makes model-engineering claims checkable, and
  it exercises schema grounding, structured output and safety in one task.
- **Why QLoRA?** It makes fine-tuning a 4B model feasible on a single Kaggle T4 by training a
  small adapter over a 4-bit base. The cost: the adapter was trained on an NF4 base but is
  deployed on a differently quantized (Q4_K_M) base.
- **Why schema-held-out splits?** Splitting by example lets the model memorize schemas.
  Splitting by database measures transfer to unseen schemas — which is why the +33 pp
  seen-data number is reported as specialization and the +5.4 pp held-out number carries the
  generalization claim.
- **Why quantization?** F16 is 7.5 GB; Q4_K_M is 2.3 GB, which fits a small CPU host. The
  trade-off: outputs changed on a sanity sample (4/10 identical), so accuracy equivalence is
  not claimed and re-scoring the deployed artifact is the first next step.
- **Why llama.cpp?** It serves the exact frozen GGUF artifacts, runs on CPU, supports hot
  LoRA, and keeps GPU offload optional; a GPU-centric server such as vLLM was out of scope.
- **What changed after fine-tuning?** Modest EX gains, but the clearest change is reliability:
  more parseable and executable SQL. Soft-F1 did not improve externally.
- **Why is safety outside the model?** A prompt is not a security control. The model's output
  is untrusted, so parsing, allow-listing, preflight and read-only execution are deterministic
  and independently testable.
- **How was deployment productionized?** Persistent server instead of per-query loads, private
  network, hardened containers, exact-origin CORS, readiness, back-pressure, cancellation,
  hash-verified artifacts, and a real external validation.
- **Limitations and next steps:** modest external gain, CPU latency, single slot, SQLite-only,
  no auth. Next: re-score the quantized model, multiple seeds/splits with intervals, PostgreSQL
  behind the existing dialect seam, authentication, and a proper serving benchmark.
