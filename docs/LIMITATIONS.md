# SchemaForge — Limitations

The authoritative, consolidated list. Each item is a property of the finished V1, stated
plainly; deeper context is in the linked phase documents.

## Model quality

- **The external generalization gain is modest.** BIRD Mini-Dev EX moved 43.6 -> 44.8%
  (+1.2 pp) on 500 examples from a single run. It is real but small and not a statistically
  established improvement ([`RESULTS.md`](RESULTS.md)).
- **Soft-F1 did not improve externally** (47.6975 -> 47.1816). Parse and execution success
  improved (0.942 -> 0.984, 0.82 -> 0.87), which reflects well-formedness, not correctness.
- The strongest internal evidence (schema-held-out, +5.43 pp EX) uses SchemaForge's own SQLite
  comparator rather than the official evaluator, on one split.
- The seen-training gain (+33.2 pp exact match) is specialization, not generalization.
- **The deployed quantized model was not re-scored.** Evaluation used NF4 + LoRA; deployment
  uses Q4_K_M + hot LoRA, and a 10-example sanity check showed only 4/10 identical outputs.
- No state-of-the-art claim is made; there is no comparison to other systems.

## Correctness and trust

- **Semantic correctness is not guaranteed.** Deterministic safety, preflight and read-only
  execution show SQL is safe, valid for the schema and read-only — not that it answers the
  question. `semantic_correctness` is always `not_verified` ([`PHASE10.md`](PHASE10.md)).
- **Confidence is intentionally unclaimed** (`confidence = null`); no calibrated signal exists.

## Serving and deployment

- **CPU inference has material latency.** The proof host (2 vCPU) generated ~3.4 tokens/s.
- **The cloud timing is one observation, not a benchmark:** a single request took 16.0 s. No
  repeats, percentiles or load tests exist for the cloud host.
- **The proof host used one model slot** (`LLAMA_PARALLEL=1`): concurrent requests queue
  briefly (bounded) and are then rejected with `model_busy`.
- **No high availability or autoscaling:** one host, one backend worker, an in-process rate
  limiter and gate.
- **AWS compute must be recreated when the demonstration backend is not running.** The proof
  instance was terminated after evidence capture; the static Vercel frontend cannot answer
  queries without a live backend ([`PHASE11.md`](PHASE11.md#teardown-and-recreation-strategy)).
- **The GPU serving path exists but was not exercised** (`docker-compose.gpu.yml`,
  `LLAMA_NGL`).
- No response streaming to the UI; prompt caching is off for determinism.

## Product scope

- **SQLite-first V1.** One registered, read-only demo database.
- **No PostgreSQL / bring-your-own database** in V1.
- **No authentication or multi-tenant deployment;** the public demo is protected only by rate
  limiting, body limits and network hardening.
- No agents, RAG or MCP layer — by design.

## Reproducibility

- GPU training is not bit-for-bit reproducible; the frozen adapter and GGUFs are identified by
  SHA-256 ([`TRAINING.md`](TRAINING.md#reproducibility-boundaries)).
- Model weights, GGUF files, datasets and raw run archives are not in Git.

## What I would build next

Broader/repeated evaluation of the quantized deployment artifact (execution accuracy for
Q4_K_M + LoRA), multiple seeds/splits with confidence intervals, PostgreSQL support behind the
existing dialect seam, authentication, a properly benchmarked serving setup (repeats,
percentiles, GPU), and response streaming.
