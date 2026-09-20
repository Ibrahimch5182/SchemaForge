# SchemaForge — Final Certification Matrix (Phase 12)

Evidence-based status of every phase. "Evidence" points to documents, tests, commits and tags in
this repository. Where evidence is a local, uncommitted archive (Kaggle outputs, raw AWS
captures) that is stated explicitly. Results are in [`RESULTS.md`](RESULTS.md).

**Automated checks at Phase 12 (2026-09-20):** backend `uv run pytest -q` — 675 passed, 1
skipped; frontend typecheck, lint and production build clean; frontend tests 114/115 passing
(the one exception is the landing-page benchmark-display assertion, an accepted presentation
change that is not scientific evidence).

| Phase | Capability | Evidence | Status |
|---|---|---|---|
| 1 | Data foundation: reproducible BIRD pipeline, canonical prompt, DB-level split with zero leakage, explicit validation | [`DATA_CONTRACT.md`](DATA_CONTRACT.md) · `tests/data/` · commit `6e55709` · tag `phase-1-pass` | Passed |
| 2 | Evaluation engine: locked Mini-Dev (500 SELECT-only), gold isolation, vendored unmodified official evaluator, oracle sanity | [`EVALUATION.md`](EVALUATION.md) · `tests/benchmark/` · commit `5026885` · tag `phase-2-pass` | Passed |
| 3 | Untuned baseline: Qwen3-4B NF4 on Mini-Dev, EX 43.6 / Soft-F1 47.6975, 500/500 generated | [`BASELINE.md`](BASELINE.md) · `tests/model/` · `kaggle-phase3-export/` · commit `1a9e9a8` · tag `phase-3-pass` | Passed |
| 4 | QLoRA smoke: 20/20 steps, adapter saved and reload-verified, completion-only masking validated on all 6,067 examples | [`TRAINING.md`](TRAINING.md) · `tests/train/` · `kaggle-phase4-evidence/` · commit `b367767` · tag `phase-4-pass` | Passed |
| 5 | Full specialization: adaptive schema compaction (zero examples over 4096), certification/resume tooling, canonical 2-epoch run to `checkpoint-1518` (adapter SHA-256 `f7b78b3c…34cc`) | [`TRAINING.md`](TRAINING.md) · [`CONTEXT_BUDGET.md`](CONTEXT_BUDGET.md) · [`PROJECT.md`](../PROJECT.md) (Phase 5A–5D) · `tests/train/`, `tests/schema_context/` · commits `a834085`…`e173c23` · raw Kaggle outputs are local, uncommitted archives · *no `phase-5-pass` tag exists* | Completed (untagged) |
| 6 | Held-out and external evaluation: seen-training, schema-held-out (534), Mini-Dev (500) for base vs `checkpoint-1518` | [`RESULTS.md`](RESULTS.md) · runner `scripts/run_finetuned.py` (commit `36b2c24`, `tests/model/test_run_finetuned_contracts.py`) · raw evaluation outputs are local, uncommitted archives · *no `phase-6-pass` tag exists* | Completed (untagged); numbers frozen |
| 7 | Quantization and local inference: F16 + Q4_K_M GGUF, hot LoRA, hashes, local benchmark, quantization sanity check (equivalence **not** claimed) | [`PHASE7.md`](PHASE7.md) · `tests/deploy/` · commit `17b2e11` · tag `phase-7-pass` | Passed |
| 8 | Production backend: registry, introspection, AST safety, independent read-only executor, `QueryService`, FastAPI + CLI | [`PHASE8.md`](PHASE8.md) · `tests/backend/test_safety.py`, `test_executor.py`, `test_service.py`, `test_api.py` · commit `aa67e7d` · tag `phase-8-pass` | Passed |
| 9 | Product frontend: landing, workspace, API client, strict response guards, no model/DB/SQL logic in the browser | [`PHASE9.md`](PHASE9.md) · `frontend/src/**/*.test.ts(x)` · commit `bd3ffc4` · tag `phase-9-pass` | Passed |
| 10 | Reliability and calibration: preflight, bounded concurrency, cancellation, error taxonomy, `/ready`, `semantic_correctness = not_verified`, `confidence = null` | [`PHASE10.md`](PHASE10.md) · `tests/backend/test_reliability.py`, `test_api_reliability.py` · `scripts/canary_phase10.py` · commit `a5d1452` · tag `phase-10-pass` | Passed |
| 11 | Persistent serving: private persistent llama.cpp server, Docker/Compose, prompt parity, hardening, local serving canary | [`PHASE11.md`](PHASE11.md) · `tests/backend/test_phase11_serving.py`, `test_phase11_public.py` · `scripts/canary_phase11.py` · commit `d51d36a` · tag `phase-11-pass` | Passed |
| 11 | AWS EC2 + Vercel deployment: HTTPS, exact-origin CORS, artifact SHA-256 verification, `verify_stack.sh` all PASS | [`evidence/phase11-production-proof.md`](evidence/phase11-production-proof.md) · commits `d51d36a`, `2baee97` · raw captures are local, git-ignored (`.artifacts/phase11/`) · some AWS console settings are operator-reported | Passed |
| 11 | External mobile/LTE proof: iPhone on LTE with Wi-Fi off, real question, safe SQL, result `625000` | [`evidence/phase11-production-proof.md`](evidence/phase11-production-proof.md#7-external-mobile-proof) · [`assets/phase11/`](assets/phase11/) · timing (16.0 s, 3.4 tok/s) is one observation, not a benchmark | Passed |
| 11 | Teardown and reproducibility: proof instance terminated; temporary hostname retired; runbook and recreation strategy checked in | [`PHASE11.md`](PHASE11.md#teardown-and-recreation-strategy) · [`QUICKSTART.md`](QUICKSTART.md) · demo backend is recreated on demand | Passed |
| 12 | Final release certification: README, results, architecture, limitations, quickstart, portfolio, documentation audit, final test run | this document · [`RESULTS.md`](RESULTS.md) · [`LIMITATIONS.md`](LIMITATIONS.md) · [`PORTFOLIO.md`](PORTFOLIO.md) · tag `phase-12-pass` / `v1.0.0` once created by the maintainer | Prepared — pending maintainer commit and tags |

## Certified scope and explicit non-claims

Certified: a reproducible pipeline from data to a deployed, safety-checked product; frozen
scientific results with honest framing; a real public deployment proven from an independent
device.

Not claimed: state-of-the-art accuracy; accuracy of the quantized Q4_K_M model (never
re-scored); a cloud latency benchmark; semantic correctness of generated SQL; high
availability, multi-tenancy, authentication or PostgreSQL support ([`LIMITATIONS.md`](LIMITATIONS.md)).
