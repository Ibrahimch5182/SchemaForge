# SchemaForge — Reproducibility and Quickstart

Six levels, from "no model needed" to a full cloud deployment. Placeholders like
`<your-domain>` are yours to fill in; **no secrets are needed by any step, and none belong in
Git.** GGUF/model files are never in this repository.

Requires Python 3.11 with [`uv`](https://docs.astral.sh/uv/) and (for the frontend) Node.js + npm.

## A. Install and test — no models required

```bash
uv sync
uv run pytest -q                 # backend + pipeline suite; offline, fixture-based
cd frontend && npm ci
npm run typecheck && npm run lint && npm test
VITE_API_BASE_URL=https://api.example.com npm run build   # production builds require this variable
```

Optional groups: `uv sync --group eval` (official BIRD evaluator), `uv sync --group model
--group train` (CUDA machine only; training/baseline). Heavy libraries are lazily imported, so
the default install and test suite need no GPU.

## B. Provide the model artifacts

The deployment needs two files, produced by the Phase 7 pipeline
([`PHASE7.md`](PHASE7.md)) from the frozen base model and adapter:

| File | SHA-256 |
|---|---|
| `base-Q4_K_M.gguf` | `3df3d5bfa7290f20e8b0ad2b9bae78aa06fb198b97837e4ebc28b5867851c848` |
| `lora-1518-f16.gguf` | `53ee2c6dd036ebcccdf0c71bf682c961244ca4665e0cc53bd809ac98b944ba48` |

Generate a checksum manifest next to them:
`uv run python scripts/phase11_model_manifest.py --base base-Q4_K_M.gguf --lora lora-1518-f16.gguf --out-dir ./model-bundle`.
Rebuilding them from scratch requires the adapter (`checkpoint-1518`, SHA-256 in
[`TRAINING.md`](TRAINING.md)) and a llama.cpp build; the adapter and weights are not in Git.

## C. Run locally, production-style

Backend with the development subprocess runtime (paths are yours; see [`PHASE8.md`](PHASE8.md)):

```bash
export SCHEMAFORGE_LLAMA_EXE=<path/to/llama-completion> SCHEMAFORGE_BASE_GGUF=<base-Q4_K_M.gguf> SCHEMAFORGE_LORA_GGUF=<lora-1518-f16.gguf>
uv run uvicorn localsql.backend.api:create_app_from_env --factory --port 8000
cd frontend && npm run dev            # http://localhost:5173
```

Without model variables the backend still starts and reports `model_not_configured`.

## D. Docker Compose (persistent llama.cpp server)

```bash
export MODEL_DIR=<dir with the two GGUFs + SHA256SUMS> BASE_GGUF_NAME=base-Q4_K_M.gguf LORA_GGUF_NAME=lora-1518-f16.gguf LLAMA_THREADS=<physical cores>
export SCHEMAFORGE_DOMAIN=unused SCHEMAFORGE_CORS_ORIGINS=http://localhost:5173
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build seed-db model-server backend
curl localhost:8000/ready
```

Set `LLAMA_THREADS` to the host's physical cores/vCPUs (leaving llama.cpp's default measured
~30x slower on a shared/SMT host). Optional GPU overlay: `docker-compose.gpu.yml`. Details and
security controls: [`PHASE11.md`](PHASE11.md#5-docker).

## E. Recreate the AWS deployment

Follow the runbook in [`PHASE11.md`](PHASE11.md#7-deployment-runbook-as-actually-performed):
an x86_64 host with >= 8 GiB RAM and 2+ vCPU, security group 80/443 public and 22 restricted
to your IP, `deploy/aws/bootstrap_ec2.sh`, copy the two GGUFs, `cp deploy/production.env.example
.env` (set `SCHEMAFORGE_DOMAIN`, `SCHEMAFORGE_CORS_ORIGINS`, `LLAMA_THREADS`), `docker compose up -d
--build`, then `bash deploy/aws/verify_stack.sh https://<your-domain>`. Nothing in this
repository creates AWS resources or reads AWS credentials. The original proof hostname is
retired; a new public IP means a new hostname and a frontend redeploy. Tear down as described
in the runbook to avoid ongoing charges.

## F. Deploy or update Vercel

From `frontend/`, set project environment variables `VITE_API_BASE_URL=https://<your-domain>`
and `VITE_QUERY_TIMEOUT_MS=300000` (baked in at build time), then `vercel --prod`. Use the stable
production URL and set it, as one exact `https://` origin, in `SCHEMAFORGE_CORS_ORIGINS` on the
backend (`docker compose up -d --force-recreate backend`). Never commit `.vercel/` or `.env*`.
