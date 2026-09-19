# Phase 11 — Persistent serving, containerization, cloud-deployment readiness

Phases 1–10 are frozen. Phase 11 changes **how the model is served and shipped**, not what it
does: same frozen Q4_K_M base + hot LoRA (checkpoint-1518), same canonical prompt, same
normalization, safety, preflight and read-only execution.

```
before   query → launch llama.cpp → load 2.4 GB model → generate → exit      (~10 s reload per query)
now      llama-server started once → Q4_K_M + LoRA resident → many requests  (HTTP, model stays loaded)

Vercel (static SPA) ─HTTPS→ Caddy ─→ FastAPI backend ─HTTP (private net)→ llama-server
                                     │                                      └ /models (ro): Q4_K_M + LoRA GGUF
                                     └ registered demo SQLite DB (ro volume + read-only executor)
```

**Status:** implementation gate only. The overall Phase 11 gate is the manual, off-laptop cloud
proof in [§10](#10-manual-cloud-proof-operator-sequence).

## 1. Persistent llama.cpp runtime

`localsql/backend/server_runtime.py::LlamaServerRuntime` implements the same `ModelRuntime`
protocol as the Phase 8 subprocess runtime, so `QueryService` is unchanged and runtime-agnostic.

* `POST /completion` on `llama-server`, **streamed** (SSE). Greedy: `temperature 0, top_k 1`,
  `seed 42`, `n_predict 512`, `cache_prompt false` (deterministic cold prefill by default; enable
  in `configs/backend.yaml → runtime.server.cache_prompt` for speed at the cost of bit-identity).
  Context/tokens/seed come from `configs/phase7.yaml`, exactly as for the subprocess runtime.
* `raw_completion` is returned untouched; `normalize_predicted_sql` stays in `QueryService`.
* Metrics recorded from llama.cpp's own final chunk: prompt/generated tokens, tokens/sec,
  server prompt/generation ms, `stop_type` — `null`/absent when the server does not report them.
* Failures map onto the existing taxonomy (no new codes): unreachable/malformed/HTTP error →
  `model_error`; deadline → `model_timeout`; full gate → `model_busy`; cancelled → `cancelled`.
  Internal reasons (`server_unreachable`, `malformed_response`, `incomplete_stream`, `http_status`…)
  go to the log only; API messages stay generic and never contain the server URL or paths.

### Runtime selection

| Setting | Meaning |
|---|---|
| `SCHEMAFORGE_RUNTIME_KIND` (or `runtime.kind`) | `llama_server` (persistent, production) · `llama_cpp` (subprocess; **development/fallback, still default**) |
| `SCHEMAFORGE_LLAMA_SERVER_URL` | private URL, e.g. `http://model-server:8080`. Never committed. Missing → `model_not_configured` (service still starts) |
| `SCHEMAFORGE_LLAMA_PARALLEL` | server slots = backend inference concurrency (default 1) |
| `SCHEMAFORGE_MODEL_TIMEOUT_SECONDS` | per-request generation limit (default `runtime.timeout_seconds` = 120; compose uses 240) |
| `SCHEMAFORGE_LLAMA_NGL` | informational GPU-layer count for `/health`; the real offload flag is on the server |

The subprocess `LlamaCppRuntime` and its Phase 10 controls (gate, kill-on-cancel, Phase 8
trailing-newline guard) are untouched.

## 2. Prompt / output parity

Both runtimes wrap the canonical prompt in the same ChatML envelope
(`build_chatml_prompt`) ending `assistant\n`. `llama-completion -f` strips one trailing newline
(the Phase 8 drift), so the subprocess runtime writes one extra; the server receives the string
verbatim. `tests/backend/test_phase11_serving.py` pins:
`subprocess prompt-file text == server prompt + "\n"`, that the server prompt ends
`<|im_start|>assistant\n`, that both runtimes receive the identical canonical prompt from
`QueryService`, and that generation settings equal the Phase 7 config. The local canary shows the
same SQL/answers as Phase 10's three fixture queries.

## 3. Concurrency, cancellation, readiness

* **Bounded concurrency.** `InferenceGate(slots, max_waiting=2, wait=20 s)` — concurrency equals
  the server's `--parallel` slots (not hard-wired to 1 because there is no per-request model
  load any more), the wait queue stays bounded, overflow is rejected immediately (`model_busy`,
  HTTP 429 + `Retry-After`). The entrypoint sets `-c CTX×PARALLEL` so every slot keeps the full
  8192 context. Keep `PARALLEL=1` on CPU: extra slots multiply KV memory without adding throughput.
* **Cancellation / timeout.** The stream is read on a worker thread and polled every 100 ms;
  on `POST /query/{id}/cancel` or deadline the HTTP response/connection is closed, which makes
  llama-server abort generation and free the slot. Real-server canary: cancelled in ~80 ms, `/slots`
  idle afterwards, next query succeeds on the same process.
* **Readiness never runs inference.** `GET /health` on the server (200 ready / 503 loading),
  cached 1 s. `/ready` and `/health` add a `serving` block:
  `state ∈ ready | busy | saturated | loading | unreachable`, and
  `checks{server_configured, server_reachable, model_ready}`. `/ready` is 503 with reasons
  `model_server_loading` / `model_server_unreachable`; `/health` stays 200 (backend alive) even
  when the model server is down. No URL, path or prompt is ever exposed. The canary verifies
  server token counters are still 0 after readiness/health polling.
* One uvicorn worker on purpose: the gate and rate limiter are per-process.

## 4. Observability

JSON-line events (existing logger, existing forbidden-field filter — no prompts, rows, paths, URLs):
`runtime.generated` (mode, `queue_wait_ms`, `generation_latency_ms`, tokens, tokens/sec, stop type),
`runtime.timeout`, `runtime.cancelled`, `runtime.saturated` (gate snapshot), `runtime.failure`
(reason), `runtime.server_state` (only on state transitions), `api.rate_limited`, plus the existing
`query.completed` (total backend latency + per-stage timings). The same numbers are in the
response's `model` block. The model server exposes Prometheus counters on its private `/metrics`
(`--metrics`); nothing heavier is added.

## 5. Docker

| File | Purpose |
|---|---|
| `docker/backend.Dockerfile` | python 3.11-slim, `uv sync --frozen --no-dev`, non-root uid 10001, healthcheck, 1 uvicorn worker. No weights inside. |
| `docker/model-server.Dockerfile` + `model-server-entrypoint.sh` | pinned `ghcr.io/ggml-org/llama.cpp:server-b10964` (the Phase 7 build), non-root, env-driven launch, optional `SHA256SUMS` verification (refuses to start on mismatch), `--cache-ram 0`, `--metrics`, healthcheck (`/health`). |
| `docker-compose.yml` | production stack: `seed-db` (one-shot demo DB), `model-server`, `backend`, `caddy`. Only Caddy publishes ports (80/443). |
| `docker-compose.local.yml` | local validation: publishes the backend on loopback, development CORS, optional `frontend` profile (nginx). |
| `docker-compose.gpu.yml` | **optional** CUDA overlay (`server-cuda-b10964`, `LLAMA_NGL=99`, NVIDIA device reservation). |
| `deploy/Caddyfile` | automatic HTTPS, 64 KB body cap, HSTS, `X-Forwarded-For` set by Caddy. |
| `deploy/production.env.example` | every knob, no secrets. (Not named `.env.*` — the repo ignores those.) |
| `.dockerignore` | keeps `.git`, `.artifacts`, `*.gguf`, `data/`, notebooks, zips, `node_modules` out of every build context. |

Hardening: no `privileged`; `cap_drop: ALL`; `no-new-privileges`; backend `read_only: true`
rootfs and the SQLite volume mounted `:ro`; models mounted `:ro`; the `model` network is
`internal: true` (no egress, no host route) and only backend + model-server join it; the model
server has no `ports:`. Verified locally: the backend reaches `model-server:8080`; the model server
has no outbound route; nothing on the host answers on the model-server port; writes to `/app` and
`/data/databases` fail. Compose refuses to start without `SCHEMAFORGE_CORS_ORIGINS`,
`SCHEMAFORGE_DOMAIN`, `MODEL_DIR`, `BASE_GGUF_NAME`, `LORA_GGUF_NAME`.

Local run:
```bash
export MODEL_DIR=/path/to/dir BASE_GGUF_NAME=base-Q4_K_M.gguf LORA_GGUF_NAME=lora-1518-f16.gguf LLAMA_THREADS=6
export SCHEMAFORGE_DOMAIN=unused SCHEMAFORGE_CORS_ORIGINS=http://localhost:5173
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build seed-db model-server backend
curl localhost:8000/ready
```

### CPU thread count matters (measured)

`-t -1` uses every logical CPU. On a shared/SMT machine (Docker Desktop, 12 logical CPUs) that
gave **0.4 tokens/s**; `LLAMA_THREADS=6` gave **~12 tokens/s** for the same model. Always set
`LLAMA_THREADS` to the physical-core / vCPU count of the host and check `runtime.generated` logs.
On a 2-vCPU EC2 instance use `LLAMA_THREADS=2`.

## 6. Public networking (Vercel → HTTPS backend → private model)

* **Frontend:** `VITE_API_BASE_URL` (Vercel env var). A *production* build now **fails** if it is
  unset instead of baking in `127.0.0.1` (`frontend/vite.config.ts`); local dev is unchanged.
  `frontend/vercel.json` adds the SPA rewrite and basic headers; `frontend/env.production.example`.
* **CORS:** `SCHEMAFORGE_ENV=production` makes the backend **fail at startup** unless
  `SCHEMAFORGE_CORS_ORIGINS` is an explicit list of `https://` origins (no `*`, no loopback, no
  path). `/docs`, `/redoc`, `/openapi.json` are disabled in production. Vercel *preview* URLs are
  not allowed unless you list them; use the production URL or a custom domain.
* **Request limits:** Caddy caps bodies at 64 KB; the backend refuses a declared
  `Content-Length` > `SCHEMAFORGE_MAX_BODY_BYTES` (413 `payload_too_large`).
* **Rate limit:** in-process sliding window per client on `POST /query` only
  (`SCHEMAFORGE_RATE_LIMIT_PER_MINUTE`, compose default 20; 429 `rate_limited` + `Retry-After`; the
  frontend shows a friendly "too many requests"). Memory-bounded. Not distributed, not auth.
* **Proxy headers:** the client identity is `X-Forwarded-For[-hops]` with
  `SCHEMAFORGE_TRUSTED_PROXY_HOPS=1` (Caddy overwrites client-supplied values); with 0 hops the
  header is ignored entirely so it cannot be spoofed. `X-Request-ID` is preserved end to end.
* The model server is never internet-facing; the backend is the only application boundary.

## 7. AWS zero-cost path (single EC2 host, CPU)

Nothing here creates AWS resources or uses credentials; you run these steps by hand, deploy →
prove → save evidence → terminate.

**Sizing (measured).** The resident model server used ~2.9 GiB (Q4_K_M + LoRA, 8192 context, 1
slot) plus backend/Caddy. Use **≥ 8 GiB RAM, 2+ vCPU, 30 GB gp3**. 1–2 GiB "micro/small" free-tier
types **cannot** hold the model; 4 GiB is marginal. Check in the console which instance types
your Free Tier / credit plan actually covers and pick an 8 GiB one; if none is covered, stop —
do not pay. (Instance-type eligibility changes; this repo cannot verify it.) CPU speed on 2 vCPU
will be a few tokens/s; the compose default `MODEL_TIMEOUT_SECONDS=240` and
`VITE_QUERY_TIMEOUT_MS=180000` are sized for that — raise both if you see `model_timeout`.

**Security group:** inbound TCP 80 and 443 from anywhere; TCP 22 **only from your IP** (or use SSM
Session Manager and open no SSH at all). Nothing else — 8000/8080 must stay closed.

**HTTPS / domain options (all free):** (a) `<ip-with-dashes>.sslip.io` (e.g.
`203-0-113-10.sslip.io`) — a real DNS name pointing at your Elastic IP/public IP, Let's Encrypt
works, no registration; (b) a DuckDNS subdomain; (c) your own domain with an A record. Use the same
value as `SCHEMAFORGE_DOMAIN`. A public IPv4 that changes on stop/start breaks (a): re-derive the name.

1. Launch the instance (Ubuntu 24.04 or Amazon Linux 2023), attach the security group above.
2. `scp deploy/aws/bootstrap_ec2.sh` and run it (installs Docker + Compose plugin, 4 GB swap as
   OOM safety net, creates `/opt/schemaforge/models`). Log out/in for the docker group.
3. Get the code on the host: `git clone` (or `scp -r`) — no models are in Git.
4. **Model artifacts** (~2.56 GB; the simplest secure method is `scp`, no cloud storage needed).
   On the laptop:
   ```bash
   uv run python scripts/phase11_model_manifest.py --base base-Q4_K_M.gguf --lora lora-1518-f16.gguf --out-dir ./model-bundle
   scp -i key.pem base-Q4_K_M.gguf lora-1518-f16.gguf model-bundle/SHA256SUMS ubuntu@HOST:/opt/schemaforge/models/
   ```
   The manifest hashes the exact files that were canary-tested here
   (`base-Q4_K_M.gguf` `3df3d5bf…c848`, `lora-1518-f16.gguf` `53ee2c6d…ba48` on this machine —
   compare with your own manifest output). The model-server container re-verifies `SHA256SUMS` at
   start and refuses to run on a mismatch. (S3 works too — `aws s3 cp` into the same directory —
   but is unnecessary for a short proof and may incur request/storage charges.)
5. `cp deploy/production.env.example .env`; set `SCHEMAFORGE_DOMAIN`, `SCHEMAFORGE_CORS_ORIGINS`
   (your Vercel production URL), `LLAMA_THREADS` (= vCPU count), keep `LLAMA_NGL=0`.
6. `docker compose up -d --build`, then watch `docker compose logs -f model-server`. Caddy starts
   once the backend is healthy; the model takes ~1–3 min to load on CPU (`/ready` → `loading`, then
   200).
7. `bash deploy/aws/verify_stack.sh https://$SCHEMAFORGE_DOMAIN` — checks HTTPS health, readiness,
   demo DB, hidden `/docs`, and that ports 8000/8080 are closed from outside.
8. **Vercel:** import `frontend/` as the project root, set `VITE_API_BASE_URL=https://$SCHEMAFORGE_DOMAIN`
   (and optionally `VITE_QUERY_TIMEOUT_MS`), deploy the production branch. Add that exact Vercel URL
   to `SCHEMAFORGE_CORS_ORIGINS` and `docker compose up -d` again if you set it afterwards.
9. **Restart/stop:** `docker compose restart` / `docker compose down` (keep volumes; the model reload
   is automatic). The stack has `restart: unless-stopped`, so it returns after a reboot.
10. **Shutdown / no-leftovers:** `docker compose down -v`; then in AWS **terminate** the instance,
    release any Elastic IP, delete the EBS volume if it survived (`DeleteOnTermination` is on by
    default), delete the security group and key pair if unused. Verify in the EC2 console
    (Instances, Volumes, Elastic IPs, Snapshots) in **every region you touched** — and later in
    Billing → Free Tier/Cost Explorer — that nothing remains. Remove the Vercel project or
    at least the `VITE_API_BASE_URL` so it does not point at a dead host.

## 8. Serving canary (`scripts/canary_phase11.py`)

Operations check, **not a model benchmark** (synthetic 3-table fixture, ~6 real generations, no
BIRD). It starts the model server *as production does* (the `schemaforge-model-server` image with
the real entrypoint and hash verification, models mounted read-only, loopback port), then verifies:
ready without inference (server token counters still 0) · health reports `persistent_server` ·
three sequential full-pipeline queries (Phase 10's fixture questions) all succeed and are served
by the persistent server · counters accumulate in one resident process · model loaded **once**
(log) and the container never restarted · saturation returns `model_busy` immediately ·
cancellation aborts server-side and the slot is freed (`/slots`) · no leaked gate slot · a
follow-up query succeeds on the same server · readiness healthy · database file hash unchanged.
Report: `.artifacts/phase11/canary_report.json`.

```bash
docker build -f docker/model-server.Dockerfile -t schemaforge-model-server:local .
uv run python scripts/phase11_model_manifest.py --base BASE --lora LORA --out-dir MODELDIR   # files in MODELDIR
uv run python scripts/canary_phase11.py --model-dir MODELDIR --base-name base-Q4_K_M.gguf --lora-name lora-1518-f16.gguf --threads 6
```

Local result (Docker Desktop, 6 threads, CPU): **PASSED** — ready in 67 s, model loaded once,
per-query 4.0/6.7/7.1 s total, ~11.6 generated tokens/s, ~42 prompt tokens/s, saturation reject
0 ms, cancel 79 ms. These numbers are for this laptop, not for EC2.

## 9. Limitations

* Single host, single backend worker, in-process rate limiter — a demo, not an HA service.
* CPU inference is slow on small instances; there is no streaming to the UI (a query waits for the whole answer).
* Readiness is a liveness/loaded check of the server, not a model-quality check; `semantic_correctness` stays `not_verified`.
* The HTTPS/Caddy, EC2 bootstrap, Vercel and hash-verification steps were validated for syntax
  (`caddy validate`, `docker compose config`, image builds) and, for the model/backends, end to end
  locally — **not** on AWS/Vercel; that is the manual gate.
* Free-tier eligibility and instance RAM must be confirmed by you in the AWS console.
* GPU offload is configured (`docker-compose.gpu.yml`, `LLAMA_NGL`) but was **not** exercised
  (no GPU here); it is optional and not part of the gate.
* `cache_prompt` is off, so repeated schemas are re-prefilled each time (deliberate, for determinism).
* Pre-existing, unrelated: frontend test `Landing … shows the three frozen benchmarks` expects
  `4.20%`, which `frontend/src/data/results.ts` does not contain (untouched by Phase 11).

## 10. Manual cloud proof (operator sequence)

1. Confirm the account's Free Tier/credits cover an 8 GiB instance; launch it with the §7 security group.
2. Run `deploy/aws/bootstrap_ec2.sh`; clone the repo; `scp` the two GGUFs + `SHA256SUMS` into `/opt/schemaforge/models`.
3. Copy `deploy/production.env.example` → `.env` (domain, Vercel origin, `LLAMA_THREADS`), then `docker compose up -d --build`.
4. Wait for `/ready` = 200; run `deploy/aws/verify_stack.sh https://<domain>`.
5. Deploy `frontend/` to Vercel with `VITE_API_BASE_URL=https://<domain>`; add the Vercel URL to CORS.
6. From a **different device/network**, open the Vercel URL and run a real natural-language query;
   confirm safe SQL and the correct result; save screenshots + `docker compose logs backend` (no prompts/rows are logged).
7. `docker compose down -v`, **terminate** the instance, release the IP, delete leftover volumes/snapshots/SG/key, and verify in every region that nothing is running.

## Phase 11 acceptance

**Implementation gate (this change):** persistent runtime implemented; model stays loaded across
requests; subprocess fallback preserved; Docker/Compose production config exists; CPU serving works;
GPU-capable config present but optional; readiness/concurrency/cancellation verified against a real
server; Vercel/public-backend configuration exists; AWS runbook exists; `uv run pytest -q` and the
local `canary_phase11.py` pass. **Overall Phase 11 is NOT complete** until §10 is done from a
device that does not depend on the development laptop and the cloud resources are terminated.
