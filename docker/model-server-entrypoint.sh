#!/usr/bin/env bash
# Launches llama-server from environment variables (see deploy/production.env.example).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
BASE="${MODEL_DIR}/${BASE_GGUF_NAME:?BASE_GGUF_NAME is required}"
LORA="${MODEL_DIR}/${LORA_GGUF_NAME:?LORA_GGUF_NAME is required}"
CTX="${LLAMA_CTX:-8192}"          # per-slot context; Phase 7 config value (longest prompt + output)
PARALLEL="${LLAMA_PARALLEL:-1}"   # must equal SCHEMAFORGE_LLAMA_PARALLEL on the backend
NGL="${LLAMA_NGL:-0}"             # 0 = CPU. GPU offload: e.g. 99 (needs the CUDA image + a GPU host)
THREADS="${LLAMA_THREADS:--1}"    # -1 = llama.cpp default

for f in "$BASE" "$LORA"; do
  [ -f "$f" ] || { echo "model-server: missing artifact $(basename "$f") in ${MODEL_DIR} (mount it read-only)" >&2; exit 2; }
done

# Optional provenance check against a SHA256SUMS file next to the models
# (scripts/phase11_model_manifest.py writes it). Roughly 10-30 s for the 2.4 GB pair.
if [ "${VERIFY_SHA256:-1}" = "1" ] && [ -f "${MODEL_DIR}/SHA256SUMS" ]; then
  echo "model-server: verifying artifact hashes"
  # The mount is read-only; sha256sum only reads.
  (cd "$MODEL_DIR" && sha256sum -c SHA256SUMS >&2) || { echo "model-server: hash mismatch, refusing to start" >&2; exit 3; }
fi

# Each slot gets a full CTX (-c is the total): no silent context shrink when PARALLEL > 1.
# --cache-ram 0: no host-RAM prompt cache (predictable memory on small hosts; requests also send cache_prompt=false).
# shellcheck disable=SC2086  # LLAMA_EXTRA_ARGS is intentionally word-split
exec /app/llama-server \
  -m "$BASE" --lora "$LORA" \
  -c "$((CTX * PARALLEL))" --parallel "$PARALLEL" -ngl "$NGL" -t "$THREADS" \
  --cache-ram 0 --no-webui --metrics \
  --host 0.0.0.0 --port 8080 \
  ${LLAMA_EXTRA_ARGS:-}
