# Private llama.cpp model server: Q4_K_M base + hot LoRA, loaded ONCE and kept resident.
# Pinned to the llama.cpp build the frozen GGUF pipeline was measured on (Phase 7: b10964).
#   CPU (default): ghcr.io/ggml-org/llama.cpp:server-b10964
#   GPU (optional): --build-arg LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b10964
# GGUF files are mounted read-only at /models; they are never copied into the image.
ARG LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-b10964
FROM ${LLAMA_CPP_IMAGE}
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin llama
COPY docker/model-server-entrypoint.sh /usr/local/bin/model-server-entrypoint.sh
RUN chmod 0555 /usr/local/bin/model-server-entrypoint.sh
USER 10001
EXPOSE 8080
# /health is a cheap readiness probe (503 while the model loads); it never runs inference.
# Loading ~2.4 GB on a small CPU host can take a couple of minutes, hence the long start period.
HEALTHCHECK --interval=15s --timeout=4s --start-period=300s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["/usr/local/bin/model-server-entrypoint.sh"]
