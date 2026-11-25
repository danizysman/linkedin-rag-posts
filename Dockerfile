# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

RUN /app/.venv/bin/python -c 'from fastembed import SparseTextEmbedding, TextEmbedding; TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"); SparseTextEmbedding(model_name="Qdrant/bm25")'

FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.cache/fastembed /app/.cache/fastembed
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
