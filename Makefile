.PHONY: local ingest-local ingest-cloud deploy proxy test lint

REGION ?= us-east4
SERVICE ?= linkedin-saved-rag
PORT ?= 8080
CLOUD_ENV_FILE ?= .env.cloud

local:
	@docker start qdrant-local >/dev/null 2>&1 || true
	QDRANT_URL=http://localhost:6333 QDRANT_API_KEY= uv run uvicorn src.main:app --reload --port $(PORT)

ingest-local:
	@docker start qdrant-local >/dev/null 2>&1 || true
	QDRANT_URL=http://localhost:6333 QDRANT_API_KEY= uv run python -m src.ingest data/

ingest-cloud:
	@test -f $(CLOUD_ENV_FILE) || (echo "Missing $(CLOUD_ENV_FILE). Copy .env.cloud.example to $(CLOUD_ENV_FILE)." && exit 1)
	@set -a && . ./$(CLOUD_ENV_FILE) && set +a && \
		test -n "$$CLOUD_QDRANT_URL" && \
		case "$$CLOUD_QDRANT_URL" in *localhost*|*127.0.0.1*) echo "CLOUD_QDRANT_URL must not be localhost" && exit 1;; esac && \
		QDRANT_URL="$$CLOUD_QDRANT_URL" uv run python -m src.ingest data/

deploy:
	@test -f $(CLOUD_ENV_FILE) || (echo "Missing $(CLOUD_ENV_FILE). Copy .env.cloud.example to $(CLOUD_ENV_FILE)." && exit 1)
	@set -a && . ./$(CLOUD_ENV_FILE) && set +a && \
		test -n "$$PROJECT_ID" || (echo "PROJECT_ID is required in $(CLOUD_ENV_FILE)" && exit 1) && \
		test -n "$$RUNTIME_SA" || (echo "RUNTIME_SA is required in $(CLOUD_ENV_FILE)" && exit 1) && \
		test -n "$$CLOUD_QDRANT_URL" || (echo "CLOUD_QDRANT_URL is required in $(CLOUD_ENV_FILE)" && exit 1) && \
		case "$$CLOUD_QDRANT_URL" in *localhost*|*127.0.0.1*) echo "CLOUD_QDRANT_URL must not be localhost" && exit 1;; esac && \
		gcloud run deploy $(SERVICE) \
			--source . \
			--project="$$PROJECT_ID" \
			--region="$(REGION)" \
			--service-account="$$RUNTIME_SA" \
			--min-instances=0 \
			--memory=2Gi \
			--cpu=1 \
			--cpu-boost \
			--timeout=300 \
			--set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=$$PROJECT_ID,GOOGLE_CLOUD_LOCATION=global,QDRANT_URL=$$CLOUD_QDRANT_URL,FASTEMBED_CACHE_PATH=/app/.cache/fastembed" \
			--set-secrets="QDRANT_API_KEY=qdrant-api-key:latest" \
			--no-allow-unauthenticated

proxy:
	@test -f $(CLOUD_ENV_FILE) || (echo "Missing $(CLOUD_ENV_FILE)." && exit 1)
	@set -a && . ./$(CLOUD_ENV_FILE) && set +a && \
		gcloud run services proxy $(SERVICE) --project="$$PROJECT_ID" --region="$(REGION)" --port=$(PORT)

lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .

test:
	uv run --no-sync pytest -v
