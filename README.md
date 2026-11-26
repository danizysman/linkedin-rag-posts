# LinkedIn Saved Posts Hybrid Search & RAG

Local-first, Cloud Run–ready Hybrid Search & RAG service for LinkedIn saved posts exported as JSON from **LinkedIn Saved Post Hero**.

- **Vector DB**: Qdrant (`linkedin_posts` collection) with named **dense** vectors (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384-dim `COSINE`) and **sparse** BM25 vectors (`Qdrant/bm25`, `Modifier.IDF`) fused with Reciprocal Rank Fusion (`Fusion.RRF`).
- **AI Enrichment & RAG**: `gemini-3.5-flash-lite` (structured kebab-case `ai_tags` + `ai_summary`) and `gemini-3.8-flash` (cited RAG answers with `post_url` links) via `google-genai` / Vertex AI.
- **Interfaces**: FastAPI HTTP service (`/docs`, `/search`, `/ask`, `/ingest`, `/health`), CLI ingestion (`python -m src.ingest`), local `stdio` MCP server (`src.mcp_server`), and a debounced macOS `launchd` watcher (`scripts/watch_data.py`).

---

## Environment Setup

1. **`.envrc`** (`direnv`): Active automatically in your shell for local runs, MCP, and the watcher:
   ```bash
   layout uv
   export GOOGLE_GENAI_USE_VERTEXAI=true
   export GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
   export GOOGLE_CLOUD_LOCATION=global
   export QDRANT_URL=https://<your-cluster>.qdrant.io:6333   # or http://localhost:6333 for local Docker
   export QDRANT_API_KEY=<your-qdrant-api-key>               # omit if using local unauthenticated Qdrant
   ```

2. **`.env.cloud`** (gitignored): Copy from [.env.cloud.example](.env.cloud.example) for `make ingest-cloud`, `make deploy`, and `make proxy`:
   ```bash
   cp .env.cloud.example .env.cloud
   ```
   - `PROJECT_ID`: GCP project ID
   - `RUNTIME_SA`: Least-privilege Cloud Run service account (`roles/aiplatform.user` + `roles/secretmanager.secretAccessor` on `qdrant-api-key`)
   - `CLOUD_QDRANT_URL`: Qdrant Cloud URL (`https://...:6333`, guarded against `localhost`)
   - `QDRANT_API_KEY`: Read-write Qdrant Cloud key used locally by `make ingest-cloud` (Cloud Run itself reads a read-only key from GCP Secret Manager `qdrant-api-key:latest`)

---

## Common Commands (`Makefile`)

```bash
# Local development (auto-starts qdrant-local Docker container and runs FastAPI on :8080)
make local

# Ingest staged data/*.json exports into local Qdrant
make ingest-local

# Ingest staged data/*.json exports into Qdrant Cloud (skips already-ingested post_urns)
make ingest-cloud

# Deploy to Cloud Run (scales to 0 when idle, --no-allow-unauthenticated)
make deploy

# Proxy the authenticated Cloud Run service to http://localhost:8080 (including /docs)
make proxy

# Lint, format check, and unit tests
make lint
make test
```

### CLI Ingestion Options

```bash
# Dry-run deduplication check without calling Gemini or upserting
uv run python -m src.ingest data/ --dry-run

# Ingest a single export file with a limit or custom concurrency
uv run python -m src.ingest data/linkedin-saved-posts-2026-11-26.json --limit 10 --workers 6
```

---

## API Usage & Interactive `/docs`

- **Local**: Run `make local` and open `http://localhost:8080/docs`.
- **Deployed (Cloud Run)**: Run `make proxy` and open `http://localhost:8080/docs` (the proxy attaches your `gcloud` identity token automatically).

### Endpoints

- `GET /health` — Healthcheck (`{"status": "ok"}`).
- `POST /ingest` — Upload a LinkedIn Saved Post Hero `.json` export; returns `{"ingested", "skipped", "failed"}`.
- `GET /search` — Hybrid RRF search with optional filters:
  - `q`: Search query (required)
  - `limit`: Max results (`1–50`, default `10`)
  - `author_name`: Exact author match
  - `ai_tags`: Repeated query param (`?ai_tags=llm&ai_tags=rag`)
  - `after_ts`, `before_ts`, `date_field` (`posted_ts` or `saved_ts`): Unix epoch seconds range filter
  ```bash
  curl "http://localhost:8080/search?q=vibe+coding&limit=5"
  curl "http://localhost:8080/search?q=agents&after_ts=1735689600&date_field=posted_ts"
  ```
- `POST /ask` — Top-$k$ hybrid retrieval + `gemini-3.8-flash` cited synthesis:
  ```bash
  curl -X POST "http://localhost:8080/ask" \
    -H "Content-Type: application/json" \
    -d '{"query": "What predicts vibe coding proficiency?", "top_k": 5}'
  ```

---

## Local MCP Server (`stdio`)

[src/mcp_server.py](src/mcp_server.py) exposes `search_saved_posts` and `ask_saved_posts` over `stdio`:

```bash
uv run python -m src.mcp_server
```

To register in Antigravity (`~/.gemini/config/mcp_config.json`) without duplicating secrets, use `direnv exec`:

```json
{
  "mcpServers": {
    "linkedin-saved-rag": {
      "command": "direnv",
      "args": [
        "exec",
        "/Users/dzysman/Projects/linkedin-saved-rag",
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "src.mcp_server"
      ]
    }
  }
}
```

---

## Automated `data/` Watcher (macOS `launchd`)

[scripts/watch_data.py](scripts/watch_data.py) polls `data/` for new/updated `.json` files (ignoring `.crdownload`/`.part`/`.tmp` partial downloads), waits for write stability, and runs `uv run python -m src.ingest data/` under a non-blocking `flock` lockfile.

To run it automatically on login on macOS:

```bash
cp scripts/com.linkedin-saved-rag.watcher.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.linkedin-saved-rag.watcher.plist
```

Logs are written to `/tmp/linkedin-saved-rag-watcher.out.log` and `/tmp/linkedin-saved-rag-watcher.err.log`.
