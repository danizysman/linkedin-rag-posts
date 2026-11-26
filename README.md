# LinkedIn Saved Posts Hybrid Search & RAG

Local-first, Cloud Run–ready Hybrid Search & RAG service for LinkedIn saved posts exported as JSON from **LinkedIn Saved Post Hero**.

## Quickstart

```bash
# Local development (auto-starts qdrant-local and runs FastAPI on :8080)
make local

# Ingest staged JSON exports locally
make ingest-local

# Cloud Run deploy (requires .env.cloud copied from .env.cloud.example)
cp .env.cloud.example .env.cloud
make deploy

# Proxy authenticated Cloud Run service to localhost:8080
make proxy
```

## Date Filtering

During ingestion, parseable timestamps (`posted_time`, `saved_at`, `scraped_at`, `timestamp`, `date`) are normalized to integer Unix epoch seconds (`posted_ts`, `saved_ts`) and indexed in Qdrant (`PayloadSchemaType.INTEGER`). Relative strings like `"3mo"` or `"12h •"` are stored as `null`.

Filter `GET /search` using `after_ts`, `before_ts`, and `date_field` (`posted_ts` or `saved_ts`):

```bash
curl "http://localhost:8080/search?q=agents&after_ts=1735689600&date_field=posted_ts"
```

## Automated Watcher (`launchd` on macOS)

Copy [scripts/com.linkedin-saved-rag.watcher.plist](scripts/com.linkedin-saved-rag.watcher.plist) into `~/Library/LaunchAgents/` and load it:

```bash
cp scripts/com.linkedin-saved-rag.watcher.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.linkedin-saved-rag.watcher.plist
```

Logs are written to `/tmp/linkedin-saved-rag-watcher.out.log` and `/tmp/linkedin-saved-rag-watcher.err.log`.

## Local MCP Server (stdio)

Run the MCP server over stdio:

```bash
uv run python -m src.mcp_server
```

Exposes `search_saved_posts` and `ask_saved_posts`.
