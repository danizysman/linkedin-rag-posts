"""Local MCP server exposing LinkedIn saved posts search and RAG over stdio."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from src.db import DateField, get_qdrant_client, hybrid_search
from src.main import synthesize_answer, to_search_result

mcp = MCPServer("linkedin-saved-rag")


@mcp.tool()
def search_saved_posts(
    q: str,
    limit: int = 10,
    author_name: str | None = None,
    ai_tags: list[str] | None = None,
    after_ts: int | None = None,
    before_ts: int | None = None,
    date_field: DateField = "posted_ts",
) -> list[dict[str, Any]]:
    """Search saved LinkedIn posts using hybrid dense + BM25 RRF retrieval."""
    client = get_qdrant_client()
    points = hybrid_search(
        client=client,
        query=q,
        limit=limit,
        author_name=author_name,
        ai_tags=ai_tags,
        after_ts=after_ts,
        before_ts=before_ts,
        date_field=date_field,
    )
    return [to_search_result(pt).model_dump() for pt in points]


@mcp.tool()
def ask_saved_posts(
    query: str,
    top_k: int = 5,
    author_name: str | None = None,
    ai_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve top-k saved LinkedIn posts and synthesize a cited answer."""
    client = get_qdrant_client()
    points = hybrid_search(
        client=client,
        query=query,
        limit=top_k,
        author_name=author_name,
        ai_tags=ai_tags,
    )
    sources = [to_search_result(pt) for pt in points]
    answer = synthesize_answer(query=query, sources=sources)
    return {
        "answer": answer,
        "sources": [src.model_dump() for src in sources],
    }


def main() -> None:
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
