"""Tests for local stdio MCP server tool handlers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.mcp_server import ask_saved_posts, search_saved_posts


def _mock_point() -> SimpleNamespace:
    return SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        score=0.91,
        payload={
            "post_urn": "urn:li:activity:42",
            "post_url": "https://www.linkedin.com/feed/update/urn:li:activity:42",
            "author_name": "Alice",
            "post_text": "Hybrid search with Qdrant.",
            "ai_tags": ["qdrant", "search"],
            "ai_summary": "Overview of hybrid search.",
            "has_image": False,
            "is_company_post": False,
            "posted_ts": 1768480200,
            "saved_ts": None,
        },
    )


def test_mcp_search_saved_posts() -> None:
    with (
        patch("src.mcp_server.get_qdrant_client"),
        patch("src.mcp_server.hybrid_search", return_value=[_mock_point()]) as mock_search,
    ):
        results = search_saved_posts(
            q="hybrid search",
            limit=5,
            author_name="Alice",
            ai_tags=["qdrant"],
            after_ts=1760000000,
            before_ts=1770000000,
        )

    assert len(results) == 1
    assert results[0]["post_urn"] == "urn:li:activity:42"
    assert results[0]["posted_ts"] == 1768480200
    mock_search.assert_called_once()


def test_mcp_ask_saved_posts() -> None:
    with (
        patch("src.mcp_server.get_qdrant_client"),
        patch("src.mcp_server.hybrid_search", return_value=[_mock_point()]),
        patch(
            "src.mcp_server.synthesize_answer",
            return_value="Use Qdrant [Alice](https://www.linkedin.com/feed/update/urn:li:activity:42).",
        ),
    ):
        response = ask_saved_posts(query="How to do hybrid search?", top_k=3)

    assert "https://www.linkedin.com/feed/update/urn:li:activity:42" in response["answer"]
    assert len(response["sources"]) == 1
