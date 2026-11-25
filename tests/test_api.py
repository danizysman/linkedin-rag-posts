"""Tests for FastAPI endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.main import app


def _mock_scored_point() -> SimpleNamespace:
    return SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        score=0.95,
        payload={
            "post_urn": "urn:li:activity:1",
            "post_url": "https://www.linkedin.com/feed/update/urn:li:activity:1",
            "author_name": "Alice",
            "post_text": "Building RAG with Qdrant and Gemini.",
            "ai_tags": ["rag", "qdrant"],
            "ai_summary": "A guide on hybrid RAG.",
            "has_image": False,
            "is_company_post": False,
        },
    )


def test_health_endpoint() -> None:
    with (
        patch("src.main.get_qdrant_client"),
        patch("src.main.ensure_collection"),
        patch("src.main.get_dense_model"),
        patch("src.main.get_sparse_model"),
        TestClient(app) as client,
    ):
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_endpoint() -> None:
    with (
        patch("src.main.get_qdrant_client"),
        patch("src.main.ensure_collection"),
        patch("src.main.get_dense_model"),
        patch("src.main.get_sparse_model"),
        patch("src.main.ingest_records", return_value={"ingested": 1, "skipped": 0, "failed": 0}),
        TestClient(app) as client,
    ):
        resp = client.post(
            "/ingest",
            files={"file": ("posts.json", b'[{"post_urn": "urn:1"}]', "application/json")},
        )
    assert resp.status_code == 200
    assert resp.json() == {"ingested": 1, "skipped": 0, "failed": 0}


def test_search_and_ask_endpoints() -> None:
    mock_genai = MagicMock()
    mock_genai.models.generate_content.return_value = SimpleNamespace(
        text="You can build hybrid RAG with Qdrant [Alice](https://www.linkedin.com/feed/update/urn:li:activity:1)."
    )

    with (
        patch("src.main.get_qdrant_client"),
        patch("src.main.ensure_collection"),
        patch("src.main.get_dense_model"),
        patch("src.main.get_sparse_model"),
        patch("src.main.hybrid_search", return_value=[_mock_scored_point()]),
        patch("src.main.get_genai_client", return_value=mock_genai),
        TestClient(app) as client,
    ):
        search_resp = client.get("/search", params={"q": "hybrid rag", "limit": 5})
        assert search_resp.status_code == 200
        results = search_resp.json()["results"]
        assert len(results) == 1
        assert results[0]["author_name"] == "Alice"

        ask_resp = client.post("/ask", json={"query": "How to build RAG?", "top_k": 3})
        assert ask_resp.status_code == 200
        body = ask_resp.json()
        assert "https://www.linkedin.com/feed/update/urn:li:activity:1" in body["answer"]
        assert len(body["sources"]) == 1
