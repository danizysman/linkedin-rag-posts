"""Tests for deduplication, batching, and ingestion."""

from __future__ import annotations

from unittest.mock import patch

from qdrant_client import QdrantClient, models

from src.enrich import PostEnrichment
from src.ingest import ingest_records, make_point_id


def _fake_embed(texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
    dense = [[0.1] * 384 for _ in texts]
    sparse = [models.SparseVector(indices=[1, 2], values=[1.0, 0.5]) for _ in texts]
    return dense, sparse


def test_deterministic_point_id() -> None:
    urn = "urn:li:activity:123456"
    assert make_point_id(urn) == make_point_id(urn)
    assert make_point_id(urn) != make_point_id("urn:li:activity:999999")


def test_ingest_deduplicates_and_skips_existing_before_gemini() -> None:
    qdrant = QdrantClient(":memory:")
    records = [
        {
            "post_urn": "urn:li:activity:1",
            "post_url": "https://linkedin.com/1",
            "author_name": "Alice",
            "post_text": "First post",
            "has_image": False,
            "is_company_post": False,
        },
        {
            "post_urn": "urn:li:activity:1",  # in-batch duplicate
            "post_url": "https://linkedin.com/1",
            "author_name": "Alice",
            "post_text": "First post",
        },
        {
            "post_urn": "urn:li:activity:2",
            "post_url": "https://linkedin.com/2",
            "author_name": "Bob",
            "post_text": "Second post",
        },
        {
            "post_urn": "",  # invalid URN -> failed
            "post_text": "Broken",
        },
    ]

    with (
        patch("src.ingest.enrich_post") as mock_enrich,
        patch("src.ingest.embed_texts", side_effect=_fake_embed),
    ):
        mock_enrich.return_value = PostEnrichment(ai_tags=["ai"], ai_summary="Summary.")
        first = ingest_records(records, qdrant_client=qdrant, batch_size=1)
        assert first == {"ingested": 2, "skipped": 1, "failed": 1}
        assert mock_enrich.call_count == 2

        # Re-ingest same records: Qdrant pre-check must skip without calling Gemini
        mock_enrich.reset_mock()
        second = ingest_records(records, qdrant_client=qdrant, batch_size=1)
        assert second == {"ingested": 0, "skipped": 3, "failed": 1}
        mock_enrich.assert_not_called()
