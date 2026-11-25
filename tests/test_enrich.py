"""Tests for structured post enrichment."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.enrich import PostEnrichment, enrich_post, normalize_tags


def test_normalize_tags_lowercases_deduplicates_and_caps() -> None:
    tags = ["  AI ", "Python", "ai", "LLM", "Data-Science", "Cloud", "Extra"]
    assert normalize_tags(tags) == ["ai", "python", "llm", "data-science", "cloud"]


def test_enrich_post_skips_api_when_empty() -> None:
    mock_client = MagicMock()
    result = enrich_post(post_text="   ", post_image_alt=None, client=mock_client)
    assert result == PostEnrichment(ai_tags=[], ai_summary="")
    mock_client.models.generate_content.assert_not_called()


def test_enrich_post_uses_image_alt_fallback_and_parses() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        parsed=PostEnrichment(
            ai_tags=["AI", "Workforce", "ai"],
            ai_summary="  Summary text.  ",
        ),
        text=None,
    )

    result = enrich_post(
        post_text="",
        post_image_alt="Chart showing AI adoption",
        client=mock_client,
    )
    assert result.ai_tags == ["ai", "workforce"]
    assert result.ai_summary == "Summary text."
    mock_client.models.generate_content.assert_called_once()
