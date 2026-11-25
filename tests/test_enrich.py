"""Tests for structured post enrichment."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import APIError

from src.enrich import PostEnrichment, enrich_post, normalize_tags


def test_normalize_tags_kebab_case_deduplicates_and_caps() -> None:
    tags = [
        "  AI & ML ",
        "Python_3",
        "ai ml",
        "   ---  ",
        "LLM Ops",
        "Data Science!",
        "Cloud Native",
        "Extra Tag",
    ]
    assert normalize_tags(tags) == [
        "ai-ml",
        "python-3",
        "llm-ops",
        "data-science",
        "cloud-native",
    ]


def test_enrich_post_skips_api_when_empty() -> None:
    mock_client = MagicMock()
    result = enrich_post(post_text="   ", post_image_alt=None, client=mock_client)
    assert result == PostEnrichment(ai_tags=[], ai_summary="")
    mock_client.models.generate_content.assert_not_called()


def test_enrich_post_uses_image_alt_fallback_and_parses() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        parsed=PostEnrichment(
            ai_tags=["AI Tools", "Workforce_Dev", "ai tools"],
            ai_summary="  Summary text.  ",
        ),
        text=None,
    )

    result = enrich_post(
        post_text="",
        post_image_alt="Chart showing AI adoption",
        client=mock_client,
    )
    assert result.ai_tags == ["ai-tools", "workforce-dev"]
    assert result.ai_summary == "Summary text."
    mock_client.models.generate_content.assert_called_once()


def test_enrich_post_retries_transient_errors_and_succeeds() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        APIError(503, {"error": {"status": "UNAVAILABLE", "message": "Overloaded"}}),
        APIError(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Rate limit"}}),
        SimpleNamespace(
            parsed=PostEnrichment(ai_tags=["Generative AI"], ai_summary="Recovered."),
            text=None,
        ),
    ]

    with patch("src.enrich.time.sleep") as mock_sleep:
        result = enrich_post(post_text="Hello AI", client=mock_client)

    assert result == PostEnrichment(ai_tags=["generative-ai"], ai_summary="Recovered.")
    assert mock_client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


def test_enrich_post_stops_after_max_retries_on_transient_error() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = APIError(
        503, {"error": {"status": "UNAVAILABLE", "message": "Down"}}
    )

    with (
        patch("src.enrich.time.sleep") as mock_sleep,
        pytest.raises(APIError),
    ):
        enrich_post(post_text="Hello AI", client=mock_client)

    assert mock_client.models.generate_content.call_count == 5
    assert mock_sleep.call_count == 4


def test_enrich_post_does_not_retry_non_transient_error() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = APIError(
        400, {"error": {"status": "INVALID_ARGUMENT", "message": "Bad request"}}
    )

    with (
        patch("src.enrich.time.sleep") as mock_sleep,
        pytest.raises(APIError),
    ):
        enrich_post(post_text="Hello AI", client=mock_client)

    assert mock_client.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()
