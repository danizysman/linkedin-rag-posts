"""Tests for timestamp normalization and date range filtering."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import models

from src.db import build_filter
from src.ingest import parse_timestamp
from src.main import app


def test_parse_timestamp_formats() -> None:
    assert parse_timestamp("2026-01-15T12:30:00Z") == 1768480200
    assert parse_timestamp("2026-01-15") == 1768435200
    assert parse_timestamp(1768480200) == 1768480200
    assert parse_timestamp(1768480200000) == 1768480200
    assert parse_timestamp("1768480200000") == 1768480200
    # Relative LinkedIn strings must not be guessed
    assert parse_timestamp("3mo") is None
    assert parse_timestamp("12h •") is None
    assert parse_timestamp("Reposted from Rola Dali • 11mo •") is None
    assert parse_timestamp(None) is None


def test_build_filter_date_range() -> None:
    qfilter = build_filter(
        author_name="Alice",
        after_ts=1700000000,
        before_ts=1750000000,
        date_field="saved_ts",
    )
    assert qfilter is not None
    assert len(qfilter.must) == 2
    range_cond = qfilter.must[1]
    assert isinstance(range_cond, models.FieldCondition)
    assert range_cond.key == "saved_ts"
    assert range_cond.range == models.Range(gte=1700000000, lte=1750000000)


def test_build_filter_rejects_invalid_date_field() -> None:
    with pytest.raises(ValueError, match="Unsupported date_field"):
        build_filter(after_ts=1700000000, date_field="unknown_field")  # type: ignore[arg-type]


def test_search_endpoint_forwards_date_filters() -> None:
    with (
        patch("src.main.get_qdrant_client"),
        patch("src.main.ensure_collection"),
        patch("src.main.get_dense_model"),
        patch("src.main.get_sparse_model"),
        patch("src.main.hybrid_search", return_value=[]) as mock_search,
        TestClient(app) as client,
    ):
        resp = client.get(
            "/search",
            params={
                "q": "agents",
                "after_ts": 1700000000,
                "before_ts": 1750000000,
                "date_field": "saved_ts",
            },
        )
    assert resp.status_code == 200
    mock_search.assert_called_once()
    kwargs = mock_search.call_args.kwargs
    assert kwargs["after_ts"] == 1700000000
    assert kwargs["before_ts"] == 1750000000
    assert kwargs["date_field"] == "saved_ts"
