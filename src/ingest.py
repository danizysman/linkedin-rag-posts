"""Ingestion pipeline and CLI for LinkedIn saved posts JSON exports."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from google import genai
from google.genai.errors import APIError
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from src.db import (
    COLLECTION_NAME,
    embed_texts,
    ensure_collection,
    get_existing_ids,
    get_qdrant_client,
)
from src.enrich import enrich_post, get_genai_client

logger = logging.getLogger(__name__)

BATCH_SIZE = 50


def make_point_id(post_urn: str) -> str:
    """Generate a deterministic UUIDv5 point ID from a LinkedIn post URN."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, post_urn))


def _build_embedding_text(record: dict[str, Any], ai_tags: list[str], ai_summary: str) -> str:
    """Combine post text (or image alt fallback) with AI summary and tags for embedding."""
    base_text = (record.get("post_text") or "").strip() or (
        record.get("post_image_alt") or ""
    ).strip()
    parts = [base_text] if base_text else []
    if ai_summary:
        parts.append(f"Summary: {ai_summary}")
    if ai_tags:
        parts.append(f"Tags: {', '.join(ai_tags)}")
    return "\n\n".join(parts) or str(record.get("post_urn", ""))


def _build_payload(
    record: dict[str, Any],
    ai_tags: list[str],
    ai_summary: str,
) -> dict[str, Any]:
    """Extract normalized Qdrant payload fields from a LinkedIn post record."""
    return {
        "post_urn": record.get("post_urn"),
        "post_url": record.get("post_url"),
        "author_name": record.get("author_name") or "",
        "author_headline": record.get("author_headline"),
        "post_text": record.get("post_text") or "",
        "ai_tags": ai_tags,
        "ai_summary": ai_summary,
        "has_image": bool(record.get("has_image", False)),
        "has_video": bool(record.get("has_video", False)),
        "is_company_post": bool(record.get("is_company_post", False)),
        "post_image_alt": record.get("post_image_alt"),
        "posted_time": record.get("posted_time"),
        "connection_degree": record.get("connection_degree"),
    }


def ingest_records(
    records: list[dict[str, Any]],
    qdrant_client: QdrantClient | None = None,
    genai_client: genai.Client | None = None,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, int]:
    """Deduplicate, enrich, embed, and upsert LinkedIn post records in batches."""
    qdrant = qdrant_client or get_qdrant_client()
    ensure_collection(qdrant, collection_name=collection_name)

    target_records = records[:limit] if limit is not None else records
    seen_urns: set[str] = set()
    unique_items: list[tuple[str, dict[str, Any]]] = []
    skipped = 0
    failed = 0

    for record in target_records:
        post_urn = record.get("post_urn")
        if not isinstance(post_urn, str) or not post_urn.strip():
            failed += 1
            continue
        if post_urn in seen_urns:
            skipped += 1
            continue
        seen_urns.add(post_urn)
        unique_items.append((make_point_id(post_urn), record))

    existing_ids = get_existing_ids(
        qdrant,
        [pid for pid, _ in unique_items],
        collection_name=collection_name,
    )
    new_items = [(pid, rec) for pid, rec in unique_items if pid not in existing_ids]
    skipped += len(unique_items) - len(new_items)

    if dry_run:
        return {"ingested": len(new_items), "skipped": skipped, "failed": failed}

    llm = genai_client or get_genai_client()
    ingested = 0

    for start in range(0, len(new_items), batch_size):
        batch = new_items[start : start + batch_size]
        enriched_batch: list[tuple[str, dict[str, Any], list[str], str, str]] = []

        for point_id, record in batch:
            try:
                enrichment = enrich_post(
                    post_text=record.get("post_text"),
                    post_image_alt=record.get("post_image_alt"),
                    client=llm,
                )
            except (APIError, ValidationError, ValueError, RuntimeError) as exc:
                logger.warning("Failed to enrich %s: %s", record.get("post_urn"), exc)
                failed += 1
                continue

            embed_text = _build_embedding_text(
                record,
                ai_tags=enrichment.ai_tags,
                ai_summary=enrichment.ai_summary,
            )
            enriched_batch.append(
                (point_id, record, enrichment.ai_tags, enrichment.ai_summary, embed_text)
            )

        if not enriched_batch:
            continue

        dense_vecs, sparse_vecs = embed_texts([item[4] for item in enriched_batch])
        points = [
            models.PointStruct(
                id=point_id,
                vector={"dense": dense_vec, "sparse": sparse_vec},
                payload=_build_payload(record, ai_tags=ai_tags, ai_summary=ai_summary),
            )
            for (point_id, record, ai_tags, ai_summary, _), dense_vec, sparse_vec in zip(
                enriched_batch, dense_vecs, sparse_vecs, strict=True
            )
        ]

        qdrant.upsert(collection_name=collection_name, points=points)
        ingested += len(points)
        logger.info("Upserted batch %d-%d (%d total ingested)", start, start + len(batch), ingested)

    return {"ingested": ingested, "skipped": skipped, "failed": failed}


def load_records_from_path(path: Path) -> list[dict[str, Any]]:
    """Load LinkedIn post records from a single JSON file or a directory of *.json files."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    records: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise TypeError(f"Expected a JSON array in {file_path}")
        records.extend(data)
    return records


def main() -> None:
    """CLI entrypoint for ingesting LinkedIn saved posts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Ingest LinkedIn Saved Post Hero JSON exports into Qdrant."
    )
    parser.add_argument("path", type=Path, help="Path to a .json file or directory of *.json files")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Upsert batch size")
    parser.add_argument("--limit", type=int, default=None, help="Max input records to process")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check deduplication counts without calling Gemini or upserting",
    )
    args = parser.parse_args()

    records = load_records_from_path(args.path)
    summary = ingest_records(
        records,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
