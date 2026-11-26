"""Qdrant connection, collection setup, FastEmbed models, and hybrid search."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "linkedin_posts")
DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DENSE_VECTOR_SIZE = 384
SPARSE_MODEL_NAME = "Qdrant/bm25"

DateField = Literal["posted_ts", "saved_ts"]
VALID_DATE_FIELDS = {"posted_ts", "saved_ts"}

PAYLOAD_INDEXES: list[tuple[str, models.PayloadSchemaType]] = [
    ("author_name", models.PayloadSchemaType.KEYWORD),
    ("ai_tags", models.PayloadSchemaType.KEYWORD),
    ("has_image", models.PayloadSchemaType.BOOL),
    ("is_company_post", models.PayloadSchemaType.BOOL),
    ("posted_ts", models.PayloadSchemaType.INTEGER),
    ("saved_ts", models.PayloadSchemaType.INTEGER),
]


def get_qdrant_client() -> QdrantClient:
    """Create a QdrantClient configured from environment variables."""
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    """Return the cached multilingual dense FastEmbed model."""
    return TextEmbedding(model_name=DENSE_MODEL_NAME)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    """Return the cached BM25 sparse FastEmbed model."""
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def ensure_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Create the collection and any missing payload indexes idempotently."""
    existing_indexes: set[str] = set()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
    else:
        info = client.get_collection(collection_name)
        existing_indexes = set((info.payload_schema or {}).keys())

    for field_name, field_schema in PAYLOAD_INDEXES:
        if field_name in existing_indexes:
            continue
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )


def get_existing_ids(
    client: QdrantClient,
    point_ids: list[str],
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 500,
) -> set[str]:
    """Return the subset of point_ids that already exist in Qdrant."""
    existing: set[str] = set()
    for i in range(0, len(point_ids), chunk_size):
        chunk = point_ids[i : i + chunk_size]
        points = client.retrieve(
            collection_name=collection_name,
            ids=chunk,
            with_payload=False,
            with_vectors=False,
        )
        existing.update(str(pt.id) for pt in points)
    return existing


def embed_texts(
    texts: list[str],
) -> tuple[list[list[float]], list[models.SparseVector]]:
    """Compute dense and BM25 sparse embeddings for a batch of texts."""
    dense_vecs = [vec.tolist() for vec in get_dense_model().embed(texts)]
    sparse_vecs = [
        models.SparseVector(
            indices=sp.indices.tolist(),
            values=sp.values.tolist(),
        )
        for sp in get_sparse_model().embed(texts)
    ]
    return dense_vecs, sparse_vecs


def build_filter(
    author_name: str | None = None,
    ai_tags: list[str] | None = None,
    after_ts: int | None = None,
    before_ts: int | None = None,
    date_field: DateField = "posted_ts",
) -> models.Filter | None:
    """Build an optional Qdrant Filter for author_name, ai_tags, and timestamp range."""
    if date_field not in VALID_DATE_FIELDS:
        raise ValueError(f"Unsupported date_field: {date_field}")

    conditions: list[models.Condition] = []
    if author_name:
        conditions.append(
            models.FieldCondition(
                key="author_name",
                match=models.MatchValue(value=author_name),
            )
        )
    if ai_tags:
        normalized = [tag.strip().lower() for tag in ai_tags if tag.strip()]
        if normalized:
            conditions.append(
                models.FieldCondition(
                    key="ai_tags",
                    match=models.MatchAny(any=normalized),
                )
            )
    if after_ts is not None or before_ts is not None:
        conditions.append(
            models.FieldCondition(
                key=date_field,
                range=models.Range(gte=after_ts, lte=before_ts),
            )
        )
    return models.Filter(must=conditions) if conditions else None


def hybrid_search(
    client: QdrantClient,
    query: str,
    limit: int = 10,
    author_name: str | None = None,
    ai_tags: list[str] | None = None,
    after_ts: int | None = None,
    before_ts: int | None = None,
    date_field: DateField = "posted_ts",
    collection_name: str = COLLECTION_NAME,
) -> list[models.ScoredPoint]:
    """Run hybrid dense + BM25 sparse search fused with Reciprocal Rank Fusion."""
    dense_query = next(iter(get_dense_model().query_embed(query))).tolist()
    sparse_raw = next(iter(get_sparse_model().query_embed(query)))
    sparse_query = models.SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist(),
    )

    query_filter = build_filter(
        author_name=author_name,
        ai_tags=ai_tags,
        after_ts=after_ts,
        before_ts=before_ts,
        date_field=date_field,
    )
    prefetch_limit = max(limit * 3, 20)

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using="dense",
                limit=prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_query,
                using="sparse",
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return response.points
