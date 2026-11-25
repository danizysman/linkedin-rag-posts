"""FastAPI service for ingesting, searching, and asking questions over LinkedIn saved posts."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.db import (
    DateField,
    ensure_collection,
    get_dense_model,
    get_qdrant_client,
    get_sparse_model,
    hybrid_search,
)
from src.enrich import get_genai_client
from src.ingest import ingest_records

RAG_MODEL = "gemini-3.8-flash"

RAG_PROMPT = """You are a helpful assistant answering questions using the user's saved LinkedIn posts.
Use ONLY the provided posts as sources. Cite every claim inline using markdown links to the post's `post_url` (e.g. [Author Name](post_url)).
If the provided posts do not contain enough information, say so clearly.

Question: {query}

Sources:
{context}
"""


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    failed: int


class SearchResult(BaseModel):
    score: float
    post_urn: str
    post_url: str | None = None
    author_name: str = ""
    post_text: str = ""
    ai_tags: list[str] = Field(default_factory=list)
    ai_summary: str = ""
    has_image: bool = False
    is_company_post: bool = False
    posted_ts: int | None = None
    saved_ts: int | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    author_name: str | None = None
    ai_tags: list[str] | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]


def to_search_result(point: Any) -> SearchResult:
    """Convert a Qdrant ScoredPoint into a SearchResult model."""
    payload = point.payload or {}
    return SearchResult(
        score=float(point.score),
        post_urn=str(payload.get("post_urn", point.id)),
        post_url=payload.get("post_url"),
        author_name=payload.get("author_name") or "",
        post_text=payload.get("post_text") or "",
        ai_tags=list(payload.get("ai_tags") or []),
        ai_summary=payload.get("ai_summary") or "",
        has_image=bool(payload.get("has_image", False)),
        is_company_post=bool(payload.get("is_company_post", False)),
        posted_ts=payload.get("posted_ts"),
        saved_ts=payload.get("saved_ts"),
    )


def synthesize_answer(query: str, sources: list[SearchResult]) -> str:
    """Synthesize a cited markdown answer from retrieved SearchResult sources."""
    if not sources:
        return "No matching saved posts were found to answer your question."

    context_blocks = [
        (
            f"[{idx}] Author: {src.author_name or 'Unknown'}\n"
            f"URL: {src.post_url or 'N/A'}\n"
            f"Tags: {', '.join(src.ai_tags)}\n"
            f"Summary: {src.ai_summary}\n"
            f"Text: {src.post_text}"
        )
        for idx, src in enumerate(sources, start=1)
    ]
    prompt = RAG_PROMPT.format(query=query, context="\n\n---\n\n".join(context_blocks))
    response = get_genai_client().models.generate_content(
        model=RAG_MODEL,
        contents=prompt,
    )
    return (response.text or "").strip()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    client = get_qdrant_client()
    ensure_collection(client)
    get_dense_model()
    get_sparse_model()
    yield


app = FastAPI(
    title="LinkedIn Saved Posts RAG",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Healthcheck endpoint for local and Cloud Run probes."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: Annotated[UploadFile, File(...)]) -> IngestResponse:
    """Ingest an uploaded LinkedIn Saved Post Hero JSON export."""
    raw = await file.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array of posts.")

    result = ingest_records(data)
    return IngestResponse(**result)


@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    q: Annotated[str, Query(min_length=1)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    author_name: Annotated[str | None, Query()] = None,
    ai_tags: Annotated[list[str] | None, Query()] = None,
    after_ts: Annotated[int | None, Query()] = None,
    before_ts: Annotated[int | None, Query()] = None,
    date_field: Annotated[DateField, Query()] = "posted_ts",
) -> SearchResponse:
    """Hybrid dense + BM25 sparse search fused with RRF and optional payload/date filters."""
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
    return SearchResponse(results=[to_search_result(pt) for pt in points])


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    """Retrieve top-k hybrid results and synthesize a cited answer with gemini-3.8-flash."""
    client = get_qdrant_client()
    points = hybrid_search(
        client=client,
        query=body.query,
        limit=body.top_k,
        author_name=body.author_name,
        ai_tags=body.ai_tags,
    )
    sources = [to_search_result(pt) for pt in points]
    answer = synthesize_answer(query=body.query, sources=sources)
    return AskResponse(answer=answer, sources=sources)
