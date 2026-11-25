"""Structured post enrichment using Gemini 3.5 Flash-Lite."""

from __future__ import annotations

import logging
import random
import re
import time
from functools import lru_cache

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ENRICH_MODEL = "gemini-3.5-flash-lite"
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
TRANSIENT_CODES = {429, 503}
TRANSIENT_MARKERS = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")

ENRICH_PROMPT = (
    "Analyze this LinkedIn post and return:\n"
    "1. ai_tags: 3 to 5 concise, normalized lowercase kebab-case topic tags "
    "(e.g. 'llm', 'career-advice', 'python').\n"
    "2. ai_summary: A clear 1 to 2 sentence summary of the post's key takeaway.\n\n"
    "Post:\n{text}"
)


class PostEnrichment(BaseModel):
    """Structured enrichment output for a LinkedIn post."""

    ai_tags: list[str] = Field(
        default_factory=list,
        description="3 to 5 normalized lowercase kebab-case topic tags.",
    )
    ai_summary: str = Field(
        default="",
        description="1 to 2 sentence concise summary of the post.",
    )


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    """Return a cached Google GenAI client using GEMINI_API_KEY."""
    return genai.Client()


def normalize_tags(tags: list[str]) -> list[str]:
    """Normalize tags to kebab-case, drop empty strings, deduplicate, and cap at 5."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in tags:
        tag = re.sub(r"[^a-z0-9]+", "-", raw.strip().lower()).strip("-")
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
        if len(result) >= 5:
            break
    return result


def _is_transient_error(exc: Exception) -> bool:
    """Return True if exc represents a retryable rate-limit or availability error."""
    if isinstance(exc, APIError):
        if exc.code in TRANSIENT_CODES:
            return True
        if (exc.status or "").upper() in {"UNAVAILABLE", "RESOURCE_EXHAUSTED"}:
            return True
    message = str(exc).upper()
    return any(marker in message for marker in TRANSIENT_MARKERS)


def _generate_with_retry(
    client: genai.Client,
    text: str,
) -> types.GenerateContentResponse:
    """Call generate_content with exponential backoff and random jitter on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=ENRICH_MODEL,
                contents=ENRICH_PROMPT.format(text=text),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PostEnrichment,
                    temperature=0.1,
                ),
            )
        except Exception as exc:
            if attempt >= MAX_RETRIES or not _is_transient_error(exc):
                raise
            delay = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
            jitter = random.uniform(0, delay * 0.5)
            sleep_for = delay + jitter
            logger.warning(
                "Transient Gemini error (attempt %d/%d): %s. Retrying in %.2fs",
                attempt,
                MAX_RETRIES,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    raise RuntimeError("Unreachable retry loop exit")


def enrich_post(
    post_text: str | None,
    post_image_alt: str | None = None,
    client: genai.Client | None = None,
) -> PostEnrichment:
    """Call gemini-3.5-flash-lite to generate ai_tags and ai_summary."""
    text = (post_text or "").strip() or (post_image_alt or "").strip()
    if not text:
        return PostEnrichment(ai_tags=[], ai_summary="")

    genai_client = client or get_genai_client()
    response = _generate_with_retry(genai_client, text)

    parsed = response.parsed
    if not isinstance(parsed, PostEnrichment):
        parsed = PostEnrichment.model_validate_json(response.text or "{}")

    return PostEnrichment(
        ai_tags=normalize_tags(parsed.ai_tags),
        ai_summary=parsed.ai_summary.strip(),
    )
