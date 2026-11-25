"""Structured post enrichment using Gemini 3.5 Flash-Lite."""

from __future__ import annotations

from functools import lru_cache

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

ENRICH_MODEL = "gemini-3.5-flash-lite"

ENRICH_PROMPT = (
    "Analyze this LinkedIn post and return:\n"
    "1. ai_tags: 3 to 5 concise, normalized lowercase topic tags (e.g. 'llm', 'career-advice', 'python').\n"
    "2. ai_summary: A clear 1 to 2 sentence summary of the post's key takeaway.\n\n"
    "Post:\n{text}"
)


class PostEnrichment(BaseModel):
    """Structured enrichment output for a LinkedIn post."""

    ai_tags: list[str] = Field(
        default_factory=list,
        description="3 to 5 normalized lowercase topic tags.",
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
    """Normalize tags to lowercase, deduplicate preserving order, and cap at 5."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in tags:
        tag = raw.strip().lower()
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
        if len(result) >= 5:
            break
    return result


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
    response = genai_client.models.generate_content(
        model=ENRICH_MODEL,
        contents=ENRICH_PROMPT.format(text=text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PostEnrichment,
            temperature=0.1,
        ),
    )

    parsed = response.parsed
    if not isinstance(parsed, PostEnrichment):
        parsed = PostEnrichment.model_validate_json(response.text or "{}")

    return PostEnrichment(
        ai_tags=normalize_tags(parsed.ai_tags),
        ai_summary=parsed.ai_summary.strip(),
    )
