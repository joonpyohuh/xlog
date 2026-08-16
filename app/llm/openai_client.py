"""OpenAI wrapper — the cross-verification model for hallucination reduction.

Claude is the primary editor; OpenAI (GPT) is the independent second
witness. It re-checks Claude's factual claims (moments against frames) and
gives a second judging opinion. Every caller must treat failures as
non-fatal (fail open) — cross-checking may never block the pipeline.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def complete_json(
    system: str,
    user: str | list[dict[str, Any]],
    schema: dict[str, Any],
    schema_name: str = "result",
    max_tokens: int = 8192,
    model: str | None = None,
) -> dict[str, Any]:
    """Structured JSON completion via OpenAI (strict json_schema mode)."""
    resp = client().chat.completions.create(
        model=model or config.OPENAI_MODEL,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
    )
    return json.loads(resp.choices[0].message.content)


def frames_content(
    frames_b64: list[str],
    text: str,
    timestamps: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build an OpenAI user-content array: prompt text + labelled frames."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for i, b64 in enumerate(frames_b64):
        if timestamps is not None:
            content.append({"type": "text", "text": f"[frame @ {timestamps[i]:.1f}s]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    return content
