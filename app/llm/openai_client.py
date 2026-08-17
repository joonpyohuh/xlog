"""OpenAI wrapper — the independent second witness.

Claude writes and judges; GPT re-checks Claude's claims against the same
frames and gives a second judging opinion. Failures are non-fatal: callers
must treat them as fail-open. An exhausted quota or a bad key kills further
OpenAI calls for the rest of the process so we don't stall on 429s.
"""
from __future__ import annotations

import json
from typing import Any

from openai import (
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from app import config

_client: OpenAI | None = None
_dead_reason: str | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=1)
    return _client


def available() -> bool:
    return bool(config.OPENAI_API_KEY) and _dead_reason is None


def mark_dead(error: Exception) -> None:
    global _dead_reason
    if isinstance(error, (AuthenticationError, PermissionDeniedError)) or (
        isinstance(error, RateLimitError)
        and ("quota" in str(error) or "credit" in str(error))
    ):
        _dead_reason = type(error).__name__
        print(f"[openai] disabled for this run: {error}")


def complete_json(
    system: str,
    user: str | list[dict[str, Any]],
    schema: dict[str, Any],
    schema_name: str = "result",
    max_tokens: int = 8192,
    model: str | None = None,
) -> dict[str, Any]:
    if _dead_reason is not None:
        raise RuntimeError(f"OpenAI disabled: {_dead_reason}")
    try:
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
    except (AuthenticationError, PermissionDeniedError, RateLimitError) as e:
        mark_dead(e)
        raise
    raw = resp.choices[0].message.content
    if not raw:
        raise RuntimeError("OpenAI returned empty content")
    return json.loads(raw)


def frames_content(
    frames_b64: list[str],
    text: str,
    timestamps: list[float] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for i, b64 in enumerate(frames_b64):
        if timestamps is not None:
            content.append({"type": "text", "text": f"[frame @ {timestamps[i]:.1f}s]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content
