"""Thin wrapper around the Anthropic SDK for all xlog LLM stages.

Three entry points:
  - complete_text():   plain text generation
  - complete_json():   schema-constrained JSON via structured outputs
  - analyze_frames():  vision call with base64 frames interleaved with text
"""
from __future__ import annotations

import json
from typing import Any

import anthropic

from app import config

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def complete_text(system: str, user: str, max_tokens: int = 4096) -> str:
    resp = client().messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request (refusal)")
    return next(b.text for b in resp.content if b.type == "text")


def complete_json(
    system: str,
    user: str | list[dict[str, Any]],
    schema: dict[str, Any],
    max_tokens: int = 8192,
    effort: str | None = None,
) -> dict[str, Any]:
    """Structured-output call: response is guaranteed to match `schema`.
    `effort` ("low"|"medium"|"high") trades quality for latency per stage."""
    output_config: dict[str, Any] = {
        "format": {"type": "json_schema", "schema": schema}
    }
    if effort:
        output_config["effort"] = effort
    resp = client().messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config=output_config,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request (refusal)")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def frames_content(
    frames_b64: list[str],
    text: str,
    timestamps: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build a user-content array: prompt text + labelled frames."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for i, b64 in enumerate(frames_b64):
        if timestamps is not None:
            content.append({"type": "text", "text": f"[frame @ {timestamps[i]:.1f}s]"})
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            }
        )
    return content


def analyze_frames(
    system: str,
    prompt: str,
    frames_b64: list[str],
    schema: dict[str, Any],
    timestamps: list[float] | None = None,
    max_tokens: int = 8192,
    effort: str | None = None,
) -> dict[str, Any]:
    """Vision analysis returning schema-validated JSON."""
    content = frames_content(frames_b64, prompt, timestamps)
    return complete_json(system, content, schema, max_tokens=max_tokens, effort=effort)
