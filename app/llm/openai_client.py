"""OpenAI wrapper — editor (Terra) plus independent mini witness.

Shot plans and footage frames use OPENAI_EDITOR_MODEL. Verify stays on
OPENAI_MODEL (mini). Failures are fail-open: quota/auth kills further
OpenAI calls for the rest of the process so we don't stall on 429s.
"""
from __future__ import annotations

import json
from pathlib import Path
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
    effort: str | None = None,
) -> dict[str, Any]:
    if _dead_reason is not None:
        raise RuntimeError(f"OpenAI disabled: {_dead_reason}")
    chosen = model or config.OPENAI_MODEL
    kwargs: dict[str, Any] = {
        "model": chosen,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
    }
    if effort and str(chosen).startswith("gpt-5"):
        kwargs["extra_body"] = {"reasoning_effort": effort}
    try:
        resp = client().chat.completions.create(**kwargs)
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


def transcribe_words(path: Path, offset: float = 0.0) -> list[dict]:
    """whisper-1 word timestamps. Cheap STT; empty list if the key is dead."""
    if not available() or not Path(path).is_file():
        return []
    try:
        with Path(path).open("rb") as f:
            resp = client().audio.transcriptions.create(
                model=config.STT_MODEL,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
    except Exception as e:  # noqa: BLE001 — filler trim is optional
        print(f"[openai] transcribe skipped: {e}")
        return []
    out = []
    for w in getattr(resp, "words", None) or []:
        token = (getattr(w, "word", None) or "").strip()
        if not token:
            continue
        out.append({
            "w": token,
            "t0": round(offset + float(getattr(w, "start", 0) or 0), 3),
            "t1": round(offset + float(getattr(w, "end", 0) or 0), 3),
        })
    return out


def analyze_frames(
    system: str,
    prompt: str,
    frames_b64: list[str],
    schema: dict[str, Any],
    timestamps: list[float] | None = None,
    max_tokens: int = 8192,
    model: str | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    content = frames_content(frames_b64, prompt, timestamps)
    return complete_json(
        system, content, schema,
        schema_name="moments", max_tokens=max_tokens,
        model=model or config.OPENAI_EDITOR_MODEL,
        effort=effort if effort is not None else config.OPENAI_EDITOR_EFFORT,
    )


if __name__ == "__main__":
    packed = frames_content(["abc"], "watch", [1.5])
    assert packed[0]["text"] == "watch"
    assert "1.5" in packed[1]["text"]
    assert packed[2]["image_url"]["url"].startswith("data:image/jpeg")
    print("openai_client self-check ok")
