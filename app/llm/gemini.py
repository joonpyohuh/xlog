"""Gemini for bulk video understanding and cheap JSON rewrites.

Native video in, JSON moments out — one call per source instead of JPEG batches.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app import config

_client = None


def available() -> bool:
    return bool(config.GEMINI_API_KEY)


def client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini rejects additionalProperties / some JSON-Schema-07 extras."""
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k != "additionalProperties"}
    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {k: _schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _schema(out["items"])
    return out


def complete_json(
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int = 8192,
    model: str | None = None,
) -> dict[str, Any]:
    from google.genai import types

    resp = client().models.generate_content(
        model=model or config.GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=_schema(schema),
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty JSON")
    return json.loads(text)


def _wait_file(uploaded, timeout_sec: float = 180.0):
    deadline = time.time() + timeout_sec
    name = uploaded.name
    while time.time() < deadline:
        info = client().files.get(name=name)
        state = str(getattr(info, "state", "") or "")
        if state.endswith("ACTIVE") or state == "ACTIVE":
            return info
        if state.endswith("FAILED") or state == "FAILED":
            raise RuntimeError(f"Gemini file processing failed: {state}")
        time.sleep(2)
    raise TimeoutError(f"Gemini file {name} not ACTIVE after {timeout_sec:.0f}s")


def analyze_video(
    path: Path,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    max_tokens: int = 8192,
    model: str | None = None,
    fps: float = 1.0,
    media: str = "low",
) -> dict[str, Any]:
    """Upload one source file and extract schema-validated moments."""
    from google.genai import types

    uploaded = client().files.upload(file=str(path))
    try:
        ready = _wait_file(uploaded)
        mime = getattr(ready, "mime_type", None) or "video/mp4"
        uri = getattr(ready, "uri", None)
        video_part = types.Part(
            file_data=types.FileData(file_uri=uri, mime_type=mime),
            video_metadata=types.VideoMetadata(fps=float(fps) or 1.0),
        )
        res = (
            types.MediaResolution.MEDIA_RESOLUTION_HIGH
            if (media or "").lower() == "high"
            else types.MediaResolution.MEDIA_RESOLUTION_LOW
        )
        resp = client().models.generate_content(
            model=model or config.GEMINI_MODEL,
            contents=[video_part, prompt],
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=_schema(schema),
                media_resolution=res,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty JSON for video")
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            print(
                f"[gemini] video {path.name}: "
                f"in={getattr(usage, 'prompt_token_count', '?')} "
                f"out={getattr(usage, 'candidates_token_count', '?')}"
            )
        return json.loads(text)
    finally:
        try:
            client().files.delete(name=uploaded.name)
        except Exception:
            pass


def analyze_frames(
    system: str,
    prompt: str,
    frames_b64: list[str],
    schema: dict[str, Any],
    timestamps: list[float] | None = None,
    max_tokens: int = 8192,
    model: str | None = None,
) -> dict[str, Any]:
    """JPEG fallback when the Files API will not take the source."""
    import base64
    from google.genai import types

    parts: list[Any] = [prompt]
    for i, b64 in enumerate(frames_b64):
        if timestamps is not None:
            parts.append(f"[frame @ {timestamps[i]:.1f}s]")
        parts.append(types.Part.from_bytes(
            data=base64.standard_b64decode(b64), mime_type="image/jpeg",
        ))
    resp = client().models.generate_content(
        model=model or config.GEMINI_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=_schema(schema),
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty JSON for frames")
    return json.loads(text)


if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    r = complete_json("Reply with ok=true.", "ping", schema, max_tokens=64)
    assert r.get("ok") is True, r
    print("gemini self-check ok")
