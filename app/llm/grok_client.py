"""Grok 4.6 via xAI.

complete_json() matches the Claude/OpenAI wrappers (Chat Completions).
x_search is a server-side tool on the Responses API, so tool calls go
through complete_with_tools() instead of stuffing tools into JSON mode.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from datetime import date, timedelta
from typing import Any

from openai import OpenAI

from app import config

_client: OpenAI | None = None

_FALLBACK = "특별한 트렌드 없음. 기본 감각으로 진행"

_RESEARCH_SYSTEM = (
    "너는 2026년 이번 주 X(트위터)·틱톡 자막을 훔쳐 쓰는 편집 조수다. "
    "목적은 이 영상에 이식할 수 있는 지금 도는 훅 문장과 반응 말투를 가져오는 것이다.\n\n"
    "반드시 x_search로 최근 14일 한국어 포스트를 확인해라. "
    "2019~2023 감성, 죽은 유행, 관련 없는 챌린지는 버려라.\n"
    "- 주제 밈을 억지로 끼우지 말고, 훅 구조만 가져와라. 문장은 이 영상 디테일로 다시 쓴다.\n"
    "- 욕·유행어 복붙 금지. 이 컷에 없는 감정을 트렌드로 넣지 마라.\n\n"
    "출력:\n"
    "- 지금 써도 되는 훅/반응 문장 3개 (이 영상 상황에 맞춰 변형 힌트 포함)\n"
    "- 추천 캡션 톤 (한 줄)\n"
    "- 추천 훅 구조 (반전/자기비하/의문문 중 뭐가 맞는지)\n"
    "- 지금 쓰면 촌스러운 표현 (있으면)"
)

# auto-mode: only spend an x_search when the job is asking for that register.
_TRENDY = re.compile(
    r"재밌|웃기|밈|드립|바이럴|트렌드|유행|센스|리액션|예능|개그|"
    r"funny|meme|viral|trend|joke|humor|comedy|lol",
    re.I,
)

# ~500-600 tokens of mixed KO/EN. Hard cap so the writer context stays bounded.
_RESEARCH_CHARS = 1600


def available() -> bool:
    return bool(config.XAI_API_KEY)


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            max_retries=1,
        )
    return _client


def complete_json(
    system: str,
    user: str | list[dict[str, Any]],
    schema: dict[str, Any],
    max_tokens: int = 8192,
    effort: str | None = None,
    model: str | None = None,
    schema_name: str = "result",
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Schema-constrained JSON. `tools` is accepted for interface parity but
    ignored — JSON mode and x_search cannot share a request; use
    complete_with_tools() for live X search."""
    _ = tools
    chosen = model or config.GROK_MODEL
    kwargs: dict[str, Any] = {
        "model": chosen,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    }
    # API default is high; always send so a cheap stage cannot silently spend.
    if effort:
        kwargs["extra_body"] = {"reasoning_effort": effort}
    resp = client().chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("Grok returned empty content")
    return json.loads(raw)


def complete_with_tools(
    system: str,
    user: str,
    tools: list[dict[str, Any]],
    max_tokens: int = 1024,
    effort: str | None = "low",
    model: str | None = None,
) -> str:
    """Free-text call that can invoke server-side tools (x_search, etc.)."""
    kwargs: dict[str, Any] = {
        "model": model or config.GROK_MODEL,
        "instructions": system,
        "input": [{"role": "user", "content": user}],
        "tools": tools,
        "max_output_tokens": max_tokens,
    }
    if effort:
        kwargs["extra_body"] = {"reasoning_effort": effort}
    resp = client().responses.create(**kwargs)
    text = (getattr(resp, "output_text", None) or "").strip()
    if text:
        return text
    chunks: list[str] = []
    for item in getattr(resp, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            t = getattr(part, "text", None)
            if t:
                chunks.append(t)
    return "\n".join(chunks).strip()


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


def analyze_frames(
    system: str,
    prompt: str,
    frames_b64: list[str],
    schema: dict[str, Any],
    timestamps: list[float] | None = None,
    max_tokens: int = 8192,
    effort: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    content = frames_content(frames_b64, prompt, timestamps)
    return complete_json(
        system, content, schema,
        max_tokens=max_tokens, effort=effort, model=model,
        schema_name="moments",
    )


def should_research(instruction: str, moments_summary: str) -> bool:
    """auto: only when the job is asking for a trendy/funny register."""
    mode = (config.GROK_TRENDS or "auto").strip().lower()
    if mode in ("0", "off", "false", "no"):
        return False
    if mode in ("1", "on", "true", "yes", "always"):
        return True
    blob = f"{instruction}\n{moments_summary}"
    return bool(_TRENDY.search(blob))


def _clip(text: str, max_chars: int = _RESEARCH_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def research_trends(instruction: str, moments_summary: str) -> str:
    """Live X sense for this short. Fail-open: never blocks the writer."""
    if not available():
        return _FALLBACK
    user = (
        f"사용자 요청:\n{(instruction or '').strip() or '(없음)'}\n\n"
        f"영상에서 뽑힌 소재/감정:\n{(moments_summary or '').strip() or '(없음)'}\n\n"
        "위 소재에 이식할 수 있는, 최근 14일 X/숏폼에서 실제로 도는 "
        "훅 문장과 반응 말투의 구조만 찾아라. 문장은 이 영상에 맞게 다시 쓸 "
        "힌트만. 관련 없는 챌린지, 죽은 유행, 욕 복붙은 버려라."
    )
    from_date = (date.today() - timedelta(days=14)).isoformat()
    try:
        text = complete_with_tools(
            _RESEARCH_SYSTEM,
            user,
            tools=[{"type": "x_search", "from_date": from_date}],
            max_tokens=700,
            effort="low",
        )
    except Exception:
        traceback.print_exc()
        return _FALLBACK
    text = _clip(text)
    if not text:
        return _FALLBACK
    print(f"[grok] x_search research {len(text)} chars")
    return text


if __name__ == "__main__":
    assert should_research("재밌는 자막 넣어줘", "") is True
    assert should_research("funny captions", "dog fails") is True
    assert should_research("차분한 정보 전달", "회의 자료 요약") is False
    assert _clip("abcd" * 1000).endswith("…")
    assert len(_clip("가" * 5000)) == _RESEARCH_CHARS
    print("grok_client self-check ok")
    if "live" in sys.argv:
        print(research_trends("육아 브이로그를 웃기게", "아이가 넘어지고 크게 웃는다"))
