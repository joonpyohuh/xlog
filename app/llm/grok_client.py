"""Grok 4.6 via xAI.

complete_json() matches the OpenAI wrapper (Chat Completions).
x_search is a server-side tool on the Responses API, so live X research
goes through complete_with_tools() — JSON mode cannot hold that tool.
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

_RESEARCH_DAYS = 14
_RESEARCH_CHARS = 2400
_X_HOST = re.compile(r"https?://(?:www\.)?(?:x\.com|twitter\.com)/", re.I)

_RESEARCH_SYSTEM = (
    "너는 x_search만으로 최근 한국어 X 반응을 가져오는 조수다. "
    "학습된 옛 밈으로 답하지 마라. 도구를 안 쓰면 실패다.\n"
    f"검색 범위: 오늘부터 {_RESEARCH_DAYS}일 안. 한국어 포스트 우선.\n"
    "같은 감정·상황을 여러 쿼리로 찾아라 "
    "(상황 키워드, 감정 키워드, 올해 숏폼 훅 말투).\n"
    "출력은 검색에서 나온 것만:\n"
    "1) 최근 반응/훅 문장 4~6개 — 실제 포스트에서 인용, 누가/무슨 맥락인지 한 조각\n"
    "2) 이 영상에 이식 가능한 올해 밈·포맷 이름 (없으면 생략)\n"
    "3) 지금 도는 말투 한 줄\n"
    "4) 각 인용을 이 영상 고유명사·동작에 어떻게 심을지 한 줄\n"
    "5) 출처 x.com URL\n"
    "관련 없는 챌린지, 2019~2024 감, 검색에 없는 문장 창작은 버려라."
)


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
    """Schema-constrained JSON. `tools` is ignored — JSON mode cannot run
    x_search. Use complete_with_tools() for live X search."""
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
    max_tokens: int = 2048,
    effort: str | None = "medium",
    model: str | None = None,
    tool_choice: str | dict[str, Any] = "required",
) -> dict[str, Any]:
    """Responses API with server-side tools. Returns text + whether search ran."""
    kwargs: dict[str, Any] = {
        "model": model or config.GROK_MODEL,
        "instructions": system,
        "input": [{"role": "user", "content": user}],
        "tools": tools,
        "max_output_tokens": max_tokens,
        "tool_choice": tool_choice,
    }
    if effort:
        kwargs["extra_body"] = {"reasoning_effort": effort}
    try:
        resp = client().responses.create(**kwargs)
    except Exception:
        if tool_choice == "required":
            kwargs["tool_choice"] = "auto"
            resp = client().responses.create(**kwargs)
        else:
            raise
    return parse_search_response(resp)


def parse_search_response(resp: Any) -> dict[str, Any]:
    """Pull output text, x.com citations, and whether x_search actually ran."""
    text = (getattr(resp, "output_text", None) or "").strip()
    citations: list[str] = []
    types: list[str] = []

    for raw in _as_list(getattr(resp, "citations", None)):
        url = _url_of(raw)
        if url:
            citations.append(url)

    for item in _as_list(getattr(resp, "output", None)):
        itype = str(_pick(item, "type") or "")
        name = str(_pick(item, "name") or "")
        if itype:
            types.append(itype)
        if name:
            types.append(name)
        for part in _as_list(_pick(item, "content")):
            if not text:
                t = _pick(part, "text")
                if t:
                    text = str(t).strip()
            for ann in _as_list(_pick(part, "annotations")):
                url = _url_of(ann)
                if url:
                    citations.append(url)

    usage = getattr(resp, "usage", None)
    sources_used = _pick(usage, "num_sources_used") or 0
    try:
        sources_used = int(sources_used)
    except (TypeError, ValueError):
        sources_used = 0

    citations = _dedupe_urls(citations)
    searched = bool(
        sources_used
        or any(_X_HOST.search(u) for u in citations)
        or any("x_search" in t or t.endswith("_search_call") for t in types)
    )
    if not text:
        chunks: list[str] = []
        for item in _as_list(getattr(resp, "output", None)):
            for part in _as_list(_pick(item, "content")):
                t = _pick(part, "text")
                if t:
                    chunks.append(str(t))
        text = "\n".join(chunks).strip()
    return {
        "text": text,
        "citations": citations,
        "searched": searched,
        "types": types,
        "sources_used": sources_used,
    }


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


def should_research(
    instruction: str, moments_summary: str, mode: str | None = None,
) -> bool:
    """on = every job. auto = only funny/meme briefs. off = never."""
    chosen = (mode or config.GROK_TRENDS or "on").strip().lower()
    if chosen in ("0", "off", "false", "no"):
        return False
    if chosen in ("1", "on", "true", "yes", "always"):
        return True
    blob = f"{instruction}\n{moments_summary}"
    return bool(re.search(
        r"재밌|웃기|밈|드립|바이럴|트렌드|유행|센스|리액션|예능|개그|"
        r"funny|meme|viral|trend|joke|humor|comedy|lol",
        blob, re.I,
    ))


def research_trends(instruction: str, moments_summary: str) -> dict[str, Any]:
    """Live X pack for this short. searched=False → writer must not fake memes."""
    empty = {"text": "", "citations": [], "searched": False}
    if not available():
        print("[grok] x_search skipped: no XAI_API_KEY")
        return empty
    today = date.today()
    from_date = (today - timedelta(days=_RESEARCH_DAYS)).isoformat()
    to_date = today.isoformat()
    user = (
        f"오늘 날짜: {to_date}. 검색은 {from_date} ~ {to_date}만.\n\n"
        f"사용자 요청:\n{(instruction or '').strip() or '(없음)'}\n\n"
        f"영상 모먼트/감정:\n{(moments_summary or '').strip() or '(없음)'}\n\n"
        "x_search로 위 상황·감정에 맞는 최근 한국어 반응, 밈, 말투를 찾아라. "
        "학습 기억으로 메우지 마라. 인용 + x.com 출처가 없으면 실패한 것이다."
    )
    tools = [{"type": "x_search", "from_date": from_date, "to_date": to_date}]
    try:
        pack = complete_with_tools(
            _RESEARCH_SYSTEM, user, tools,
            max_tokens=2048, effort="medium", tool_choice="required",
        )
        if not pack.get("searched"):
            print("[grok] x_search produced no citations, retrying")
            pack = complete_with_tools(
                _RESEARCH_SYSTEM,
                user + "\n\n이전 답은 검색 없이 나왔다. x_search를 반드시 호출하고 "
                "x.com 링크를 남겨라.",
                tools,
                max_tokens=2048, effort="high", tool_choice="required",
            )
    except Exception:
        traceback.print_exc()
        print("[grok] x_search failed open")
        return empty

    text = _clip(pack.get("text") or "")
    citations = [u for u in pack.get("citations") or [] if _X_HOST.search(u)]
    searched = bool(pack.get("searched") and text and citations)
    if not searched:
        # Tool ran but returned no X URLs — still usable if the model quoted posts.
        searched = bool(pack.get("searched") and text)
    if not searched:
        print("[grok] x_search unused; writer will not claim live memes")
        return empty
    brief = format_research(text, citations)
    print(f"[grok] x_search {len(citations)} citations, {len(brief)} chars")
    return {"text": brief, "citations": citations, "searched": True}


def format_research(text: str, citations: list[str]) -> str:
    body = _clip(text)
    if citations:
        src = "\n".join(f"- {u}" for u in citations[:8])
        body = f"{body}\n\n출처 (최근 {_RESEARCH_DAYS}일 X):\n{src}"
    return body


def _clip(text: str, max_chars: int = _RESEARCH_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _as_list(val: Any) -> list[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _pick(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _url_of(obj: Any) -> str:
    if isinstance(obj, str) and obj.startswith("http"):
        return obj
    url = _pick(obj, "url") or _pick(obj, "uri")
    return str(url) if url else ""


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


if __name__ == "__main__":
    assert should_research("재밌는 자막 넣어줘", "", mode="auto") is True
    assert should_research("funny captions", "dog fails", mode="auto") is True
    assert should_research("차분한 정보 전달", "회의 자료 요약", mode="auto") is False
    assert should_research("차분한 정보 전달", "회의 자료 요약", mode="on") is True
    assert should_research("재밌게", "", mode="off") is False
    fake = type("R", (), {
        "output_text": "요즘 반응: 아니 이게 된다고",
        "citations": ["https://x.com/foo/status/1", "https://example.com/nope"],
        "output": [type("I", (), {"type": "x_search_call", "name": "x_search", "content": []})()],
        "usage": type("U", (), {"num_sources_used": 4})(),
    })()
    parsed = parse_search_response(fake)
    assert parsed["searched"] is True, parsed
    assert parsed["citations"][0].startswith("https://x.com/"), parsed
    packed = format_research(parsed["text"], parsed["citations"])
    assert "x.com/foo" in packed and "아니 이게 된다고" in packed
    assert _clip("abcd" * 1000).endswith("…")
    print("grok_client self-check ok")
    if "live" in sys.argv:
        pack = research_trends("육아 브이로그를 웃기게", "아이가 넘어지고 크게 웃는다")
        sys.stdout.buffer.write(
            (json.dumps(pack, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
        )
