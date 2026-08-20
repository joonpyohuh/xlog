"""Shot-plan generation (requirements 2, 4, 7 + user instruction).

Takes the extracted moments plus the user's free-form instruction (e.g.
"재밌는 자막을 넣고 눈에 확 들어오게 만들어줘") and produces TWO different
30-60s shot plans (variant A / B), each following the mainstream shorts
form grammar and the current learned rubric. Shots may carry burned-in
captions.
"""
from __future__ import annotations

import json
import re
import traceback

from app import config
from app.evaluation import taste
from app.llm import grok, openai_client
from app.llm.grok_client import research_trends, should_research
from app.pipeline import captions as captions_mod
from app.pipeline import quality as quality_mod

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},           # "A" / "B"
                    "concept": {"type": "string"},          # one-line editorial angle
                    "hook_rationale": {"type": "string"},
                    "shots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "video_index": {"type": "integer"},
                                "start_sec": {"type": "number"},
                                "end_sec": {"type": "number"},
                                "role": {"type": "string"},  # hook/setup/development/payoff/ending
                                "reason": {"type": "string"},
                                # burned-in caption ("" = none)
                                "caption": {"type": "string"},
                                "caption_style": {
                                    "type": "string",
                                    "enum": list(captions_mod.CAPTION_STYLES),
                                },
                                "fx": {
                                    "type": "string",
                                    "enum": list(captions_mod.SHOT_FX),
                                },
                            },
                            "required": [
                                "video_index", "start_sec", "end_sec",
                                "role", "reason", "caption", "caption_style", "fx",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["label", "concept", "hook_rationale", "shots"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["variants"],
    "additionalProperties": False,
}

_FUNNY_ASK = re.compile(
    r"웃기|재밌|드립|개그|유머|말맛|funny|joke|humor|comedy", re.I,
)
_SWEAR_ASK = re.compile(r"욕|비속어|쌍욕|swear|profan", re.I)

_MEME_BLOCK = (
    "### 이 크리에이터의 고르는 이유 (체크리스트보다 앞)\n"
    "1) 훅 1–2줄 = 지금 상황 또는 화두.\n"
    "2) 캡션 = 그 샷의 대사·어긋남·반응. 원본 박힌 자막 금지.\n"
    "3) 같은 컷 반복 금지. 한 전제 → 진행 → 회수.\n"
    "4) 장면이 이어지게. 과정 설명 세 샷 연속 금지.\n"
    "5) 웃기게 ≠ 욕. 템플릿 금지: 실화냐, 이게 내 인생, 존나 쎄네.\n"
    "페이오프는 훅을 회수한다. 새 주제로 새지 마라.\n\n"
)

_FUNNY_BLOCK = (
    "### THIS JOB: 웃기게 = 이 영상에서만 되는 농담\n"
    "문법 체크리스트를 채우지 마라. 친구에게 캡처해서 보낼 한 줄이 아니면 다시 써라.\n"
    "B는 A보다 욕이 많은 버전이 아니다. 같은 컷의 다른 각도 "
    "(더 구체적인 디테일, 또는 한 박자 늦은 인지).\n\n"
)

_SWEAR_BLOCK = (
    "### THIS JOB: 사용자가 욕을 요청함\n"
    "훅 또는 페이오프 중 1줄만. 그 비트에 굴욕·실패·충격이 화면에서 보여야 한다.\n"
    "존나/미친 정도. 씨발은 그 장면에서 사람이 진짜 욕할 때만. "
    "매 줄 금지. 지워도 농담이 남으면 지워라. 검열·별표·ㅋㅋ 치환 금지.\n\n"
)


def wants_funny(instruction: str) -> bool:
    return bool(_FUNNY_ASK.search(instruction or ""))


def wants_swears(instruction: str) -> bool:
    return bool(_SWEAR_ASK.search(instruction or ""))


def _voice_block(instruction: str) -> str:
    parts = [_MEME_BLOCK]
    if wants_funny(instruction):
        parts.append(_FUNNY_BLOCK)
    if wants_swears(instruction):
        parts.append(_SWEAR_BLOCK)
    return "".join(parts)


_TREND_BLOCK = (
    "### 지금 도는 말투 (x_search, 최근 14일 — 추측 금지)\n"
    "{research_result}\n\n"
    "이 블록이 검색 결과다. 학습된 옛 밈으로 대체하지 마라.\n"
    "- 인용된 반응 문장을 이 영상 고유명사·동작·대사에 이식해서 caption에 실제로 써라.\n"
    "- 한 베리언트에 최소 3개 캡션이 위 인용의 변형이어야 한다.\n"
    "- concept는 가져온 밈/말투 중 이 영상에 맞는 것을 밝혀라.\n"
    "- 관련 없는 챌린지·죽은 유행은 버려라. 검색에 없는 욕을 만들지 마라.\n\n"
)


def moments_blurb(analysis: dict) -> str:
    """Compact moment list for trend search — not the full JSON dump."""
    lines: list[str] = []
    for v in analysis.get("videos") or []:
        s = (v.get("summary") or "").strip()
        if s:
            lines.append(s)
    for m in (analysis.get("moments") or [])[:12]:
        desc = (m.get("description") or "").strip()
        if not desc:
            continue
        mood = (m.get("mood") or "").strip()
        lines.append(f"- {desc}" + (f" ({mood})" if mood else ""))
    return "\n".join(lines)[:1200]


def will_research(instruction: str, analysis: dict) -> bool:
    """Search is independent of who writes the shot list — it only needs Grok."""
    if not grok.available():
        return False
    return should_research(instruction, moments_blurb(analysis))


def research_text(research: object) -> str:
    """Only a pack that actually searched may reach the writer."""
    if isinstance(research, dict):
        if not research.get("searched"):
            return ""
        return (research.get("text") or "").strip()
    text = (research or "").strip() if isinstance(research, str) else ""
    if not text or text.startswith("특별한 트렌드 없음"):
        return ""
    return text


def _complete_plans(system: str, user: str, spec: dict) -> dict:
    use_gpt = spec.get("writer") == "gpt" and openai_client.available()
    if config.USE_GROK_FOR_WRITER:
        use_gpt = False
    if use_gpt:
        try:
            print("[screenwriter] writer=GPT", config.OPENAI_EDITOR_MODEL)
            return openai_client.complete_json(
                system, user, _PLAN_SCHEMA,
                schema_name="plans", max_tokens=16000,
                model=config.OPENAI_EDITOR_MODEL,
                effort=config.OPENAI_EDITOR_EFFORT,
            )
        except Exception:
            traceback.print_exc()
            print("[screenwriter] GPT writer failed, Grok fallback")
    print("[screenwriter] writer=Grok", config.GROK_MODEL)
    return grok.complete_json(
        system, user, _PLAN_SCHEMA,
        max_tokens=16000, effort=config.WRITER_EFFORT, schema_name="plans",
    )


def write_plans(
    analysis: dict, instruction: str = "", research: str | dict = "",
    quality: str = "fast",
) -> list[dict]:
    """Return VARIANTS_PER_JOB shot plans built from the analyzed moments."""
    moments_json = json.dumps(analysis["moments"], ensure_ascii=False, indent=1)
    system = taste.writer_system()
    instruction_block = (
        f"User's request for this short (follow it faithfully):\n{instruction}\n\n"
        if instruction.strip()
        else "The user gave no specific request — still write a joke that only this footage supports, not a safe recap.\n\n"
    )
    instruction_block += _voice_block(instruction)
    if not research_text(research) and will_research(instruction, analysis):
        research = research_trends(instruction, moments_blurb(analysis))
    live = research_text(research)
    trend_block = _TREND_BLOCK.format(research_result=live) if live else ""
    must_note = (analysis.get("must_note") or "").strip()
    user = (
        instruction_block
        + trend_block
        + (must_note + "\n\n" if must_note else "")
        + f"Available moments (from {len(analysis['videos'])} source video(s)):\n"
        f"{moments_json}\n\n"
        f"Design exactly {config.VARIANTS_PER_JOB} DIFFERENT shot plans "
        f"(labels 'A', 'B', ...). Each must:\n"
        f"- total {config.SHORT_MIN_SEC}-{config.SHORT_MAX_SEC} seconds "
        "(sum of shot durations, excluding the branding outro)\n"
        "- open with a hook shot, follow the form structure\n"
        "- build shots around the given moments: a moment marks the peak, so "
        "pad it with a few seconds of lead-in/lead-out (staying inside the "
        "source video) instead of cutting exactly on its boundaries\n"
        "- Prefer moments with high brief_fit: those ARE the reason this "
        "short exists. A pretty filler shot that ignores the brief is a miss.\n"
        "- Treat the shot list as a story, not a montage: each shot must "
        "cause the next (setup → turn → payoff). Drop a high-intensity "
        "moment if it does not serve that arc.\n"
        "- NEVER reuse the same footage twice: every shot's time range must be "
        "disjoint from every other shot in that variant. Repeating a moment is "
        "the worst defect a plan can have.\n"
        "- cover the whole story, not one cluster: draw shots from the early, "
        "middle and late parts of the source so the short summarizes it\n"
        "- differ in joke angle, not rudeness: A is the screenshot hook, "
        "B is a different specific joke on the same footage\n"
        "- every caption names something in THAT shot (what they did, said, "
        "or got wrong) — if a moment quotes a spoken line, the caption must "
        "use those words or twist them, never replace them with 실화냐/존나 쎄네\n"
        + (
            "- live X lines are in the prompt: mutate at least 3 captions "
            "from those quotes onto this footage\n"
            if live else
            "- no live X pack: do not invent 2026 memes from memory\n"
        )
        + "- run 8-14 shots of 2.5-5s each; a 1s shot is too short to read\n"
        "- vary caption_style shot to shot and spend the fx palette: at least "
        "three different fx values per variant (punch_in/zoom_in/zoom_out/"
        "shake/flash/whip), with a flash or whip on a structural pivot"
    )
    spec = quality_mod.resolve(quality)
    result = _complete_plans(system, user, spec)
    variants = result["variants"][: config.VARIANTS_PER_JOB]
    for v in variants:
        for s in v.get("shots") or []:
            if s.get("fx") not in captions_mod.SHOT_FX:
                s["fx"] = "none"
            if s.get("caption_style") not in captions_mod.CAPTION_STYLES:
                s["caption_style"] = "normal"
        v["total_sec"] = round(
            sum(s["end_sec"] - s["start_sec"] for s in v["shots"]), 2
        )
    return variants


if __name__ == "__main__":
    fake = {
        "videos": [{"summary": "아이가 공원에서 넘어진다"}],
        "moments": [{"description": "넘어지고 크게 웃는다", "mood": "폭소"}],
    }
    blurb = moments_blurb(fake)
    assert "넘어지고" in blurb and "폭소" in blurb, blurb
    block = _TREND_BLOCK.format(research_result="인용: 아니 이게 된다고\n- https://x.com/a/status/1")
    assert "x_search" in block and "최소 3개" in block
    assert research_text("") == ""
    assert research_text("특별한 트렌드 없음. 기본 감각으로 진행") == ""
    assert research_text({"searched": False, "text": "밈"}) == ""
    assert "아니 이게" in research_text({"searched": True, "text": "인용: 아니 이게 된다고"})
    assert wants_funny("자막 웃기게") and wants_funny("funny captions")
    assert not wants_funny("차분한 정보 전달")
    assert wants_swears("욕 섞어줘")
    swear_block = _voice_block("욕 섞고 웃기게")
    assert "1줄" in swear_block
    funny_only = _voice_block("재밌는 자막")
    assert "씨발" not in funny_only and "시발" not in funny_only
    assert "더 독하게" not in funny_only
    assert "이 영상에서만" in funny_only
    assert "화두" in _voice_block("차분한 정보")
    assert "텍스트 레이어" not in _voice_block("차분한 정보")
    print("screenwriter trend-block self-check ok")
