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
from app.llm import grok
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
    "### 이 영상의 농담을 찾아라\n"
    "concept / hook_rationale / 매 샷 caption이 이 숏의 전부다. "
    "컷이 좋아도 캡션이 설명문·템플릿이면 실패다.\n"
    "훅부터 써라: 이 컷의 어긋남(의도 vs 결과, 말 vs 행동)을 고유명사·동작·대사로. "
    "그 줄이 이 영상이 아니면 성립하지 않아야 한다.\n"
    "페이오프는 훅을 회수하거나 배신한다. 새 주제로 새지 마라.\n"
    "금지: 이게 내 인생, 실화냐, 존나 멸망, 그래서 뭐, 오늘도 화이팅, "
    "귀여운 순간, 오늘은 ~했다, 장면과 무관한 욕.\n"
    "욕은 기본 금지. 웃기게 ≠ 욕.\n\n"
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
    "### 지금 도는 말투 (live X search)\n"
    "{research_result}\n\n"
    "구조만 빌려라. 문장은 이 영상의 고유명사·동작·대사로 다시 써라. "
    "트렌드 문장·욕을 복붙하지 마라. 죽은 유행은 버려라.\n\n"
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
    if not config.USE_GROK_FOR_WRITER or not grok.available():
        return False
    return should_research(instruction, moments_blurb(analysis))


def _complete_plans(system: str, user: str, spec: dict) -> dict:
    if spec.get("writer") == "claude" and config.ANTHROPIC_API_KEY:
        try:
            from app.llm import claude
            print("[screenwriter] writer=Claude", config.CLAUDE_MODEL)
            return claude.complete_json(
                system, user, _PLAN_SCHEMA,
                max_tokens=16000, effort="high", model=config.CLAUDE_MODEL,
            )
        except Exception:
            traceback.print_exc()
            print("[screenwriter] Claude writer failed, Grok fallback")
    return grok.complete_json(
        system, user, _PLAN_SCHEMA,
        max_tokens=16000, effort=config.WRITER_EFFORT, schema_name="plans",
    )


def write_plans(
    analysis: dict, instruction: str = "", research: str = "",
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
    if not research and will_research(instruction, analysis):
        research = research_trends(instruction, moments_blurb(analysis))
    trend_block = _TREND_BLOCK.format(research_result=research) if research else ""
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
        "or got wrong) — never a label and never a template that fits another "
        "video; hook and payoff must pay each other back\n"
        "- run 8-14 shots of 2.5-5s each; a 1s shot is too short to read\n"
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
    block = _TREND_BLOCK.format(research_result="특별한 트렌드 없음. 기본 감각으로 진행")
    assert "live X search" in block
    assert "복붙" in block and "고유명사" in block
    assert wants_funny("자막 웃기게") and wants_funny("funny captions")
    assert not wants_funny("차분한 정보 전달")
    assert wants_swears("욕 섞어줘")
    swear_block = _voice_block("욕 섞고 웃기게")
    assert "1줄" in swear_block
    funny_only = _voice_block("재밌는 자막")
    assert "씨발" not in funny_only and "시발" not in funny_only
    assert "더 독하게" not in funny_only
    assert "이 영상에서만" in funny_only
    assert "어긋남" in _voice_block("차분한 정보")
    assert "텍스트 레이어" not in _voice_block("차분한 정보")
    print("screenwriter trend-block self-check ok")
