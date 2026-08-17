"""Learned taste.

Picks with a written reason may nudge the rubric. One-off rules stay out of
the prompt until they show up twice. GPT is the independent verifier.
"""
from __future__ import annotations

from app import config
from app.evaluation.rubric import SEED_RUBRIC
from app.storage import memory as memory_store

JUDGE_SYSTEM = (
    "You are the xlog quality judge for 2026 X/TikTok natives. "
    "Pick the short a group chat would screenshot because the joke is "
    "specific to THIS footage — not because it is ruder. "
    "Score each candidate 0-10 on the rubric. "
    "Penalize textbook labels, cheerleading (오늘도 화이팅), captions that "
    "would work on any other video (이게 내 인생, 실화냐, 존나 멸망, 그래서 뭐), "
    "and swearing that is not earned by a visible fail, shock, or self-roast. "
    "Do not reward extra profanity. A specific funny line beats a safer cut; "
    "a swear with no joke does not."
)

WRITER_HEAD = (
    "너는 2026년 X/틱톡 네이티브 숏폼 편집자다. 목표는 무난한 요약이 아니라 "
    "이 영상에서만 성립하는 농담으로 스크롤을 멈추는 것이다.\n"
    "샷 구조보다 캡션·훅·컨셉 한 줄이 중요하다. 컨셉을 잡을 때 먼저 물어라: "
    "이 컷의 어긋남(의도 vs 결과, 말 vs 행동)이 한 줄로 되는가? 아니면 다시 써라.\n"
    "From understood moments, write a 30-60s vertical short. One thread: "
    "hook names the mismatch, middle escalates that same joke, payoff pays "
    "it back or betrays it. Never reuse the same footage twice.\n"
    "Captions: Korean unless the user wrote otherwise, <= 26 chars, spoken. "
    "No emoji, no title cards. Vary caption_style and fx (at least three fx "
    "per variant; flash or whip on a pivot). Hook and payoff must have captions.\n\n"
    "## 농담 (폼·루브릭보다 우선)\n"
    "매 캡션은 그 샷에 있는 고유명사·동작·대사를 집어라. "
    "아무 영상에나 붙는 문장은 실패다.\n"
    "절대 하지 말 것 (한 줄이라도 있으면 그 샷은 실패):\n"
    "- 템플릿: 이게 내 인생, 실화냐, 존나 멸망, 그래서 뭐, 왜 하필\n"
    "- 교과서 설명: 공원에서 넘어진다, 아이가 웃고 있다\n"
    "- 무난 응원: 오늘도 화이팅, 행복한 하루, 귀여운 순간, 오늘은 ~했습니다\n"
    "- 장면에 굴욕·실패·충격이 없는데 넣는 욕\n"
    "욕은 기본 금지. 웃기게 ≠ 욕. 지워도 농담이 남으면 욕을 지워라. "
    "집단 혐오 슬로건은 금지.\n"
    "concept는 다큐 각도가 아니라 이 영상 전용 농담 한 줄. "
    "Variant A: 캡처해서 보낼 훅. Variant B: 같은 컷, 다른 각도 "
    "(더 구체적인 디테일 또는 더 늦은 인지). 더 많은 욕이 아니다."
)


def writer_system() -> str:
    from app.knowledge import shorts_form
    base = WRITER_HEAD
    learned = taste_prompt()
    if learned:
        base += (
            "\n\n" + learned
            + "\nHouse rules are cut and pacing only. They never license "
            "generic captions or unearned swearing."
        )
    base += "\n\n" + shorts_form.form_as_prompt()
    return base


def taste_rules() -> list[str]:
    """Rules that survived confirmation: seen twice+, most-reinforced first."""
    out: list[str] = []
    for p in memory_store.list_preferences():
        if int(p.get("times_seen") or 0) < config.TASTE_MIN_SEEN:
            continue
        rule = (p.get("rule") or "").replace("\n", " ").strip()
        if not rule:
            continue
        out.append(rule[: config.TASTE_RULE_CHARS])
        if len(out) >= config.TASTE_RULES_IN_PROMPT:
            break
    return out


def taste_prompt() -> str:
    rules = taste_rules()
    if not rules:
        return ""
    return "\n".join(
        ["## House style learned from this creator's picks (highest confidence first)"]
        + [f"- {r}" for r in rules]
    )


def status() -> dict:
    """What the creator's picks have actually changed, for the UI panel."""
    from app.evaluation import rubric as rubric_store
    from app.llm import openai_client
    from app.pipeline import verify

    rubric = rubric_store.load_rubric()
    feedback = memory_store.list_feedback()
    agreed = sum(1 for f in feedback if f.get("agreement"))
    seed = {c["name"]: c["weight"] for c in SEED_RUBRIC["criteria"]}
    gpt_live = openai_client.available()
    return {
        "engine": "gemini+grok+gpt",
        "model": config.GROK_MODEL,
        "vision_model": config.GEMINI_MODEL,
        "judge_model": config.GROK_MODEL,
        "fast_model": config.GEMINI_MODEL,
        "verify_model": verify.verifier_name(),
        "gpt_live": gpt_live,
        "rubric_version": rubric.get("version", 1),
        "rubric_notes": rubric.get("notes", ""),
        "criteria": [
            {
                "name": c["name"],
                "weight": c["weight"],
                "delta": c["weight"] - seed.get(c["name"], c["weight"]),
            }
            for c in rubric.get("criteria", [])
        ],
        "picks": len(feedback),
        "agreed": agreed,
        "references": len(memory_store.list_references()),
        "rules_total": len(memory_store.list_preferences()),
        "rules_in_prompt": taste_rules(),
        "recent": [_pick_summary(f) for f in feedback[-3:][::-1]],
    }


def _pick_summary(ev: dict) -> dict:
    variants = ev.get("variants") or []
    choice = ev.get("user_choice")
    chosen = next((v for v in variants if v.get("label") == choice), {})
    rejected = next((v for v in variants if v.get("label") != choice), {})
    return {
        "ts": ev.get("ts"),
        "choice": choice,
        "comment": ev.get("user_comment") or "",
        "agreement": bool(ev.get("agreement")),
        "chosen": (chosen.get("concept") or "")[:180],
        "rejected": (rejected.get("concept") or "")[:180],
        "rubric_version": ev.get("rubric_version"),
    }
