"""Learned taste.

Picks with a written reason may nudge the rubric. One-off rules stay out of
the prompt until they show up twice. The five house rules below are the
creator's actual pick reasons — they always ride in the writer/judge.
"""
from __future__ import annotations

from app import config
from app.evaluation.rubric import SEED_RUBRIC
from app.storage import memory as memory_store

# Distilled from the pilot's written picks. Episode details (food, HUD, this
# clip's captions) never belong here.
HOUSE_RULES: list[str] = [
    "첫 1–2줄에 지금 상황을 선언하거나 화두를 던져라. 예쁜 오프닝보다 전제가 이긴다.",
    "자막은 그 컷의 대사·어긋남·반응만. 원본에 박힌 자막은 제거하거나 그 구간을 잘라라.",
    "같은 컷을 반복하지 마라. 자막만 돌리고 영상이 없으면 실패다. 한 전제 → 진행 → 회수.",
    "타이트한 컷보다 장면이 이어지는 편집이 이긴다. 과정 설명 샷을 세 개 잇지 마라.",
    "웃기게 ≠ 욕 ≠ 밈 체크리스트. 그 장면에서 벌어진 일만. "
    "템플릿(실화냐, 이게 내 인생, 존나 쎄네)은 실패다.",
]

JUDGE_SYSTEM = (
    "You are the xlog quality judge. Score 0-10, then pick. "
    "House taste, in order: (1) first line states the situation or a hook, "
    "(2) captions quote this footage's line/mismatch; source burned-in "
    "subtitles must be gone, (3) no looping the same clips — one thread, "
    "(4) connected scenes beat tighter isolated cuts, (5) funny ≠ swearing "
    "≠ meme checklist. Penalize templates (실화냐, 이게 내 인생, 존나 쎄네) "
    "and unearned swears. If both fail house taste, say so and pick the "
    "less-bad; do not reward the ruder empty one."
)

WRITER_HEAD = (
    "너는 이 크리에이터의 숏폼 편집자다. 아래 다섯 가지가 폼·루브릭보다 앞선다.\n"
    "1) 훅 1–2줄에 지금 상황 또는 화두. 예쁜 첫 프레임보다 전제.\n"
    "2) 캡션은 그 샷의 대사·어긋남·반응(?????, 폭주). 원본 자막은 넣지 마라.\n"
    "3) 같은 컷 반복 금지. 한 전제가 밀리다가 회수. 자막만 살리고 영상 루프는 실패.\n"
    "4) 컷을 잘게 쪼개기보다 장면이 이어지게. 과정 설명 세 샷 연속 금지.\n"
    "5) 웃기게 ≠ 욕. 밈은 이 장면에 이식될 때만. 템플릿 금지: "
    "실화냐, 이게 내 인생, 존나 쎄네, 존나 멸망, 그래서 뭐.\n"
    "From understood moments, write a 30-60s vertical short. "
    "Never reuse the same footage twice.\n"
    "Captions: Korean unless the user wrote otherwise, <= 26 chars, spoken. "
    "No emoji, no title cards. Vary caption_style and fx (at least three fx "
    "per variant; flash or whip on a pivot). Hook and payoff must have captions.\n"
    "욕은 기본 금지. 지워도 농담이 남으면 욕을 지워라. 집단 혐오 금지.\n"
    "concept는 이 영상 전용 농담 한 줄. "
    "Variant A: 캡처해서 보낼 훅. Variant B: 같은 컷, 다른 각도. 더 많은 욕이 아니다."
)


def writer_system() -> str:
    from app.knowledge import shorts_form
    base = WRITER_HEAD + "\n\n" + house_prompt()
    learned = taste_prompt(include_house=False)
    if learned:
        base += (
            "\n\n" + learned
            + "\nLearned extras never override the five house rules. "
            "Ignore episode-specific food, HUD, or this-clip wording."
        )
    base += "\n\n" + shorts_form.form_as_prompt()
    return base


def house_prompt() -> str:
    return "\n".join(
        ["## House taste (this creator's actual pick reasons)"]
        + [f"- {r}" for r in HOUSE_RULES]
    )


def taste_rules() -> list[str]:
    """Confirmed extras that are still on the live rubric. House is separate."""
    from app.evaluation import rubric as rubric_store
    live = {
        (p or "").replace("\n", " ").strip()
        for p in (rubric_store.load_rubric().get("preferences") or [])
    }
    house = {r.replace("\n", " ").strip() for r in HOUSE_RULES}
    out: list[str] = []
    for p in memory_store.list_preferences():
        if int(p.get("times_seen") or 0) < config.TASTE_MIN_SEEN:
            continue
        rule = (p.get("rule") or "").replace("\n", " ").strip()
        if not rule or rule not in live or rule in house:
            continue
        out.append(rule[: config.TASTE_RULE_CHARS])
        if len(out) >= config.TASTE_RULES_IN_PROMPT:
            break
    return out


def taste_prompt(include_house: bool = True) -> str:
    parts: list[str] = []
    if include_house:
        parts.append(house_prompt())
    extra = taste_rules()
    if extra:
        parts.append(
            "\n".join(
                ["## Secondary (ignore if it fights house taste)"]
                + [f"- {r}" for r in extra]
            )
        )
    return "\n\n".join(parts)


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
        "engine": "gpt+grok+gemini",
        "model": config.OPENAI_EDITOR_MODEL,
        "vision_model": config.OPENAI_EDITOR_MODEL,
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
        "rules_in_prompt": HOUSE_RULES + taste_rules(),
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
