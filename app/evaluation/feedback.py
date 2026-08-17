"""Rubric learning loop (requirement 8).

Picks are always logged. The rubric only moves when the creator wrote why —
a silent A/B is too small a sample to rewrite house style. One comment may
nudge weights and add at most one reusable rule; episode-specific wording
is dropped.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from app import config
from app.evaluation import rubric as rubric_store
from app.llm import gemini
from app.storage import memory as memory_store

FEEDBACK_LOG: Path = config.RUBRIC_DIR / "feedback_log.jsonl"
_MAX_NEW_PREFS = 1
_PREF_MIN_CHARS = 24
_PREF_MAX_CHARS = 140


def record_feedback(
    job_id: str,
    variants: list[dict],
    judge_verdict: dict | None,
    user_choice: str,
    user_comment: str,
) -> dict:
    """Persist the evaluation event and maybe update the rubric."""
    current = rubric_store.load_rubric()
    if _already_learned(job_id):
        return current

    event = {
        "ts": int(time.time()),
        "job_id": job_id,
        "user_choice": user_choice,
        "user_comment": user_comment,
        "judge_winner": (judge_verdict or {}).get("winner"),
        "agreement": (judge_verdict or {}).get("winner") == user_choice,
    }
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    new_rubric = current
    try:
        new_rubric = _update_rubric(variants, judge_verdict, user_choice, user_comment)
    except Exception:  # noqa: BLE001 — pick must stick even if the LLM is down
        traceback.print_exc()
        new_rubric = current
    try:
        memory_store.record_feedback(
            job_id=job_id,
            user_choice=user_choice,
            user_comment=user_comment,
            judge_winner=(judge_verdict or {}).get("winner"),
            agreement=event["agreement"],
            variants=variants,
            judge_verdict=judge_verdict,
            rubric_version=new_rubric.get("version"),
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    return new_rubric


def _already_learned(job_id: str) -> bool:
    return any(ev.get("job_id") == job_id for ev in memory_store.list_feedback())


def _update_rubric(
    variants: list[dict],
    judge_verdict: dict | None,
    user_choice: str,
    user_comment: str,
) -> dict:
    current = rubric_store.load_rubric()
    comment = (user_comment or "").strip()
    if not comment:
        return current

    disagreement = (
        judge_verdict is not None and judge_verdict.get("winner") != user_choice
    )
    system = (
        "You maintain the editing rubric for xlog. The creator's comment is "
        "the only teaching signal — ignore details that only apply to this "
        "one video (foods, names, this clip's captions, bike vs walking).\n"
        "- keep the same criterion names; do not add or drop criteria\n"
        "- copy criterion descriptions unchanged\n"
        "- if the judge disagreed, nudge at most two weights by "
        f"±{config.TASTE_WEIGHT_NUDGE}; if they agreed, keep weights\n"
        "- add at most ONE new preference, copied from the comment, reusable "
        "on a different video next week; otherwise return the existing list\n"
        "- increment `version`; put a one-line reason in `notes`"
        + (
            "\n- the judge disagreed: say what the rubric over-weighted."
            if disagreement
            else ""
        )
    )
    user = json.dumps(
        {
            "current_rubric": current,
            "candidate_edits": variants,
            "judge_verdict": judge_verdict,
            "user_choice": user_choice,
            "user_comment": comment,
        },
        ensure_ascii=False,
        indent=1,
    )
    proposed = gemini.complete_json(
        system, user, rubric_store.RUBRIC_SCHEMA, max_tokens=8192,
    )
    new_rubric = regularize(current, proposed, variants, disagreement)
    restated = list(new_rubric.pop("_restated", []) or [])
    rubric_store.save_rubric(new_rubric, source="feedback")
    if restated:
        memory_store.reinforce_preferences(restated)
    return new_rubric


def regularize(
    current: dict,
    proposed: dict,
    variants: list[dict],
    disagreement: bool,
) -> dict:
    """One pick may nudge weights and add one general rule. Nothing else."""
    nudge = config.TASTE_WEIGHT_NUDGE if disagreement else 0
    proposed_w = {
        c.get("name"): c.get("weight")
        for c in (proposed.get("criteria") or [])
        if isinstance(c, dict)
    }
    criteria = []
    for c in current.get("criteria") or []:
        w = int(c.get("weight") or 0)
        raw = proposed_w.get(c["name"])
        if nudge and isinstance(raw, int):
            w = max(w - nudge, min(w + nudge, raw))
        criteria.append({
            "name": c["name"],
            "weight": w,
            "description": c.get("description") or "",
        })
    old_prefs = [p for p in (current.get("preferences") or []) if str(p).strip()]
    added: list[str] = []
    restated: list[str] = []
    for rule in proposed.get("preferences") or []:
        match = _existing_match(rule, old_prefs)
        if match:
            if match not in restated:
                restated.append(match)
            continue
        if len(added) >= _MAX_NEW_PREFS:
            continue
        if _reusable_pref(rule, old_prefs + added, variants):
            added.append(rule.strip())
    return {
        "version": int(current.get("version") or 1) + 1,
        "owner": current.get("owner") or proposed.get("owner") or "",
        "criteria": criteria,
        "preferences": old_prefs + added,
        "notes": (proposed.get("notes") or "nudge from comment")[:300],
        "_restated": restated,
    }


def _existing_match(rule: str, existing: list[str]) -> str | None:
    text = (rule or "").strip()
    if not text:
        return None
    for e in existing:
        e = (e or "").strip()
        if not e:
            continue
        if text == e or text in e or e in text:
            return e
    return None


def _reusable_pref(rule: str, existing: list[str], variants: list[dict]) -> bool:
    text = (rule or "").strip()
    if not (_PREF_MIN_CHARS <= len(text) <= _PREF_MAX_CHARS):
        return False
    for e in existing:
        e = (e or "").strip()
        if not e:
            continue
        if text == e or text in e or e in text:
            return False
    for cap in _variant_captions(variants):
        if cap and cap in text:
            return False
    return True


def _variant_captions(variants: list[dict]) -> list[str]:
    out: list[str] = []
    for v in variants or []:
        for key in ("concept", "hook_rationale"):
            val = (v.get(key) or "").strip()
            if len(val) >= 6:
                out.append(val)
        for s in v.get("shots") or []:
            cap = (s.get("caption") or "").strip()
            if len(cap) >= 4:
                out.append(cap)
    return out


if __name__ == "__main__":
    current = {
        "version": 3,
        "owner": "check",
        "criteria": [
            {"name": "hook_strength", "weight": 25, "description": "stop the scroll"},
            {"name": "pacing", "weight": 15, "description": "no dead air"},
        ],
        "preferences": ["Open cold on the highest-action moment."],
        "notes": "",
    }
    variants = [{"label": "A", "concept": "마늘 통닭으로 회복", "shots": [
        {"caption": "38.5도, 아무것도 못 먹는 중"},
    ]}]
    proposed = {
        "version": 99,
        "owner": "rewritten",
        "criteria": [
            {"name": "hook_strength", "weight": 40, "description": "essay about garlic"},
            {"name": "pacing", "weight": 5, "description": "changed"},
            {"name": "caption_voice", "weight": 8, "description": "new"},
        ],
        "preferences": [
            "Open cold on the highest-action moment.",
            "When captions carry comedy, keep visual variety — do not loop the same two clips.",
            "Prioritize bike versus walking in the opening frames like 38.5도, 아무것도 못 먹는 중",
        ],
        "notes": "overhaul",
    }
    silent = regularize(current, proposed, variants, disagreement=False)
    assert silent.pop("_restated") == ["Open cold on the highest-action moment."]
    assert silent["version"] == 4, silent
    assert silent["criteria"][0]["weight"] == 25, silent
    assert silent["criteria"][0]["description"] == "stop the scroll", silent
    assert len(silent["criteria"]) == 2, silent
    assert silent["preferences"][-1].startswith("When captions"), silent
    assert all("38.5" not in p for p in silent["preferences"]), silent

    nudged = regularize(current, proposed, variants, disagreement=True)
    nudged.pop("_restated", None)
    assert nudged["criteria"][0]["weight"] == 27, nudged
    assert nudged["criteria"][1]["weight"] == 13, nudged
    print("feedback regularize self-check ok")
