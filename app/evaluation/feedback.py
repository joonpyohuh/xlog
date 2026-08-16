"""Rubric learning loop (requirement 8).

When the pilot user picks a variant (and optionally explains why), the LLM
revises the rubric: adjusting weights, refining criterion wording, and
appending concrete learned preferences. Disagreement between the AI judge
and the user is highlighted to the LLM as the strongest learning signal.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from app import config
from app.evaluation import rubric as rubric_store
from app.llm import claude
from app.storage import memory as memory_store

FEEDBACK_LOG: Path = config.RUBRIC_DIR / "feedback_log.jsonl"


def record_feedback(
    job_id: str,
    variants: list[dict],
    judge_verdict: dict | None,
    user_choice: str,
    user_comment: str,
) -> dict:
    """Persist the evaluation event and let the LLM update the rubric.
    Returns the new rubric."""
    event = {
        "ts": int(time.time()),
        "job_id": job_id,
        "user_choice": user_choice,
        "user_comment": user_comment,
        "judge_winner": (judge_verdict or {}).get("winner"),
        "agreement": (judge_verdict or {}).get("winner") == user_choice,
    }
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    current = rubric_store.load_rubric()
    try:
        new_rubric = _update_rubric(variants, judge_verdict, user_choice, user_comment)
    except Exception:  # noqa: BLE001 — pick must stick even if Claude is down
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


def _update_rubric(
    variants: list[dict],
    judge_verdict: dict | None,
    user_choice: str,
    user_comment: str,
) -> dict:
    current = rubric_store.load_rubric()
    disagreement = (
        judge_verdict is not None and judge_verdict.get("winner") != user_choice
    )
    system = (
        "You maintain the editing rubric for xlog, the pilot creator's "
        "personal shorts tool. The creator's judgment is ground truth — the "
        "local app exists to learn THEIR criteria. Given the current rubric, "
        "the two candidate edits, the AI judge's verdict, and the user's "
        "actual choice + comment, produce a revised rubric:\n"
        "- adjust criterion weights or wording only when the evidence "
        "supports it (one data point should nudge, not overhaul)\n"
        "- distill any concrete, reusable preference from the user's comment "
        "into the `preferences` list (deduplicate against existing entries)\n"
        "- increment `version`; summarize what changed and why in `notes`\n"
        + (
            "- IMPORTANT: the AI judge disagreed with the user. Analyze what "
            "the rubric is missing or over-weighting that caused the judge "
            "to prefer the other edit.\n"
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
            "user_comment": user_comment,
        },
        ensure_ascii=False,
        indent=1,
    )
    new_rubric = claude.complete_json(
        system, user, rubric_store.RUBRIC_SCHEMA, max_tokens=8192
    )
    new_rubric["version"] = current["version"] + 1  # enforce monotonic versioning
    rubric_store.save_rubric(new_rubric, source="feedback")
    return new_rubric
