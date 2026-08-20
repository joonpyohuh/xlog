"""AI judge: scores the two RENDERED shorts by watching their frames.

Learned taste rides in as a capped, reinforcement-ranked rule block so the
prompt stays a fixed size as evidence accumulates (see taste.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from app.evaluation import rubric as rubric_store, taste
from app.llm import grok
from app.pipeline import preprocess

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "per_criterion": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "criterion": {"type": "string"},
                                "score": {"type": "integer"},
                                "comment": {"type": "string"},
                            },
                            "required": ["criterion", "score", "comment"],
                            "additionalProperties": False,
                        },
                    },
                    "weighted_total": {"type": "number"},
                },
                "required": ["label", "per_criterion", "weighted_total"],
                "additionalProperties": False,
            },
        },
        "winner": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["scores", "winner", "reasoning"],
    "additionalProperties": False,
}


def _pixel_user(
    variants: list[dict],
    analysis_summary: str,
    instruction: str,
    outputs: dict[str, str],
    work_dir: Path | None,
) -> list[dict] | str:
    """Vision user payload: rendered frames first, plans as a short caption."""
    from app import config
    tmp = (work_dir or config.TMP_DIR) / "judge_frames"
    parts: list[dict] = []
    intro = (
        (f"The creator's request for this short:\n{instruction}\n\n"
         if instruction.strip() else "")
        + f"Source footage summary:\n{analysis_summary}\n\n"
        "You are watching the RENDERED 9:16 shorts (pixels), not just the "
        "shot list. The creator's brief is the scoring key: does this short "
        "contain the scenes they would have cut themselves, and are those "
        "scenes punched the way they asked? House taste: situation in the "
        "first line, captions from THIS footage, no clip looping, connected "
        "scenes over denser cuts, funny ≠ swearing. Extra swearing is not a "
        "point. Score what you SEE.\n"
        "Shot-plan metadata (for reference only):\n"
        + json.dumps(
            [
                {"label": v.get("label"), "concept": v.get("concept"),
                 "hook_rationale": v.get("hook_rationale"),
                 "total_sec": v.get("total_sec")}
                for v in variants
            ],
            ensure_ascii=False,
        )
    )
    parts.append({"type": "text", "text": intro})
    for v in variants:
        label = v.get("label")
        raw = (outputs or {}).get(label)
        if not raw or not Path(raw).exists():
            continue
        for tag, b64 in preprocess.sample_short(Path(raw), label, tmp):
            parts.append({"type": "text", "text": tag})
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    if len(parts) == 1:
        return intro + "\n\n(no rendered files — fall back to plans)\n" + json.dumps(
            variants, ensure_ascii=False, indent=1
        )
    return parts


def judge_variants(
    variants: list[dict],
    analysis_summary: str,
    instruction: str = "",
    outputs: dict[str, str] | None = None,
    work_dir: Path | None = None,
) -> dict:
    """Compare the rendered shorts against trained taste + seed criteria."""
    from app import config

    system = (
        taste.JUDGE_SYSTEM
        + " Judge the actual rendered video. Captions, crop, pacing, and "
        "dead frames you see outweigh the written shot plan."
    )
    learned = taste.taste_prompt()
    system += "\n\n" + rubric_store.rubric_as_prompt() + ("\n\n" + learned if learned else "")
    user = _pixel_user(
        variants, analysis_summary, instruction, outputs or {}, work_dir,
    )
    return grok.complete_json(
        system, user, _VERDICT_SCHEMA,
        max_tokens=8192, effort=config.JUDGE_EFFORT, schema_name="verdict",
    )
