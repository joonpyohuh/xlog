"""AI judge: scores the two RENDERED shorts by watching their frames.

Learned taste lives in the fine-tuned (DPO) model, not a growing prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.evaluation import finetune
from app.llm import claude
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


def _sample_short(path: Path, label: str, tmp: Path) -> tuple[list[str], list[str]]:
    """Frames + text labels for one rendered short."""
    from app import config
    frames = preprocess.extract_frames(
        path, tmp / f"judge_{label}", fps=config.JUDGE_FPS,
    )
    if not frames:
        return [], []
    step = max(1, len(frames) // config.JUDGE_MAX_FRAMES)
    sampled = frames[::step][: config.JUDGE_MAX_FRAMES]
    b64s = [preprocess.frame_to_b64(f["path"]) for f in sampled]
    tags = [f"[edit {label} @ {f['t']:.1f}s]" for f in sampled]
    return b64s, tags


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
        "shot list. Frames from each edit follow, labelled with edit letter "
        "and timestamp. Score what you SEE: hook in the first seconds, "
        "caption color/placement, punch-ins, dead air, clarity of payoff.\n"
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
        b64s, tags = _sample_short(Path(raw), label, tmp)
        for tag, b64 in zip(tags, b64s):
            parts.append({"type": "text", "text": tag})
            parts.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
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
        finetune.JUDGE_SYSTEM
        + " Judge the actual rendered video. Captions, crop, pacing, and "
        "dead frames you see outweigh the written shot plan."
    )
    if finetune.active_model() is None:
        extra = finetune.seed_criteria_prompt()
        taste = finetune.fallback_taste_prompt()
        system = system + "\n\n" + extra + (("\n\n" + taste) if taste else "")
    user = _pixel_user(
        variants, analysis_summary, instruction, outputs or {}, work_dir,
    )
    return claude.complete_json(
        system, user, _VERDICT_SCHEMA, max_tokens=8192, effort=config.JUDGE_EFFORT
    )
