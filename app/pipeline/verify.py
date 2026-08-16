"""Hallucination reduction layer.

Three defenses, cheapest first:
  1. validate_variants(): DETERMINISTIC code checks on shot plans — clamp
     time ranges to real video bounds, drop degenerate shots. No LLM.
  2. cross_check_moments(): GPT independently re-examines the same frames
     and confirms / adjusts / rejects each moment Claude extracted.
     Rejected moments are dropped before the screenwriter sees them.
  3. second_opinion(): GPT judges the variants too; the verdict records
     whether both models agree (disagreement is surfaced, not hidden).

All LLM cross-checks fail open: an OpenAI error logs and returns the
input unchanged so turnaround time and availability never suffer.
"""
from __future__ import annotations

import json
import traceback

from app import config
from app.llm import openai_client

# ------------------ 1. deterministic plan validation ------------------ #


def validate_variants(variants: list[dict], video_infos: list[dict]) -> list[dict]:
    """Clamp every shot into its source video's real bounds; drop shots that
    become degenerate (<1s) or reference a nonexistent video. Records what
    was fixed in variant['validation_warnings']."""
    for v in variants:
        warnings: list[str] = []
        cleaned: list[dict] = []
        for i, shot in enumerate(v.get("shots", [])):
            idx = shot.get("video_index", 0)
            if not (0 <= idx < len(video_infos)):
                warnings.append(f"shot {i}: invalid video_index {idx} — dropped")
                continue
            duration = video_infos[idx]["duration_sec"]
            start = max(0.0, min(shot["start_sec"], duration))
            end = max(0.0, min(shot["end_sec"], duration))
            if (start, end) != (shot["start_sec"], shot["end_sec"]):
                warnings.append(
                    f"shot {i}: clamped {shot['start_sec']:.1f}-{shot['end_sec']:.1f}"
                    f" -> {start:.1f}-{end:.1f} (video is {duration:.1f}s)"
                )
            if end - start < 1.0:
                warnings.append(f"shot {i}: <1s after clamping — dropped")
                continue
            shot["start_sec"], shot["end_sec"] = start, end
            cleaned.append(shot)
        v["shots"] = cleaned
        v["total_sec"] = round(sum(s["end_sec"] - s["start_sec"] for s in cleaned), 2)
        if not (config.SHORT_MIN_SEC <= v["total_sec"] <= config.SHORT_MAX_SEC):
            warnings.append(
                f"total {v['total_sec']}s outside {config.SHORT_MIN_SEC}-"
                f"{config.SHORT_MAX_SEC}s target"
            )
        v["validation_warnings"] = warnings
    return variants


# ------------------ 2. GPT moment verification ------------------ #

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": ["confirmed", "adjusted", "rejected"],
                    },
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["index", "verdict", "start_sec", "end_sec", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

_VERIFY_SYSTEM = (
    "You are an independent fact-checker for a video editing agent. Another "
    "model claimed the following 'moments' exist in a video; you see the "
    "same sampled frames (with timestamps). For each claimed moment decide:\n"
    "- confirmed: the frames support the described content at that time range\n"
    "- adjusted: the content exists but at a different range (give the "
    "corrected range, within the frames you can see)\n"
    "- rejected: the frames do NOT support the description (hallucination)\n"
    "Be strict about content, lenient about wording. Echo the given range "
    "for confirmed/rejected."
)


def cross_check_moments(
    moments: list[dict],
    frames_b64: list[str],
    timestamps: list[float],
) -> list[dict]:
    """Verify one chunk's moments against the same frames using GPT.
    Returns the filtered/corrected moment list. Fails open."""
    if not config.CROSS_CHECK or not moments:
        return moments
    try:
        claims = [
            {"index": i, "start_sec": m["start_sec"], "end_sec": m["end_sec"],
             "description": m["description"]}
            for i, m in enumerate(moments)
        ]
        content = openai_client.frames_content(
            frames_b64,
            "Claimed moments:\n" + json.dumps(claims, ensure_ascii=False)
            + "\n\nFrames follow. Verify each claim.",
            timestamps,
        )
        result = openai_client.complete_json(
            _VERIFY_SYSTEM, content, _VERIFY_SCHEMA, schema_name="verification"
        )
        verdicts = {v["index"]: v for v in result["verdicts"]}
        kept: list[dict] = []
        for i, m in enumerate(moments):
            v = verdicts.get(i)
            if v is None or v["verdict"] == "confirmed":
                m["verified"] = True
                kept.append(m)
            elif v["verdict"] == "adjusted":
                m["start_sec"], m["end_sec"] = v["start_sec"], v["end_sec"]
                m["verified"] = True
                m["verify_note"] = v["note"]
                kept.append(m)
            # rejected -> dropped (hallucinated moment never reaches the writer)
        return kept
    except Exception:  # noqa: BLE001 — cross-check must never block the pipeline
        traceback.print_exc()
        return moments


# ------------------ 3. GPT second judging opinion ------------------ #

_OPINION_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["winner", "reasoning"],
    "additionalProperties": False,
}


def second_opinion(
    variants: list[dict],
    analysis_summary: str,
    instruction: str,
    rubric_prompt: str,
    outputs: dict[str, str] | None = None,
    work_dir=None,
) -> dict | None:
    """Independent GPT verdict. Prefers rendered frames when present. Fails open."""
    if not config.CROSS_CHECK:
        return None
    try:
        from pathlib import Path
        from app.pipeline import preprocess

        system = (
            "You are an independent judge of short-form video edits. Watch the "
            "rendered frames when they are provided. Pick the better candidate "
            "by the seed rubric.\n\n"
            + rubric_prompt
        )
        text = (
            (f"The creator's request:\n{instruction}\n\n" if instruction.strip() else "")
            + f"Source footage summary:\n{analysis_summary}\n\n"
            "You are watching RENDERED shorts. Frames are labelled by edit.\n"
            "Plans (reference only):\n"
            + json.dumps(
                [{"label": v.get("label"), "concept": v.get("concept")}
                 for v in variants],
                ensure_ascii=False,
            )
        )
        content = [{"type": "text", "text": text}]
        tmp = Path(work_dir) / "judge_frames_gpt" if work_dir else config.TMP_DIR / "judge_frames_gpt"
        saw_pixels = False
        for v in variants:
            label = v.get("label")
            raw = (outputs or {}).get(label)
            if not raw or not Path(raw).exists():
                continue
            frames = preprocess.extract_frames(
                Path(raw), tmp / label, fps=config.JUDGE_FPS,
            )
            if not frames:
                continue
            step = max(1, len(frames) // config.JUDGE_MAX_FRAMES)
            sampled = frames[::step][: config.JUDGE_MAX_FRAMES]
            for f in sampled:
                content.append({"type": "text", "text": f"[edit {label} @ {f['t']:.1f}s]"})
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + preprocess.frame_to_b64(f["path"])
                    },
                })
                saw_pixels = True
        if not saw_pixels:
            content = text + "\n\nCandidate edits:\n" + json.dumps(
                variants, ensure_ascii=False, indent=1
            )
        return openai_client.complete_json(
            system, content, _OPINION_SCHEMA, schema_name="opinion"
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return None
