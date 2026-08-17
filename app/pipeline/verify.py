"""Hallucination reduction layer.

Three defenses, cheapest first:
  1. validate_variants(): DETERMINISTIC code checks on shot plans — clamp
     time ranges to real video bounds, drop degenerate shots. No LLM.
  2. cross_check_moments(): GPT independently re-examines the same frames
     (Claude fallback if OpenAI is down) and confirms / adjusts / rejects
     each moment. Rejected moments never reach the screenwriter.
  3. second_opinion(): the same verifier judges the rendered variants; the
     verdict records whether both agree (disagreement is surfaced, not hidden).

GPT is a different vendor seeing only frames + claims — a fresh witness,
not the author grading itself. All LLM cross-checks fail open.
"""
from __future__ import annotations

import json
import traceback
from typing import Any

from app import config
from app.llm import grok, openai_client

# ------------------ 1. deterministic plan validation ------------------ #


def _trim_against_used(
    start: float, end: float, used: list[tuple[float, float]]
) -> tuple[float, float]:
    """Push the range past footage already spent, so no clip repeats."""
    for u_start, u_end in sorted(used):
        if end <= u_start or start >= u_end:
            continue
        overlap = min(end, u_end) - max(start, u_start)
        if overlap <= config.MAX_SHOT_OVERLAP_SEC:
            continue
        if start >= u_start:
            start = u_end
        else:
            end = u_start
        if end <= start:
            break
    return start, end


def _extend_to_target(
    shots: list[dict],
    used: dict[int, list[tuple[float, float]]],
    video_infos: list[dict],
    target: float,
) -> float:
    """Grow shots into the unused footage next to them until the edit is long
    enough. A model that plans six 1s shots would otherwise ship a 6s short."""
    total = sum(s["end_sec"] - s["start_sec"] for s in shots)
    for cap in (5.0, 9.0, 15.0):
        for s in shots:
            if total >= target:
                return total
            idx = s["video_index"]
            old = (s["start_sec"], s["end_sec"])
            lo, hi = 0.0, video_infos[idx]["duration_sec"]
            for u_start, u_end in used[idx]:
                if (u_start, u_end) == old:
                    continue
                if u_end <= old[0]:
                    lo = max(lo, u_end)
                if u_start >= old[1]:
                    hi = min(hi, u_start)
            end = max(old[1], min(hi, old[0] + cap, old[1] + (target - total)))
            total += end - old[1]
            start = old[0]
            if total < target:
                start = min(old[0], max(lo, end - cap, old[0] - (target - total)))
                total += old[0] - start
            s["start_sec"], s["end_sec"] = round(start, 2), round(end, 2)
            used[idx].remove(old)
            used[idx].append((s["start_sec"], s["end_sec"]))
    return total


def _fill_from_gaps(
    shots: list[dict],
    used: dict[int, list[tuple[float, float]]],
    video_infos: list[dict],
    target: float,
) -> float:
    """Last resort: cut extra shots out of the footage nobody used yet."""
    total = sum(s["end_sec"] - s["start_sec"] for s in shots)
    if total >= target or not shots:
        return total
    gaps: list[tuple[float, int, float, float]] = []
    for idx, info in enumerate(video_infos):
        cursor = 0.0
        end_marker = (info["duration_sec"], info["duration_sec"])
        for span_start, span_end in sorted(used.get(idx, [])) + [end_marker]:
            if span_start - cursor >= 2.0:
                gaps.append((span_start - cursor, idx, cursor, span_start))
            cursor = max(cursor, span_end)
    gaps.sort(reverse=True)
    for _, idx, start, end in gaps:
        if total >= target:
            break
        take = min(end - start, 5.0, target - total)
        if take < 2.0:
            continue
        shot = dict(shots[-1])
        shot.update({
            "video_index": idx,
            "start_sec": round(start, 2),
            "end_sec": round(start + take, 2),
            "role": "b-roll",
            "reason": "auto-filled: the plan was under the minimum length",
            "caption": "",
            "caption_style": "none",
            "fx": "none",
        })
        shots.append(shot)
        used.setdefault(idx, []).append((shot["start_sec"], shot["end_sec"]))
        total += take
    return total


def validate_variants(variants: list[dict], video_infos: list[dict]) -> list[dict]:
    """Clamp every shot into its source video's real bounds; drop shots that
    become degenerate (<1s), reference a nonexistent video, or replay footage
    an earlier shot already used. Records fixes in variant['validation_warnings']."""
    for v in variants:
        warnings: list[str] = []
        cleaned: list[dict] = []
        used: dict[int, list[tuple[float, float]]] = {}
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
            trimmed = _trim_against_used(start, end, used.get(idx, []))
            if trimmed != (start, end):
                warnings.append(
                    f"shot {i}: repeats earlier footage — trimmed to "
                    f"{trimmed[0]:.1f}-{trimmed[1]:.1f}"
                )
            start, end = trimmed
            if end - start < 1.0:
                warnings.append(f"shot {i}: <1s after clamping — dropped")
                continue
            shot["start_sec"], shot["end_sec"] = start, end
            used.setdefault(idx, []).append((start, end))
            cleaned.append(shot)
        v["shots"] = cleaned
        if cleaned and sum(
            s["end_sec"] - s["start_sec"] for s in cleaned
        ) < config.SHORT_MIN_SEC:
            total = _extend_to_target(cleaned, used, video_infos, config.SHORT_MIN_SEC)
            if total < config.SHORT_MIN_SEC:
                total = _fill_from_gaps(cleaned, used, video_infos, config.SHORT_MIN_SEC)
            warnings.append(
                f"shots too short — stretched to {total:.1f}s "
                f"(target {config.SHORT_MIN_SEC}s)"
            )
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


def verifier_name() -> str:
    if openai_client.available():
        return config.OPENAI_MODEL
    return config.GROK_MODEL


def _complete(
    system: str,
    text: str,
    schema: dict[str, Any],
    images: list[tuple[str, str]] | None = None,
    schema_name: str = "result",
) -> dict[str, Any]:
    """Prefer GPT; if the key/quota is dead, a different Claude still verifies."""
    if openai_client.available():
        try:
            if images:
                content: str | list[dict[str, Any]] = [{"type": "text", "text": text}]
                for tag, b64 in images:
                    content.append({"type": "text", "text": tag})
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
            else:
                content = text
            return openai_client.complete_json(
                system, content, schema, schema_name=schema_name,
            )
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    if images:
        content = [{"type": "text", "text": text}]
        for tag, b64 in images:
            content.append({"type": "text", "text": tag})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
    else:
        content = text
    return grok.complete_json(
        system, content, schema, effort="low", schema_name=schema_name,
    )


def cross_check_moments(
    moments: list[dict],
    frames_b64: list[str],
    timestamps: list[float],
) -> list[dict]:
    """Verify one chunk's moments against the same frames with the verifier
    model. Returns the filtered/corrected moment list. Fails open."""
    if not config.CROSS_CHECK or not moments:
        return moments
    try:
        claims = [
            {"index": i, "start_sec": m["start_sec"], "end_sec": m["end_sec"],
             "description": m["description"]}
            for i, m in enumerate(moments)
        ]
        images = [
            (f"[frame @ {t:.1f}s]", b64)
            for t, b64 in zip(timestamps, frames_b64)
        ]
        result = _complete(
            _VERIFY_SYSTEM,
            "Claimed moments:\n" + json.dumps(claims, ensure_ascii=False)
            + "\n\nFrames follow. Verify each claim.",
            _VERIFY_SCHEMA,
            images=images,
            schema_name="verification",
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


# ------------------ 3. second judging opinion ------------------ #

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
    """Independent verifier verdict on the rendered pixels. Fails open."""
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
        # same dir the judge used: the frames are already on disk, no re-encode
        tmp = (Path(work_dir) if work_dir else config.TMP_DIR) / "judge_frames"
        images: list[tuple[str, str]] = []
        for v in variants:
            label = v.get("label")
            raw = (outputs or {}).get(label)
            if not raw or not Path(raw).exists():
                continue
            images += preprocess.sample_short(Path(raw), label, tmp)
        if not images:
            text += "\n\nCandidate edits:\n" + json.dumps(
                variants, ensure_ascii=False, indent=1
            )
        return _complete(
            system, text, _OPINION_SCHEMA,
            images=images or None,
            schema_name="opinion",
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return None
