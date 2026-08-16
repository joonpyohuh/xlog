"""Reference-style learning from YouTube links.

The local app is the pilot creator's personal criteria-building tool: give
it a YouTube (Shorts) link whose editing you admire and it will
  1. download the video (yt-dlp),
  2. sample frames densely and have Claude analyze the *editing* — pacing,
     caption usage, hook construction, structure,
  3. distill concrete reusable rules and merge them into the rubric's
     `preferences` (versioned, like every rubric change).

Every learned reference is logged to data/rubric/references.jsonl.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app import config
from app.evaluation import rubric as rubric_store
from app.llm import claude
from app.pipeline import ingest, preprocess
from app.storage import cleanup
from app.storage import memory as memory_store

REFERENCES_LOG: Path = config.RUBRIC_DIR / "references.jsonl"


class ReferenceError(ValueError):
    pass


def _download(url: str) -> Path:
    try:
        return ingest.download_youtube(url, config.REFERENCE_DIR)
    except ingest.IngestError as e:
        raise ReferenceError(str(e)) from e


_STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "style_summary": {"type": "string"},
        "hook_technique": {"type": "string"},
        "pacing": {"type": "string"},
        "caption_style": {"type": "string"},
        "structure": {"type": "string"},
        "learned_rules": {
            "type": "array",
            "items": {"type": "string"},   # concrete, reusable editing rules
        },
    },
    "required": [
        "style_summary", "hook_technique", "pacing",
        "caption_style", "structure", "learned_rules",
    ],
    "additionalProperties": False,
}

_STYLE_SYSTEM = (
    "You are an editing analyst for xlog. You are shown frames sampled at "
    "1fps from a published short-form video the pilot creator admires. "
    "Reverse-engineer its EDITING: how the hook is built, cut rhythm "
    "(estimate shot lengths from frame-to-frame changes), caption/text "
    "usage and placement, and overall structure. Then distill 3-7 concrete, "
    "reusable editing rules ('learned_rules') that xlog should imitate — "
    "each a single imperative sentence, specific enough to act on."
)


def learn_from_youtube(url: str, notes: str = "") -> dict:
    """Download, analyze, and fold the reference's style into the rubric.
    Returns {"reference": ..., "rubric_version": ...}."""
    video = None
    frames_dir = None
    try:
        video = _download(url)
        info = ingest.probe(video)
        if info["duration_sec"] > config.REFERENCE_MAX_SEC:
            raise ReferenceError(
                f"reference is {info['duration_sec']:.0f}s — only short-form "
                f"references (<= {config.REFERENCE_MAX_SEC}s) are supported"
            )

        frames_dir = config.TMP_DIR / f"ref_frames_{video.stem}"
        frames = preprocess.extract_frames(
            video, frames_dir, fps=config.REFERENCE_ANALYSIS_FPS,
        )
        # cap frames to one request
        step = max(1, len(frames) // config.MAX_FRAMES_PER_REQUEST)
        sampled = frames[::step][: config.MAX_FRAMES_PER_REQUEST]
        style = claude.analyze_frames(
            system=_STYLE_SYSTEM,
            prompt=(
                f"Reference video ({info['duration_sec']:.0f}s). "
                + (f"Creator's note on why they like it: {notes}\n" if notes else "")
                + "Analyze the editing style."
            ),
            frames_b64=[preprocess.frame_to_b64(f["path"]) for f in sampled],
            timestamps=[f["t"] for f in sampled],
            schema=_STYLE_SCHEMA,
        )

        new_rubric = _merge_into_rubric(url, notes, style)

        record = {
            "ts": int(time.time()),
            "url": url,
            "notes": notes,
            "file": "",
            "style": style,
            "rubric_version": new_rubric["version"],
        }
        with REFERENCES_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        memory_store.record_reference(
            url=url,
            notes=notes,
            file_path="",
            style=style,
            rubric_version=new_rubric["version"],
            ts=record["ts"],
        )
        return {"reference": record, "rubric_version": new_rubric["version"]}
    finally:
        cleanup.after_reference(video, frames_dir)


def _merge_into_rubric(url: str, notes: str, style: dict) -> dict:
    current = rubric_store.load_rubric()
    new_rubric = claude.complete_json(
        system=(
            "You maintain the editing rubric for xlog, the pilot creator's "
            "personal shorts tool. The creator provided a reference video "
            "whose editing style they want xlog to learn. Merge the analyzed "
            "style into the rubric:\n"
            "- fold the learned_rules into `preferences` (deduplicate; keep "
            "each rule one imperative sentence)\n"
            "- adjust criterion descriptions only if the reference clearly "
            "contradicts them\n"
            "- increment `version`; note the reference URL and what was "
            "learned in `notes`"
        ),
        user=json.dumps(
            {
                "current_rubric": current,
                "reference_url": url,
                "creator_notes": notes,
                "analyzed_style": style,
            },
            ensure_ascii=False,
            indent=1,
        ),
        schema=rubric_store.RUBRIC_SCHEMA,
    )
    new_rubric["version"] = current["version"] + 1
    rubric_store.save_rubric(new_rubric, source="reference")
    return new_rubric


def list_references() -> list[dict]:
    stored = memory_store.list_references()
    if stored:
        return stored
    if not REFERENCES_LOG.exists():
        return []
    return [
        json.loads(line)
        for line in REFERENCES_LOG.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
