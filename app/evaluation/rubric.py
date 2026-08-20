"""The editing rubric (requirements 7, 8).

The rubric encodes what a "good edit" means. It starts from a seed written
for the pilot user and evolves: every time the user picks a variant and
leaves feedback, the LLM rewrites the rubric (see feedback.py). All
versions are kept so learning is auditable and reversible.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app import config
from app.storage import memory as memory_store

RUBRIC_PATH: Path = config.RUBRIC_DIR / "rubric.json"
HISTORY_DIR: Path = config.RUBRIC_DIR / "history"

SEED_RUBRIC: dict = {
    "version": 1,
    "owner": "xlog baseline — mainstream shorts conventions",
    "criteria": [
        {
            "name": "hook_strength",
            "weight": 25,
            "description": "First 1–2 caption lines declare the situation or throw "
                           "a hook. A prettier 0s frame loses to a clear premise.",
        },
        {
            "name": "mainstream_convention",
            "weight": 18,
            "description": "Reaction captions on this footage. Source burned-in "
                           "subtitles must be removed or cut around — cropped "
                           "source text is a hard fail.",
        },
        {
            "name": "instruction_fit",
            "weight": 18,
            "description": "Faithfully realizes the user's free-form request "
                           "(caption tone, effects, mood). Ignore this criterion "
                           "(redistribute judgment) when no request was given.",
        },
        {
            "name": "pacing",
            "weight": 14,
            "description": "Connected scenes beat tighter isolated cuts. Never "
                           "three process/explanation shots in a row.",
        },
        {
            "name": "narrative_clarity",
            "weight": 15,
            "description": "One premise, stated early, paid back at the end. "
                           "Looping the same clips while captions carry the story "
                           "is a fail. One thread only.",
        },
        {
            "name": "moment_selection",
            "weight": 10,
            "description": "Best raw moments, enough visual variety to support "
                           "the caption story. Do not repeat footage.",
        },
    ],
    "preferences": [],  # house rules live in taste.HOUSE_RULES; extras accrue here
    "notes": "Seed: situation first, this-footage captions, no loops, "
             "connected scenes, funny is not swearing.",
}


def load_rubric() -> dict:
    stored = memory_store.load_rubric()
    if stored is not None:
        return stored
    if RUBRIC_PATH.exists():
        return json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    save_rubric(SEED_RUBRIC, source="seed")
    return SEED_RUBRIC


def save_rubric(rubric: dict, source: str = "unknown") -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    RUBRIC_PATH.write_text(
        json.dumps(rubric, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    snap = HISTORY_DIR / f"rubric_v{rubric['version']}_{int(time.time())}.json"
    snap.write_text(json.dumps(rubric, ensure_ascii=False, indent=2), encoding="utf-8")
    memory_store.save_rubric(rubric, source=source)


def rubric_as_prompt() -> str:
    """Weights may move; wording stays on the seed so one pick cannot rewrite
    what 'hook' means. Extra criteria the learner invented stay out."""
    r = load_rubric()
    seed_desc = {c["name"]: c["description"] for c in SEED_RUBRIC["criteria"]}
    live = {c["name"]: c for c in r.get("criteria") or []}
    lines = [f"## Editing rubric (learned, v{r.get('version', 1)})"]
    for c in SEED_RUBRIC["criteria"]:
        weight = (live.get(c["name"]) or c).get("weight", c["weight"])
        lines.append(f"- {c['name']} (weight {weight}): {seed_desc[c['name']]}")
    return "\n".join(lines)


RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "integer"},
        "owner": {"type": "string"},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "weight": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["name", "weight", "description"],
                "additionalProperties": False,
            },
        },
        "preferences": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["version", "owner", "criteria", "preferences", "notes"],
    "additionalProperties": False,
}
