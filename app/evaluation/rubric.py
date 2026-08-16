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
            "description": "The first 3 seconds must make the viewer stop scrolling.",
        },
        {
            "name": "mainstream_convention",
            "weight": 20,
            "description": "Follows the most universally accepted editing style of "
                           "actual popular Shorts/Reels: proven cut rhythm, caption "
                           "placement, and structure. Safe and broadly appealing "
                           "beats experimental or niche choices.",
        },
        {
            "name": "instruction_fit",
            "weight": 20,
            "description": "Faithfully realizes the user's free-form request "
                           "(caption tone, effects, mood). Ignore this criterion "
                           "(redistribute judgment) when no request was given.",
        },
        {
            "name": "pacing",
            "weight": 15,
            "description": "Rhythmic cuts, no dead air, energy rises toward the payoff.",
        },
        {
            "name": "narrative_clarity",
            "weight": 10,
            "description": "One clear thread from hook to payoff; no confusing jumps.",
        },
        {
            "name": "moment_selection",
            "weight": 10,
            "description": "The genuinely best raw moments were used, not filler.",
        },
    ],
    "preferences": [
        # Learned preferences accumulate here from creator evaluations,
        # e.g. "prefers cuts on motion", "dislikes shots longer than 5s".
    ],
    "notes": "Seed rubric. Baseline = the most universal, safe mainstream "
             "shorts editing style, as seen in actual popular shorts. Refined "
             "continuously from the pilot creator's evaluations.",
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
    """Fixed-size seed only. Learned taste is fine-tuned, not injected."""
    r = SEED_RUBRIC
    lines = ["## Editing rubric (seed baseline — learned taste is fine-tuned)"]
    for c in r["criteria"]:
        lines.append(f"- {c['name']} (weight {c['weight']}): {c['description']}")
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
