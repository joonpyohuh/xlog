"""Shorts "form" knowledge (requirement 4).

The structural grammar of Instagram Reels / YouTube Shorts, distilled by an
LLM into an editable JSON document. A static seed ships with the app; the
LLM can regenerate/refine it (`refresh_form()`), and the screenwriter injects
it into every shot-plan prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

from app import config
from app.llm import gemini
from app.storage import memory as memory_store

FORM_PATH: Path = config.RUBRIC_DIR / "shorts_form.json"

# LLM-distilled baseline of the dominant shorts form as of 2026.
SEED_FORM: dict = {
    "version": 1,
    "source": "seed (LLM-distilled conventions of IG Reels / YT Shorts)",
    "structure": [
        {
            "name": "hook",
            "window_sec": [0, 3],
            "rule": "Open with the single most arresting moment or claim. "
                    "No intros, no logos, no context-setting. The first frame "
                    "must already be interesting.",
        },
        {
            "name": "setup",
            "window_sec": [3, 10],
            "rule": "Minimal context so the payoff lands. One idea only.",
        },
        {
            "name": "development",
            "window_sec": [10, 45],
            "rule": "Escalate: each cut adds new information or raises "
                    "tension. Cut on action; keep average shot length 1.5-4s.",
        },
        {
            "name": "payoff",
            "window_sec": [45, 58],
            "rule": "Deliver the promised moment. This is what the hook sold.",
        },
        {
            "name": "ending",
            "window_sec": [58, 60],
            "rule": "End abruptly right after the payoff (loop-friendly), "
                    "then the branding card follows.",
        },
    ],
    "global_rules": [
        "Vertical 9:16, subject centered in the safe area (UI overlays cover edges).",
        "Never let more than ~3 seconds pass without visual change.",
        "Keep the original audio of chosen moments; it carries authenticity.",
        "One narrative thread per short. Discard good moments that don't serve it.",
        "Prefer moments with faces, motion, or emotional peaks over scenery.",
        "The video should make sense with the sound off.",
    ],
}


def load_form() -> dict:
    stored = memory_store.load_form()
    if stored is not None:
        return stored
    if FORM_PATH.exists():
        return json.loads(FORM_PATH.read_text(encoding="utf-8"))
    save_form(SEED_FORM)
    return SEED_FORM


def save_form(form: dict) -> None:
    FORM_PATH.write_text(json.dumps(form, ensure_ascii=False, indent=2), encoding="utf-8")
    memory_store.save_form(form)


def form_as_prompt() -> str:
    """Render the form document for injection into screenwriter prompts."""
    form = load_form()
    lines = [
        "## Shot timing rails only — do not use this to make captions polite or generic"
    ]
    for sec in form["structure"]:
        lo, hi = sec["window_sec"]
        lines.append(f"- {sec['name']} ({lo}-{hi}s): {sec['rule']}")
    lines.append("## Global rules")
    lines += [f"- {r}" for r in form["global_rules"]]
    return "\n".join(lines)


_FORM_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "integer"},
        "source": {"type": "string"},
        "structure": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "window_sec": {"type": "array", "items": {"type": "number"}},
                    "rule": {"type": "string"},
                },
                "required": ["name", "window_sec", "rule"],
                "additionalProperties": False,
            },
        },
        "global_rules": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["version", "source", "structure", "global_rules"],
    "additionalProperties": False,
}


def refresh_form(notes: str = "") -> dict:
    """Ask Claude to regenerate the form document from its knowledge of
    current IG Reels / YT Shorts conventions, optionally guided by `notes`."""
    current = load_form()
    result = gemini.complete_json(
        system=(
            "You are a short-form video editor who deeply understands the "
            "current grammar of Instagram Reels and YouTube Shorts: hooks, "
            "pacing, retention editing, loopability. Produce a revised form "
            "document as JSON matching the given schema."
        ),
        user=(
            "Current form document:\n"
            f"{json.dumps(current, ensure_ascii=False, indent=2)}\n\n"
            f"Editor notes (may be empty):\n{notes}\n\n"
            "Return an improved version. Increment `version` and set `source` "
            "to a one-line description of this revision."
        ),
        schema=_FORM_SCHEMA,
        model=config.GEMINI_MODEL,
    )
    save_form(result)
    return result
