"""Shot-plan generation (requirements 2, 4, 7 + user instruction).

Takes the extracted moments plus the user's free-form instruction (e.g.
"재밌는 자막을 넣고 눈에 확 들어오게 만들어줘") and produces TWO different
30-60s shot plans (variant A / B), each following the mainstream shorts
form grammar and the current learned rubric. Shots may carry burned-in
captions.
"""
from __future__ import annotations

import json
import traceback

from app import config
from app.evaluation import finetune
from app.llm import claude, openai_client

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},           # "A" / "B"
                    "concept": {"type": "string"},          # one-line editorial angle
                    "hook_rationale": {"type": "string"},
                    "shots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "video_index": {"type": "integer"},
                                "start_sec": {"type": "number"},
                                "end_sec": {"type": "number"},
                                "role": {"type": "string"},  # hook/setup/development/payoff/ending
                                "reason": {"type": "string"},
                                # burned-in caption ("" = none)
                                "caption": {"type": "string"},
                                "caption_style": {
                                    "type": "string",
                                    "enum": [
                                        "none", "normal", "emphasis", "pop",
                                        "neon", "hot", "mint", "gold",
                                        "plate", "box", "impact",
                                    ],
                                },
                                "fx": {
                                    "type": "string",
                                    "enum": ["none", "punch_in", "zoom_in"],
                                },
                            },
                            "required": [
                                "video_index", "start_sec", "end_sec",
                                "role", "reason", "caption", "caption_style", "fx",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["label", "concept", "hook_rationale", "shots"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["variants"],
    "additionalProperties": False,
}


def write_plans(analysis: dict, instruction: str = "") -> list[dict]:
    """Return VARIANTS_PER_JOB shot plans built from the analyzed moments."""
    moments_json = json.dumps(analysis["moments"], ensure_ascii=False, indent=1)
    system = finetune.writer_system()
    model = finetune.active_model()
    if model is None:
        taste = finetune.fallback_taste_prompt()
        if taste:
            system = system + "\n\n" + taste
    instruction_block = (
        f"User's request for this short (follow it faithfully):\n{instruction}\n\n"
        if instruction.strip()
        else "The user gave no specific request — use mainstream defaults.\n\n"
    )
    user = (
        instruction_block
        + f"Available moments (from {len(analysis['videos'])} source video(s)):\n"
        f"{moments_json}\n\n"
        f"Design exactly {config.VARIANTS_PER_JOB} DIFFERENT shot plans "
        f"(labels 'A', 'B', ...). Each must:\n"
        f"- total {config.SHORT_MIN_SEC}-{config.SHORT_MAX_SEC} seconds "
        "(sum of shot durations, excluding the branding outro)\n"
        "- open with a hook shot, follow the form structure\n"
        "- use only time ranges inside the given moments (you may trim a "
        "moment tighter, never extend beyond it)\n"
        "- differ meaningfully from each other in concept, pacing, or "
        "moment selection so the editor has a real choice\n"
        "- keep every shot at least 1.0s long\n"
        "- vary caption_style (not all 'normal') and use fx punch_in/zoom_in "
        "on at least one hook or payoff shot per variant"
    )
    result = None
    if model is not None:
        try:
            result = openai_client.complete_json(
                system, user, _PLAN_SCHEMA, schema_name="plans",
                max_tokens=16000, model=model,
            )
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            result = None
    if result is None:
        result = claude.complete_json(
            system, user, _PLAN_SCHEMA, max_tokens=16000, effort=config.WRITER_EFFORT
        )
    variants = result["variants"][: config.VARIANTS_PER_JOB]
    for v in variants:
        for s in v.get("shots") or []:
            s.setdefault("fx", "none")
            if s.get("caption_style") not in (
                "none", "normal", "emphasis", "pop", "neon", "hot",
                "mint", "gold", "plate", "box", "impact",
            ):
                s["caption_style"] = "normal"
        v["total_sec"] = round(
            sum(s["end_sec"] - s["start_sec"] for s in v["shots"]), 2
        )
    return variants
