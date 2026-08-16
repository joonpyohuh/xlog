"""AI judge (requirement 7): scores the two variants.

Learned taste lives in the fine-tuned model (and a capped fallback),
not in a growing prompt.
"""
from __future__ import annotations

import json
import traceback

from app.evaluation import finetune
from app.llm import claude, openai_client

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
                                "score": {"type": "integer"},  # 0-10
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


def judge_variants(
    variants: list[dict], analysis_summary: str, instruction: str = ""
) -> dict:
    """Compare shot plans against trained taste + a fixed seed rubric."""
    from app import config

    system = finetune.JUDGE_SYSTEM
    model = finetune.active_model()
    if model is None:
        extra = finetune.seed_criteria_prompt()
        taste = finetune.fallback_taste_prompt()
        system = system + "\n\n" + extra + (("\n\n" + taste) if taste else "")
    user = (
        (f"The creator's request for this short:\n{instruction}\n\n"
         if instruction.strip() else "")
        + f"Source footage summary:\n{analysis_summary}\n\n"
        "Candidate edits:\n"
        f"{json.dumps(variants, ensure_ascii=False, indent=1)}"
    )
    if model is not None:
        try:
            return openai_client.complete_json(
                system, user, _VERDICT_SCHEMA, schema_name="verdict",
                max_tokens=8192, model=model,
            )
        except Exception:  # noqa: BLE001 — fall back to Claude
            traceback.print_exc()
    return claude.complete_json(
        system, user, _VERDICT_SCHEMA, max_tokens=8192, effort=config.JUDGE_EFFORT
    )
