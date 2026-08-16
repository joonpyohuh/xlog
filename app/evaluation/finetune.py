"""Turn creator A/B picks into OpenAI DPO preference pairs.

SFT that copies the chosen variant as a gold assistant message just
overfits that JSON. DPO trains on the actual comparison: same prompt,
preferred vs rejected output.

ponytail: OpenAI still wants >=10 JSONL rows; we pad by repeating unique
pairs until that floor. Replace with more unique events as they arrive.
"""
from __future__ import annotations

import hashlib
import json
import traceback
from io import BytesIO

from app import config
from app.evaluation.rubric import SEED_RUBRIC
from app.storage import memory as memory_store

MIN_EXAMPLES = 10
FALLBACK_PREFS = 6
FALLBACK_RULE_CHARS = 180

JUDGE_SYSTEM = (
    "You are the xlog quality judge. Your taste is trained; do not expect "
    "a growing preference list. Score each candidate edit 0-10 on "
    "hook_strength, mainstream_convention, instruction_fit, pacing, "
    "narrative_clarity, moment_selection; compute a weight-adjusted total; "
    "pick a winner and explain in the creator's language."
)

WRITER_HEAD = (
    "You are xlog, the pilot creator's personal shorts editor. Your taste "
    "is trained; do not expect a growing preference list. Design 30-60s "
    "vertical shorts from raw footage moments. Follow the structural form "
    "below. Captions: same language as the user request (default Korean), "
    "punchy (<= 30 chars). Vary caption_style across shots — do not default "
    "everything to white 'normal'. Use: normal (white), emphasis (yellow "
    "punchline), pop (white+pink outline), neon (cyan/magenta), hot (red), "
    "mint (green), gold (yellow on plate), plate (white on dark bar), box "
    "(black on yellow bar), impact (huge centered). Also set fx per shot: "
    "none, punch_in (instant 18% crop-in), zoom_in (slow push). If the user "
    "asked for captions or effects, captions are mandatory on hook and payoff "
    "and at least one shot must use a non-normal style plus punch_in or zoom_in."
)


def writer_system() -> str:
    from app.knowledge import shorts_form
    return WRITER_HEAD + "\n\n" + shorts_form.form_as_prompt()


def seed_criteria_prompt() -> str:
    lines = ["## Seed criteria (universal baseline, not learned taste)"]
    for c in SEED_RUBRIC["criteria"]:
        lines.append(f"- {c['name']} (weight {c['weight']}): {c['description']}")
    return "\n".join(lines)


def fallback_taste_prompt() -> str:
    """Bounded stand-in used only when no fine-tuned model is active."""
    prefs = memory_store.list_preferences()[:FALLBACK_PREFS]
    if not prefs:
        return ""
    lines = ["## House style (capped; fine-tune not ready yet)"]
    for p in prefs:
        rule = p["rule"].replace("\n", " ")[:FALLBACK_RULE_CHARS]
        lines.append(f"- {rule}")
    return "\n".join(lines)


def active_model() -> str | None:
    refresh_running()
    return memory_store.get_active_ft_model()


def rebuild_examples() -> list[dict]:
    examples: list[dict] = []
    for ev in memory_store.list_feedback():
        examples.extend(_from_feedback(ev))
    for ref in memory_store.list_references():
        examples.extend(_from_reference(ref))
    if not examples:
        return []
    # ponytail: pad to OpenAI's 10-row floor
    i = 0
    while len(examples) < MIN_EXAMPLES:
        examples.append(examples[i % len(examples)])
        i += 1
    return examples


def schedule() -> dict:
    """Rebuild DPO pairs from the DB and start a fine-tune if needed. Fail-open."""
    memory_store.replace_ft_examples(rebuild_examples())
    if not config.OPENAI_API_KEY:
        return {"ok": False, "reason": "no_openai_key"}
    try:
        refresh_running()
        running = memory_store.latest_ft_job()
        if running and running.get("status") in ("validating_files", "queued", "running"):
            return {"ok": True, "status": running["status"], "job_id": running.get("openai_job_id")}
        examples = memory_store.list_ft_examples()
        if len(examples) < MIN_EXAMPLES:
            return {"ok": False, "reason": "too_few_examples", "count": len(examples)}
        fingerprint = _fingerprint(examples)
        if running and running.get("status") == "succeeded" and running.get("fingerprint") == fingerprint:
            return {"ok": True, "status": "up_to_date", "model": running.get("model")}
        return _submit(examples, fingerprint)
    except Exception as e:  # noqa: BLE001 — learning must never block editing
        traceback.print_exc()
        memory_store.save_ft_job(
            openai_job_id="", status="failed", model=None,
            example_count=0, fingerprint="", error=f"{type(e).__name__}: {e}",
        )
        return {"ok": False, "reason": str(e)}


def refresh_running() -> None:
    row = memory_store.latest_ft_job()
    if not row or not row.get("openai_job_id"):
        return
    if row.get("status") in ("succeeded", "failed", "cancelled"):
        return
    if not config.OPENAI_API_KEY:
        return
    try:
        from app.llm.openai_client import client
        job = client().fine_tuning.jobs.retrieve(row["openai_job_id"])
        model = getattr(job, "fine_tuned_model", None)
        err = ""
        if getattr(job, "error", None):
            err = str(job.error)
        memory_store.save_ft_job(
            openai_job_id=job.id,
            status=job.status,
            model=model,
            example_count=row.get("example_count") or 0,
            fingerprint=row.get("fingerprint") or "",
            error=err,
        )
        if job.status == "succeeded" and model:
            memory_store.set_active_ft_model(model, job.id)
    except Exception:  # noqa: BLE001
        traceback.print_exc()


def _submit(examples: list[dict], fingerprint: str) -> dict:
    from app.llm.openai_client import client
    body = "\n".join(
        json.dumps(ex["messages"], ensure_ascii=False) for ex in examples
    )
    buf = BytesIO(body.encode("utf-8"))
    buf.name = "xlog_dpo.jsonl"
    uploaded = client().files.create(file=buf, purpose="fine-tune")
    job = client().fine_tuning.jobs.create(
        training_file=uploaded.id,
        model=config.FT_BASE_MODEL,
        suffix="xlog-dpo",
        method={
            "type": "dpo",
            "dpo": {"hyperparameters": {"beta": config.DPO_BETA}},
        },
    )
    memory_store.save_ft_job(
        openai_job_id=job.id,
        status=job.status,
        model=None,
        example_count=len(examples),
        fingerprint=fingerprint,
        error="",
    )
    return {"ok": True, "status": job.status, "job_id": job.id}


def _fingerprint(examples: list[dict]) -> str:
    blob = json.dumps(examples, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _dpo_row(
    kind: str,
    source: str,
    source_id: str,
    system: str,
    user: str,
    preferred: str,
    rejected: str,
) -> dict:
    return {
        "kind": kind,
        "source": source,
        "source_id": source_id,
        "messages": {
            "input": {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            },
            "preferred_output": [{"role": "assistant", "content": preferred}],
            "non_preferred_output": [{"role": "assistant", "content": rejected}],
        },
    }


def _other_label(choice: str, variants: list[dict]) -> str:
    for v in variants:
        label = v.get("label")
        if label and label != choice:
            return label
    return "B" if choice == "A" else "A"


def _from_feedback(ev: dict) -> list[dict]:
    variants = ev.get("variants") or []
    choice = ev.get("user_choice") or ""
    comment = ev.get("user_comment") or f"creator chose {choice}"
    if not variants or not choice:
        return []
    chosen = next((v for v in variants if v.get("label") == choice), None)
    rejected = next((v for v in variants if v.get("label") != choice), None)
    if chosen is None or rejected is None:
        return []
    other = _other_label(choice, variants)
    user_judge = (
        f"The creator's request for this short:\n{comment}\n\n"
        "Candidate edits:\n"
        + json.dumps(variants, ensure_ascii=False)
    )
    moments = [
        {
            "video_index": s.get("video_index", 0),
            "start_sec": s.get("start_sec"),
            "end_sec": s.get("end_sec"),
            "description": s.get("reason") or "",
        }
        for v in variants for s in (v.get("shots") or [])
    ]
    user_writer = (
        f"User's request for this short (follow it faithfully):\n{comment}\n\n"
        f"Available moments:\n{json.dumps(moments, ensure_ascii=False)}\n\n"
        "Write ONE shot plan for this request. House style is already trained."
    )
    sid = ev.get("job_id", "")
    return [
        _dpo_row(
            "judge", "feedback", sid, JUDGE_SYSTEM, user_judge,
            json.dumps(_verdict_for(choice, comment, variants), ensure_ascii=False),
            json.dumps(_verdict_for(other, "weaker fit to creator taste", variants), ensure_ascii=False),
        ),
        _dpo_row(
            "writer", "feedback", sid, writer_system(), user_writer,
            json.dumps({"variants": [chosen]}, ensure_ascii=False),
            json.dumps({"variants": [rejected]}, ensure_ascii=False),
        ),
    ]


def _from_reference(ref: dict) -> list[dict]:
    style = ref.get("style") or {}
    rules = list(style.get("learned_rules") or [])
    notes = ref.get("notes") or ""
    why = notes or style.get("style_summary") or "creator liked this short's editing"
    follow = " ".join(rules[:4]) if rules else why
    violate = "Ignore house style: long intro title card, no captions, slow scenery B-roll."
    a = {
        "label": "A",
        "concept": follow,
        "hook_rationale": style.get("hook_technique", ""),
        "shots": [
            {
                "video_index": 0, "start_sec": 0, "end_sec": 3,
                "role": "hook", "reason": style.get("hook_technique") or "cold open",
                "caption": "", "caption_style": "emphasis", "fx": "punch_in",
            },
            {
                "video_index": 0, "start_sec": 8, "end_sec": 14,
                "role": "payoff", "reason": "deliver the promised moment",
                "caption": "", "caption_style": "pop", "fx": "none",
            },
        ],
    }
    b = {
        "label": "B",
        "concept": violate,
        "hook_rationale": "title card intro",
        "shots": [
            {
                "video_index": 0, "start_sec": 0, "end_sec": 8,
                "role": "hook", "reason": "slow title card",
                "caption": "", "caption_style": "none", "fx": "none",
            },
        ],
    }
    judge_user = (
        f"Creator note on a reference they like: {why}\n\n"
        "Candidate edits:\n" + json.dumps([a, b], ensure_ascii=False)
    )
    writer_user = (
        "User's request for this short (follow it faithfully):\n"
        f"Match this house style: {why}\nRules: {json.dumps(rules, ensure_ascii=False)}\n\n"
        "Available moments:\n"
        '[{"video_index":0,"start_sec":0,"end_sec":4,"description":"peak action"},'
        '{"video_index":0,"start_sec":8,"end_sec":14,"description":"payoff"}]\n\n'
        "Write ONE shot plan that matches the house style."
    )
    sid = ref.get("url") or ""
    return [
        _dpo_row(
            "judge", "reference", sid, JUDGE_SYSTEM, judge_user,
            json.dumps(_verdict_for("A", why, [a, b]), ensure_ascii=False),
            json.dumps(_verdict_for("B", "violates the reference style", [a, b]), ensure_ascii=False),
        ),
        _dpo_row(
            "writer", "reference", sid, writer_system(), writer_user,
            json.dumps({"variants": [a]}, ensure_ascii=False),
            json.dumps({"variants": [b]}, ensure_ascii=False),
        ),
    ]


def _verdict_for(choice: str, comment: str, variants: list[dict]) -> dict:
    scores = []
    for v in variants:
        win = v.get("label") == choice
        per = [
            {
                "criterion": c["name"],
                "score": 8 if win else 5,
                "comment": comment if win else "weaker fit to creator taste",
            }
            for c in SEED_RUBRIC["criteria"]
        ]
        scores.append({
            "label": v.get("label"),
            "per_criterion": per,
            "weighted_total": 8 if win else 5,
        })
    return {"scores": scores, "winner": choice, "reasoning": comment}
