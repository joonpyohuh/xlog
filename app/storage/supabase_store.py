"""Supabase Postgres backend for the xlog learning memory app."""
from __future__ import annotations

import json
import time
from typing import Any

from supabase import Client, create_client

from app import config

_client: Client | None = None


def reset_client() -> None:
    global _client
    _client = None


def enabled() -> bool:
    return bool(config.SUPABASE_URL and config.SUPABASE_KEY)


def client() -> Client:
    global _client
    if _client is None:
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _client


def _row_to_rubric(row: dict) -> dict:
    criteria = row["criteria"]
    preferences = row["preferences"]
    if isinstance(criteria, str):
        criteria = json.loads(criteria)
    if isinstance(preferences, str):
        preferences = json.loads(preferences)
    return {
        "version": row["version"],
        "owner": row["owner"],
        "criteria": criteria,
        "preferences": preferences,
        "notes": row["notes"],
    }


def load_rubric() -> dict | None:
    r = (
        client()
        .table("rubric_versions")
        .select("*")
        .order("version", desc=True)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data:
        return None
    return _row_to_rubric(r.data[0])


def save_rubric(rubric: dict, source: str = "unknown") -> None:
    now = int(time.time())
    prev = load_rubric()
    old_prefs = set(prev["preferences"]) if prev else set()
    new_prefs = list(rubric.get("preferences") or [])
    added = [p for p in new_prefs if p not in old_prefs]
    already_known = {p["rule"] for p in list_preferences()} & set(added)

    client().table("rubric_versions").insert(
        {
            "version": rubric["version"],
            "owner": rubric.get("owner", ""),
            "criteria": rubric.get("criteria") or [],
            "preferences": new_prefs,
            "notes": rubric.get("notes", ""),
            "source": source,
            "created_at": now,
        }
    ).execute()
    _sync_preferences(new_prefs, source, f"rubric_v{rubric['version']}", bump=False)
    _sync_preferences(list(already_known), source, f"rubric_v{rubric['version']}", bump=True)


def record_feedback(
    job_id: str,
    user_choice: str,
    user_comment: str,
    judge_winner: str | None,
    agreement: bool | None,
    variants: list | dict | None,
    judge_verdict: dict | None,
    rubric_version: int | None,
) -> None:
    client().table("feedback_events").insert(
        {
            "ts": int(time.time()),
            "job_id": job_id,
            "user_choice": user_choice,
            "user_comment": user_comment or "",
            "judge_winner": judge_winner,
            "agreement": agreement,
            "variants": variants,
            "judge_verdict": judge_verdict,
            "rubric_version": rubric_version,
        }
    ).execute()


def record_reference(
    url: str,
    notes: str,
    file_path: str | None,
    style: dict,
    rubric_version: int | None,
    ts: int | None = None,
) -> None:
    style = style or {}
    rules = list(style.get("learned_rules") or [])
    client().table("style_references").insert(
        {
            "ts": ts or int(time.time()),
            "url": url,
            "notes": notes or "",
            "file_path": file_path,
            "style_summary": style.get("style_summary", ""),
            "hook_technique": style.get("hook_technique", ""),
            "pacing": style.get("pacing", ""),
            "caption_style": style.get("caption_style", ""),
            "structure": style.get("structure", ""),
            "learned_rules": rules,
            "style": style,
            "rubric_version": rubric_version,
        }
    ).execute()
    _sync_preferences(rules, "reference", url, bump=True)


def list_references() -> list[dict]:
    r = client().table("style_references").select("*").order("ts").order("id").execute()
    out = []
    for row in r.data:
        style = row.get("style") or {
            "style_summary": row.get("style_summary") or "",
            "hook_technique": row.get("hook_technique") or "",
            "pacing": row.get("pacing") or "",
            "caption_style": row.get("caption_style") or "",
            "structure": row.get("structure") or "",
            "learned_rules": row.get("learned_rules") or [],
        }
        out.append(
            {
                "ts": row["ts"],
                "url": row["url"],
                "notes": row["notes"],
                "file": row.get("file_path"),
                "style": style,
                "rubric_version": row.get("rubric_version"),
            }
        )
    return out


def load_form() -> dict | None:
    r = (
        client()
        .table("shorts_form_versions")
        .select("*")
        .order("version", desc=True)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data:
        return None
    row = r.data[0]
    return {
        "version": row["version"],
        "source": row["source"],
        "structure": row["structure"],
        "global_rules": row["global_rules"],
    }


def save_form(form: dict) -> None:
    client().table("shorts_form_versions").insert(
        {
            "version": form.get("version", 1),
            "source": form.get("source", ""),
            "structure": form.get("structure") or [],
            "global_rules": form.get("global_rules") or [],
            "created_at": int(time.time()),
        }
    ).execute()


def list_feedback() -> list[dict]:
    r = client().table("feedback_events").select("*").order("ts").order("id").execute()
    out = []
    for row in r.data:
        out.append(
            {
                "ts": row["ts"],
                "job_id": row["job_id"],
                "user_choice": row["user_choice"],
                "user_comment": row["user_comment"],
                "judge_winner": row.get("judge_winner"),
                "agreement": row.get("agreement"),
                "variants": row.get("variants") or [],
                "judge_verdict": row.get("judge_verdict"),
                "rubric_version": row.get("rubric_version"),
            }
        )
    return out


def replace_ft_examples(examples: list[dict]) -> None:
    c = client()
    c.table("ft_examples").delete().gte("id", 0).execute()
    now = int(time.time())
    rows = [
        {
            "ts": now,
            "kind": ex.get("kind", ""),
            "source": ex.get("source", ""),
            "source_id": ex.get("source_id", ""),
            "messages": ex.get("messages") or [],
        }
        for ex in examples
    ]
    if rows:
        c.table("ft_examples").insert(rows).execute()


def list_ft_examples() -> list[dict]:
    r = client().table("ft_examples").select("kind, source, source_id, messages").order("id").execute()
    return [
        {
            "kind": row["kind"],
            "source": row["source"],
            "source_id": row.get("source_id"),
            "messages": row["messages"],
        }
        for row in r.data
    ]


def save_ft_job(
    openai_job_id: str,
    status: str,
    model: str | None,
    example_count: int,
    fingerprint: str,
    error: str = "",
) -> None:
    client().table("ft_jobs").insert(
        {
            "ts": int(time.time()),
            "openai_job_id": openai_job_id,
            "status": status,
            "model": model,
            "example_count": example_count,
            "fingerprint": fingerprint,
            "error": error or "",
        }
    ).execute()


def latest_ft_job() -> dict | None:
    r = client().table("ft_jobs").select("*").order("id", desc=True).limit(1).execute()
    return r.data[0] if r.data else None


def set_active_ft_model(model: str, job_id: str | None) -> None:
    client().table("ft_active").upsert(
        {"id": 1, "model": model, "job_id": job_id, "updated_at": int(time.time())}
    ).execute()


def get_active_ft_model() -> str | None:
    r = client().table("ft_active").select("model").eq("id", 1).limit(1).execute()
    return r.data[0]["model"] if r.data else None


def list_preferences() -> list[dict]:
    r = (
        client()
        .table("learned_preferences")
        .select("rule, source, times_seen, last_seen_at")
        .order("times_seen", desc=True)
        .order("last_seen_at", desc=True)
        .order("id")
        .execute()
    )
    return r.data


def stats() -> dict:
    c = client()
    latest = load_rubric()
    def _count(table: str) -> int:
        return c.table(table).select("id", count="exact").limit(0).execute().count or 0
    return {
        "rubric_version": latest["version"] if latest else 0,
        "feedback_count": _count("feedback_events"),
        "reference_count": _count("style_references"),
        "preference_count": _count("learned_preferences"),
        "form_count": _count("shorts_form_versions"),
        "rubric_snapshots": _count("rubric_versions"),
        "ft_examples": _count("ft_examples"),
        "ft_model": get_active_ft_model(),
        "backend": "supabase",
    }


def is_empty() -> bool:
    return _count_rubric() == 0


def _count_rubric() -> int:
    return client().table("rubric_versions").select("id", count="exact").limit(0).execute().count or 0


def _sync_preferences(
    rules: list[str],
    source: str,
    source_id: str,
    bump: bool,
) -> None:
    now = int(time.time())
    c = client()
    for rule in rules:
        rule = (rule or "").strip()
        if not rule:
            continue
        existing = c.table("learned_preferences").select("id, times_seen").eq("rule", rule).limit(1).execute()
        if not existing.data:
            c.table("learned_preferences").insert(
                {
                    "rule": rule,
                    "source": source,
                    "source_id": source_id,
                    "first_seen_at": now,
                    "last_seen_at": now,
                    "times_seen": 1,
                }
            ).execute()
        elif bump:
            row = existing.data[0]
            c.table("learned_preferences").update(
                {
                    "times_seen": row["times_seen"] + 1,
                    "last_seen_at": now,
                    "source": source,
                    "source_id": source_id,
                }
            ).eq("id", row["id"]).execute()


def push_local_snapshot(data: dict[str, Any]) -> None:
    """One-shot upload of a local SQLite snapshot into Supabase."""
    if not is_empty():
        return
    for rubric in data.get("rubrics", []):
        save_rubric(rubric, source=rubric.get("source", "migrated"))
    for ev in data.get("feedback", []):
        record_feedback(
            job_id=ev["job_id"],
            user_choice=ev["user_choice"],
            user_comment=ev.get("user_comment", ""),
            judge_winner=ev.get("judge_winner"),
            agreement=ev.get("agreement"),
            variants=ev.get("variants"),
            judge_verdict=ev.get("judge_verdict"),
            rubric_version=ev.get("rubric_version"),
        )
    for ref in data.get("references", []):
        record_reference(
            url=ref["url"],
            notes=ref.get("notes", ""),
            file_path=ref.get("file"),
            style=ref.get("style") or {},
            rubric_version=ref.get("rubric_version"),
            ts=ref.get("ts"),
        )
    form = data.get("form")
    if form:
        save_form(form)
    replace_ft_examples(data.get("ft_examples") or [])
