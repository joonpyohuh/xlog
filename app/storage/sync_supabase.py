"""Push local SQLite learning data into Supabase (one-time)."""
from __future__ import annotations

import json
import sqlite3

from app import config
from app.storage import memory as memory_store
from app.storage import supabase_store


def _sqlite_snapshot() -> dict:
    path = config.DATA_DIR / "xlog_memory.db"
    if not path.exists():
        return {}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    def rows(table: str) -> list[dict]:
        try:
            return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
        except sqlite3.OperationalError:
            return []

    rubrics = []
    for row in rows("rubric_versions"):
        rubrics.append(
            {
                "version": row["version"],
                "owner": row["owner"],
                "criteria": json.loads(row["criteria"]),
                "preferences": json.loads(row["preferences"]),
                "notes": row["notes"],
                "source": row.get("source") or "migrated",
            }
        )

    feedback = []
    for row in rows("feedback_events"):
        feedback.append(
            {
                "job_id": row["job_id"],
                "user_choice": row["user_choice"],
                "user_comment": row["user_comment"],
                "judge_winner": row["judge_winner"],
                "agreement": bool(row["agreement"]) if row["agreement"] is not None else None,
                "variants": json.loads(row["variants"]) if row["variants"] else [],
                "judge_verdict": json.loads(row["judge_verdict"]) if row["judge_verdict"] else None,
                "rubric_version": row["rubric_version"],
            }
        )

    references = []
    for row in rows("style_references"):
        style = json.loads(row["style"]) if row["style"] else {}
        references.append(
            {
                "ts": row["ts"],
                "url": row["url"],
                "notes": row["notes"],
                "file": row["file_path"],
                "style": style,
                "rubric_version": row["rubric_version"],
            }
        )

    form = None
    forms = rows("shorts_form_versions")
    if forms:
        last = forms[-1]
        form = {
            "version": last["version"],
            "source": last["source"],
            "structure": json.loads(last["structure"]),
            "global_rules": json.loads(last["global_rules"]),
        }

    ft_examples = []
    for row in rows("ft_examples"):
        ft_examples.append(
            {
                "kind": row["kind"],
                "source": row["source"],
                "source_id": row["source_id"],
                "messages": json.loads(row["messages"]),
            }
        )

    conn.close()
    return {
        "rubrics": rubrics,
        "feedback": feedback,
        "references": references,
        "form": form,
        "ft_examples": ft_examples,
    }


def sync_if_needed() -> bool:
    if not supabase_store.enabled():
        return False
    if not supabase_store.is_empty():
        return False
    snap = _sqlite_snapshot()
    if not snap.get("rubrics"):
        memory_store.init(migrate=True, force_sqlite=True)
        snap = _sqlite_snapshot()
    if not snap.get("rubrics"):
        return False
    supabase_store.push_local_snapshot(snap)
    return True


if __name__ == "__main__":
    ok = sync_if_needed()
    print("synced" if ok else "skipped", supabase_store.stats() if supabase_store.enabled() else {})
