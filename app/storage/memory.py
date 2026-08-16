"""Persistent learning memory for xlog.

SQLite is the source of truth for everything the app learns:
  - A/B picks + the creator's reason
  - YouTube Shorts reference analyses (what's good, why, editing style)
  - rubric versions
  - accumulated editing preferences
  - shorts-form grammar versions

Existing JSON/jsonl files stay as a human-readable backup. Callers in
rubric/feedback/reference/shorts_form keep their signatures so the rest
of the app (screenwriter, judge, routes, UI) does not need to change.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from app import config
from app.storage import supabase_store

DB_PATH: Path = config.DATA_DIR / "xlog_memory.db"

_lock = threading.RLock()
_initialized = False
_migrate_files = True
_force_sqlite = False


def init(path: Path | None = None, *, migrate: bool = True, force_sqlite: bool = False) -> None:
    """Open (and create) the memory DB. Safe to call repeatedly."""
    global DB_PATH, _initialized, _migrate_files, _force_sqlite
    with _lock:
        if force_sqlite:
            _force_sqlite = True
        if supabase_store.enabled() and not _force_sqlite:
            if not _initialized:
                from app.storage import sync_supabase
                sync_supabase.sync_if_needed()
                _initialized = True
            return
        if path is not None:
            DB_PATH = Path(path)
            _initialized = False
        _migrate_files = migrate
        _ensure_locked()


def _use_cloud() -> bool:
    return supabase_store.enabled() and not _force_sqlite


def _ensure_locked() -> None:
    global _initialized
    if _initialized:
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        _create_tables(conn)
        if _migrate_files:
            _migrate_from_files(conn)
        conn.commit()
    _initialized = True


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS rubric_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            owner TEXT NOT NULL,
            criteria TEXT NOT NULL,
            preferences TEXT NOT NULL,
            notes TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'unknown',
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            user_choice TEXT NOT NULL,
            user_comment TEXT NOT NULL,
            judge_winner TEXT,
            agreement INTEGER,
            variants TEXT,
            judge_verdict TEXT,
            rubric_version INTEGER
        );

        CREATE TABLE IF NOT EXISTS style_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            url TEXT NOT NULL,
            notes TEXT NOT NULL,
            file_path TEXT,
            style_summary TEXT,
            hook_technique TEXT,
            pacing TEXT,
            caption_style TEXT,
            structure TEXT,
            learned_rules TEXT,
            style TEXT,
            rubric_version INTEGER
        );

        CREATE TABLE IF NOT EXISTS shorts_form_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            source TEXT NOT NULL,
            structure TEXT NOT NULL,
            global_rules TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS learned_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            source_id TEXT,
            first_seen_at INTEGER NOT NULL,
            last_seen_at INTEGER NOT NULL,
            times_seen INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS ft_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            kind TEXT NOT NULL,
            source TEXT NOT NULL,
            source_id TEXT,
            messages TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ft_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            openai_job_id TEXT,
            status TEXT NOT NULL,
            model TEXT,
            example_count INTEGER NOT NULL DEFAULT 0,
            fingerprint TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS ft_active (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model TEXT NOT NULL,
            job_id TEXT,
            updated_at INTEGER NOT NULL
        );
        """
    )


def _row_to_rubric(row: sqlite3.Row) -> dict:
    return {
        "version": row["version"],
        "owner": row["owner"],
        "criteria": json.loads(row["criteria"]),
        "preferences": json.loads(row["preferences"]),
        "notes": row["notes"],
    }


def load_rubric() -> dict | None:
    init()
    if _use_cloud():
        return supabase_store.load_rubric()
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM rubric_versions ORDER BY version DESC, id DESC LIMIT 1"
        ).fetchone()
    return _row_to_rubric(row) if row else None


def save_rubric(rubric: dict, source: str = "unknown") -> None:
    """Append a rubric snapshot and sync preferences (bump only newly added)."""
    init()
    if _use_cloud():
        return supabase_store.save_rubric(rubric, source=source)
    now = int(time.time())
    with _lock, _connect() as conn:
        prev = conn.execute(
            "SELECT preferences FROM rubric_versions "
            "ORDER BY version DESC, id DESC LIMIT 1"
        ).fetchone()
        old_prefs = set(json.loads(prev["preferences"])) if prev else set()
        new_prefs = list(rubric.get("preferences") or [])
        added = [p for p in new_prefs if p not in old_prefs]
        # snapshot before insert-or-ignore, otherwise every new rule looks "known"
        already_known = [
            p for p in added
            if conn.execute(
                "SELECT 1 FROM learned_preferences WHERE rule = ?", (p,)
            ).fetchone()
        ]

        conn.execute(
            "INSERT INTO rubric_versions "
            "(version, owner, criteria, preferences, notes, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                rubric["version"],
                rubric.get("owner", ""),
                json.dumps(rubric.get("criteria") or [], ensure_ascii=False),
                json.dumps(new_prefs, ensure_ascii=False),
                rubric.get("notes", ""),
                source,
                now,
            ),
        )
        _sync_preferences(conn, new_prefs, source, f"rubric_v{rubric['version']}", bump=False)
        _sync_preferences(
            conn, already_known, source, f"rubric_v{rubric['version']}", bump=True
        )
        conn.commit()


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
    init()
    if _use_cloud():
        return supabase_store.record_feedback(
            job_id, user_choice, user_comment, judge_winner, agreement,
            variants, judge_verdict, rubric_version,
        )
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO feedback_events "
            "(ts, job_id, user_choice, user_comment, judge_winner, agreement, "
            " variants, judge_verdict, rubric_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(time.time()),
                job_id,
                user_choice,
                user_comment or "",
                judge_winner,
                None if agreement is None else int(bool(agreement)),
                json.dumps(variants, ensure_ascii=False) if variants is not None else None,
                json.dumps(judge_verdict, ensure_ascii=False) if judge_verdict is not None else None,
                rubric_version,
            ),
        )
        conn.commit()


def record_reference(
    url: str,
    notes: str,
    file_path: str | None,
    style: dict,
    rubric_version: int | None,
    ts: int | None = None,
) -> None:
    init()
    if _use_cloud():
        return supabase_store.record_reference(
            url, notes, file_path, style, rubric_version, ts=ts,
        )
    style = style or {}
    rules = list(style.get("learned_rules") or [])
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO style_references "
            "(ts, url, notes, file_path, style_summary, hook_technique, pacing, "
            " caption_style, structure, learned_rules, style, rubric_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts or int(time.time()),
                url,
                notes or "",
                file_path,
                style.get("style_summary", ""),
                style.get("hook_technique", ""),
                style.get("pacing", ""),
                style.get("caption_style", ""),
                style.get("structure", ""),
                json.dumps(rules, ensure_ascii=False),
                json.dumps(style, ensure_ascii=False),
                rubric_version,
            ),
        )
        _sync_preferences(conn, rules, "reference", url, bump=True)
        conn.commit()


def list_references() -> list[dict]:
    """Return records in the same shape the JSONL log / frontend already use."""
    init()
    if _use_cloud():
        return supabase_store.list_references()
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM style_references ORDER BY ts ASC, id ASC"
        ).fetchall()
    out = []
    for row in rows:
        style = json.loads(row["style"]) if row["style"] else {
            "style_summary": row["style_summary"] or "",
            "hook_technique": row["hook_technique"] or "",
            "pacing": row["pacing"] or "",
            "caption_style": row["caption_style"] or "",
            "structure": row["structure"] or "",
            "learned_rules": json.loads(row["learned_rules"] or "[]"),
        }
        out.append(
            {
                "ts": row["ts"],
                "url": row["url"],
                "notes": row["notes"],
                "file": row["file_path"],
                "style": style,
                "rubric_version": row["rubric_version"],
            }
        )
    return out


def load_form() -> dict | None:
    init()
    if _use_cloud():
        return supabase_store.load_form()
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM shorts_form_versions ORDER BY version DESC, id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    return {
        "version": row["version"],
        "source": row["source"],
        "structure": json.loads(row["structure"]),
        "global_rules": json.loads(row["global_rules"]),
    }


def save_form(form: dict) -> None:
    init()
    if _use_cloud():
        return supabase_store.save_form(form)
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO shorts_form_versions "
            "(version, source, structure, global_rules, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                form.get("version", 1),
                form.get("source", ""),
                json.dumps(form.get("structure") or [], ensure_ascii=False),
                json.dumps(form.get("global_rules") or [], ensure_ascii=False),
                int(time.time()),
            ),
        )
        conn.commit()


def list_feedback() -> list[dict]:
    init()
    if _use_cloud():
        return supabase_store.list_feedback()
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback_events ORDER BY ts ASC, id ASC"
        ).fetchall()
    out = []
    for row in rows:
        out.append(
            {
                "ts": row["ts"],
                "job_id": row["job_id"],
                "user_choice": row["user_choice"],
                "user_comment": row["user_comment"],
                "judge_winner": row["judge_winner"],
                "agreement": row["agreement"],
                "variants": json.loads(row["variants"]) if row["variants"] else [],
                "judge_verdict": json.loads(row["judge_verdict"]) if row["judge_verdict"] else None,
                "rubric_version": row["rubric_version"],
            }
        )
    return out


def replace_ft_examples(examples: list[dict]) -> None:
    init()
    if _use_cloud():
        return supabase_store.replace_ft_examples(examples)
    now = int(time.time())
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM ft_examples")
        for ex in examples:
            conn.execute(
                "INSERT INTO ft_examples (ts, kind, source, source_id, messages) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    now,
                    ex.get("kind", ""),
                    ex.get("source", ""),
                    ex.get("source_id", ""),
                    json.dumps(ex.get("messages") or [], ensure_ascii=False),
                ),
            )
        conn.commit()


def list_ft_examples() -> list[dict]:
    init()
    if _use_cloud():
        return supabase_store.list_ft_examples()
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT kind, source, source_id, messages FROM ft_examples ORDER BY id ASC"
        ).fetchall()
    return [
        {
            "kind": r["kind"],
            "source": r["source"],
            "source_id": r["source_id"],
            "messages": json.loads(r["messages"]),
        }
        for r in rows
    ]


def save_ft_job(
    openai_job_id: str,
    status: str,
    model: str | None,
    example_count: int,
    fingerprint: str,
    error: str = "",
) -> None:
    init()
    if _use_cloud():
        return supabase_store.save_ft_job(
            openai_job_id, status, model, example_count, fingerprint, error=error,
        )
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO ft_jobs "
            "(ts, openai_job_id, status, model, example_count, fingerprint, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                int(time.time()),
                openai_job_id,
                status,
                model,
                example_count,
                fingerprint,
                error or "",
            ),
        )
        conn.commit()


def latest_ft_job() -> dict | None:
    init()
    if _use_cloud():
        return supabase_store.latest_ft_job()
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM ft_jobs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def set_active_ft_model(model: str, job_id: str | None) -> None:
    init()
    if _use_cloud():
        return supabase_store.set_active_ft_model(model, job_id)
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO ft_active (id, model, job_id, updated_at) VALUES (1, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET model = excluded.model, "
            "job_id = excluded.job_id, updated_at = excluded.updated_at",
            (model, job_id, int(time.time())),
        )
        conn.commit()


def get_active_ft_model() -> str | None:
    init()
    if _use_cloud():
        return supabase_store.get_active_ft_model()
    with _lock, _connect() as conn:
        row = conn.execute("SELECT model FROM ft_active WHERE id = 1").fetchone()
    return row["model"] if row else None


def list_preferences() -> list[dict]:
    """Preferences ordered by how often they have been reinforced."""
    init()
    if _use_cloud():
        return supabase_store.list_preferences()
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT rule, source, times_seen, last_seen_at "
            "FROM learned_preferences "
            "ORDER BY times_seen DESC, last_seen_at DESC, id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def stats() -> dict:
    init()
    if _use_cloud():
        return supabase_store.stats()
    with _lock, _connect() as conn:
        def _count(table: str) -> int:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        latest = conn.execute(
            "SELECT version FROM rubric_versions ORDER BY version DESC, id DESC LIMIT 1"
        ).fetchone()
        return {
            "rubric_version": latest["version"] if latest else 0,
            "feedback_count": _count("feedback_events"),
            "reference_count": _count("style_references"),
            "preference_count": _count("learned_preferences"),
            "form_count": _count("shorts_form_versions"),
            "rubric_snapshots": _count("rubric_versions"),
            "ft_examples": _count("ft_examples"),
            "ft_model": (
                conn.execute("SELECT model FROM ft_active WHERE id = 1").fetchone()
                or {"model": None}
            )["model"],
            "backend": "sqlite",
        }


def _sync_preferences(
    conn: sqlite3.Connection,
    rules: list[str],
    source: str,
    source_id: str,
    bump: bool,
) -> None:
    now = int(time.time())
    for rule in rules:
        rule = (rule or "").strip()
        if not rule:
            continue
        row = conn.execute(
            "SELECT id, times_seen FROM learned_preferences WHERE rule = ?", (rule,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO learned_preferences "
                "(rule, source, source_id, first_seen_at, last_seen_at, times_seen) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (rule, source, source_id, now, now),
            )
        elif bump:
            conn.execute(
                "UPDATE learned_preferences "
                "SET times_seen = times_seen + 1, last_seen_at = ?, "
                "    source = ?, source_id = ? "
                "WHERE id = ?",
                (now, source, source_id, row["id"]),
            )


def _migrate_from_files(conn: sqlite3.Connection) -> None:
    """One-shot import of the existing JSON/jsonl learning files."""
    if conn.execute("SELECT COUNT(*) FROM rubric_versions").fetchone()[0] > 0:
        return

    history_dir = config.RUBRIC_DIR / "history"
    if history_dir.exists():
        for path in sorted(history_dir.glob("rubric_v*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            stamp = _stamp_from_history_name(path.name)
            _insert_rubric_row(conn, data, source="migrated", created_at=stamp)

    current_path = config.RUBRIC_DIR / "rubric.json"
    if current_path.exists():
        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            current = None
        if current is not None:
            ver = current.get("version")
            already = conn.execute(
                "SELECT 1 FROM rubric_versions WHERE version = ?", (ver,)
            ).fetchone()
            if already is None:
                _insert_rubric_row(conn, current, source="migrated")

    latest = conn.execute(
        "SELECT preferences, version FROM rubric_versions "
        "ORDER BY version DESC, id DESC LIMIT 1"
    ).fetchone()
    if latest:
        _sync_preferences(
            conn,
            json.loads(latest["preferences"]),
            "migrated",
            f"rubric_v{latest['version']}",
            bump=False,
        )

    log_path = config.RUBRIC_DIR / "feedback_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            conn.execute(
                "INSERT INTO feedback_events "
                "(ts, job_id, user_choice, user_comment, judge_winner, agreement, "
                " variants, judge_verdict, rubric_version) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
                (
                    ev.get("ts") or int(time.time()),
                    ev.get("job_id", ""),
                    ev.get("user_choice", ""),
                    ev.get("user_comment", ""),
                    ev.get("judge_winner"),
                    None if ev.get("agreement") is None else int(bool(ev["agreement"])),
                ),
            )

    refs_path = config.RUBRIC_DIR / "references.jsonl"
    if refs_path.exists():
        for line in refs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            style = rec.get("style") or {}
            rules = list(style.get("learned_rules") or [])
            conn.execute(
                "INSERT INTO style_references "
                "(ts, url, notes, file_path, style_summary, hook_technique, pacing, "
                " caption_style, structure, learned_rules, style, rubric_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rec.get("ts") or int(time.time()),
                    rec.get("url", ""),
                    rec.get("notes", ""),
                    rec.get("file"),
                    style.get("style_summary", ""),
                    style.get("hook_technique", ""),
                    style.get("pacing", ""),
                    style.get("caption_style", ""),
                    style.get("structure", ""),
                    json.dumps(rules, ensure_ascii=False),
                    json.dumps(style, ensure_ascii=False),
                    rec.get("rubric_version"),
                ),
            )
            _sync_preferences(conn, rules, "reference", rec.get("url", ""), bump=False)

    form_path = config.RUBRIC_DIR / "shorts_form.json"
    if form_path.exists():
        try:
            form = json.loads(form_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            form = None
        if form is not None:
            conn.execute(
                "INSERT INTO shorts_form_versions "
                "(version, source, structure, global_rules, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    form.get("version", 1),
                    form.get("source", "migrated"),
                    json.dumps(form.get("structure") or [], ensure_ascii=False),
                    json.dumps(form.get("global_rules") or [], ensure_ascii=False),
                    int(time.time()),
                ),
            )


def _insert_rubric_row(
    conn: sqlite3.Connection,
    data: dict,
    source: str,
    created_at: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO rubric_versions "
        "(version, owner, criteria, preferences, notes, source, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            data.get("version", 1),
            data.get("owner", ""),
            json.dumps(data.get("criteria") or [], ensure_ascii=False),
            json.dumps(data.get("preferences") or [], ensure_ascii=False),
            data.get("notes", ""),
            source,
            created_at or int(time.time()),
        ),
    )


def _stamp_from_history_name(name: str) -> int:
    # rubric_v{version}_{unix}.json
    stem = name.rsplit(".", 1)[0]
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return int(time.time())
