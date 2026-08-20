"""How a human editor rewrote the AI starting timeline.

Only written when the editor consented. Later this pair
(ai_plan → human_plan + actions) is the training signal for a real AI editor.
The current pipeline still produces a starting cut; it is not the product.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app import config

TRACE_DIR: Path = config.DATA_DIR / "traces"
TRACE_LOG: Path = TRACE_DIR / "edit_log.jsonl"


def record(event: dict) -> dict | None:
    if not event.get("consent"):
        return None
    row = {
        "ts": int(event.get("ts") or time.time()),
        "job_id": event.get("job_id") or "",
        "kind": event.get("kind") or "edit",
        "pick": event.get("pick") or "",
        "comment": (event.get("comment") or "")[:800],
        "brief": event.get("brief") or {},
        "ai_plan": event.get("ai_plan") or [],
        "human_plan": event.get("human_plan") or [],
        "actions": event.get("actions") or [],
        "tighten": event.get("tighten") or {},
    }
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    with TRACE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        from app.storage import memory as memory_store
        memory_store.record_trace(row)
    except Exception:  # noqa: BLE001 — jsonl is the source of truth
        pass
    return row


def list_traces(limit: int = 40) -> list[dict]:
    if not TRACE_LOG.exists():
        return []
    lines = [ln for ln in TRACE_LOG.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))


def stats() -> dict:
    rows = list_traces(limit=10_000)
    edited = sum(1 for r in rows if r.get("human_plan"))
    picks = sum(1 for r in rows if r.get("pick"))
    return {
        "traces": len(rows),
        "picks": picks,
        "human_edits": edited,
        "goal": "index → stringout/selects/alts → Premiere. Learn keep/discard/handles.",
    }


def compact_plan(variants: list[dict] | None, label: str | None = None) -> list[dict]:
    """Drop render-only blobs; keep the edit decisions."""
    out = []
    for v in variants or []:
        if label and v.get("label") != label:
            continue
        shots = []
        for s in v.get("shots") or []:
            shots.append({
                "video_index": s.get("video_index"),
                "start_sec": s.get("start_sec"),
                "end_sec": s.get("end_sec"),
                "ranges": s.get("ranges"),
                "role": s.get("role"),
                "caption": s.get("caption"),
                "caption_style": s.get("caption_style"),
                "fx": s.get("fx"),
                "reason": s.get("reason"),
            })
        out.append({
            "label": v.get("label"),
            "concept": v.get("concept"),
            "total_sec": v.get("total_sec"),
            "shots": shots,
        })
    return out


if __name__ == "__main__":
    plan = compact_plan([{"label": "A", "shots": [
        {"start_sec": 1, "end_sec": 3, "caption": "훅", "role": "hook", "sfx_path": "/tmp/x"},
    ]}])
    assert plan[0]["shots"][0]["caption"] == "훅"
    assert "sfx_path" not in plan[0]["shots"][0]
    assert record({"consent": False}) is None
    print("traces self-check ok")
