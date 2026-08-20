"""File-based job store. Each job = one directory under data/jobs/<job_id>/
containing job.json (state) plus working files and rendered outputs."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from app import config

STAGES = [
    "queued",
    "ingesting",
    "analyzing",
    "writing_plans",
    "rendering",
    "judging",
    "awaiting_evaluation",   # user must pick A or B
    "done",
    "failed",
]

# Base percent when a stage starts. Sub-steps pass a higher pct.
STAGE_PROGRESS = {
    "queued": (0, "Queued"),
    "ingesting": (8, "Checking source"),
    "analyzing": (18, "Indexing scenes"),
    "writing_plans": (60, "Starting timeline"),
    "rendering": (70, "Preview"),
    "judging": (88, "Comparing starts"),
    "awaiting_evaluation": (100, "Your call"),
    "done": (100, "Done"),
    "failed": (None, "Failed"),
}


# Stages a restart can strand. `awaiting_evaluation` and later are resting
# states and must survive untouched.
IN_FLIGHT = frozenset(STAGES[: STAGES.index("awaiting_evaluation")])


def job_dir(job_id: str) -> Path:
    """Resolve a job's directory, refusing any id that escapes JOBS_DIR.
    Every reader/writer goes through here, so one guard covers them all."""
    root = config.JOBS_DIR.resolve()
    d = (root / job_id).resolve()
    if d.parent != root:
        raise FileNotFoundError(f"invalid job id: {job_id!r}")
    return d


def create_job(
    video_paths: list[str],
    instruction: str = "",
    source_urls: list[str] | None = None,
    brief: dict | None = None,
    font: str = "malgun",
    voice: str = "auto",
    source_edited: bool = False,
    quality: str = "fast",
    editor: dict | None = None,
    consent: bool = False,
) -> dict:
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    d = job_dir(job_id)
    d.mkdir(parents=True)
    now = int(time.time())
    job = {
        "id": job_id,
        "created_at": now,
        "stage": "queued",
        "error": None,
        "videos": video_paths,
        "source_urls": source_urls or [],
        "instruction": instruction,
        "brief": brief or {},
        "font": font or "malgun",
        "voice": voice or "auto",
        "source_edited": bool(source_edited),
        "quality": quality or "fast",
        "editor": editor if isinstance(editor, dict) else {},
        "consent": bool(consent),
        "analysis": None,
        "variants": None,
        "judge_verdict": None,
        "outputs": {},          # label -> rendered mp4 path
        "user_choice": None,
        "user_comment": None,
        "progress": {
            "pct": 0,
            "label": "대기 중",
            "detail": "",
            "started_at": now,
            "updated_at": now,
        },
    }
    save_job(job)
    return job


def save_job(job: dict) -> None:
    (job_dir(job["id"]) / "job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_job(job_id: str) -> dict:
    return json.loads((job_dir(job_id) / "job.json").read_text(encoding="utf-8"))


def list_jobs() -> list[dict]:
    jobs = []
    for p in sorted(config.JOBS_DIR.glob("*/job.json"), reverse=True):
        try:
            jobs.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return jobs


def set_stage(
    job: dict,
    stage: str,
    error: str | None = None,
    *,
    detail: str = "",
    pct: int | None = None,
) -> None:
    job["stage"] = stage
    job["error"] = error
    base, label = STAGE_PROGRESS.get(stage, (0, stage))
    prev = job.get("progress") or {}
    job["progress"] = {
        "pct": prev.get("pct", 0) if base is None else (pct if pct is not None else base),
        "label": label,
        "detail": detail or prev.get("detail") or "",
        "started_at": prev.get("started_at") or int(time.time()),
        "updated_at": int(time.time()),
    }
    save_job(job)


def fail_stranded() -> int:
    """Mark jobs that a restart killed mid-pipeline as failed.

    No stage is checkpointed, so a stranded job cannot be resumed — leaving it
    in `ingesting` just spins the UI forever. Returns how many were closed.
    """
    n = 0
    for job in list_jobs():
        if job.get("stage") in IN_FLIGHT:
            set_stage(
                job, "failed",
                error="서버가 재시작되어 작업이 중단되었습니다. 다시 실행해 주세요.",
                detail="재시작으로 중단됨",
            )
            n += 1
    return n


if __name__ == "__main__":
    for bad in ("../../.env", "..", "a/b", "/etc/passwd"):
        try:
            job_dir(bad)
        except FileNotFoundError:
            continue
        raise AssertionError(f"job_dir accepted {bad!r}")
    j = create_job([], instruction="self-check")
    assert job_dir(j["id"]).parent == config.JOBS_DIR.resolve()
    set_stage(j, "analyzing")
    assert fail_stranded() >= 1
    assert load_job(j["id"])["stage"] == "failed"
    set_stage(j, "awaiting_evaluation")
    fail_stranded()
    assert load_job(j["id"])["stage"] == "awaiting_evaluation", "resting stage was killed"

    import shutil
    shutil.rmtree(job_dir(j["id"]))   # don't leave a fake job in the creator's list
    print("jobs self-check ok")
