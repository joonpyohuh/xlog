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


def job_dir(job_id: str) -> Path:
    return config.JOBS_DIR / job_id


def create_job(
    video_paths: list[str],
    instruction: str = "",
    source_urls: list[str] | None = None,
) -> dict:
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    d = job_dir(job_id)
    d.mkdir(parents=True)
    job = {
        "id": job_id,
        "created_at": int(time.time()),
        "stage": "queued",
        "error": None,
        "videos": video_paths,
        "source_urls": source_urls or [],
        "instruction": instruction,
        "analysis": None,
        "variants": None,
        "judge_verdict": None,
        "outputs": {},          # label -> rendered mp4 path
        "user_choice": None,
        "user_comment": None,
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


def set_stage(job: dict, stage: str, error: str | None = None) -> None:
    job["stage"] = stage
    job["error"] = error
    save_job(job)
