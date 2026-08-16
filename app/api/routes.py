"""HTTP API for the local xlog site."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app import config
from app.evaluation import feedback, finetune, rubric as rubric_store
from app.knowledge import reference as reference_mod
from app.knowledge import shorts_form
from app.pipeline import ingest, orchestrator
from app.storage import cleanup
from app.storage import jobs as job_store

router = APIRouter(prefix="/api")


@router.post("/jobs")
async def create_job(
    background: BackgroundTasks,
    files: list[UploadFile] | None = File(default=None),
    youtube_urls: str = Form(""),
    instruction: str = Form(""),
):
    """Upload 1~3 raw videos and/or paste YouTube URLs (+ optional request)."""
    files = files or []
    try:
        urls = ingest.parse_youtube_urls(youtube_urls)
    except ingest.IngestError as e:
        raise HTTPException(400, str(e)) from e

    total = len(files) + len(urls)
    if not (config.MIN_VIDEOS <= total <= config.MAX_VIDEOS):
        raise HTTPException(
            400,
            f"give between {config.MIN_VIDEOS} and {config.MAX_VIDEOS} "
            f"videos (files and/or YouTube URLs), got {total}",
        )

    saved: list[str] = []
    batch = uuid.uuid4().hex[:8]
    for f in files:
        ext = Path(f.filename or "video.mp4").suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"unsupported format: {f.filename}")
        dest = config.UPLOAD_DIR / f"{batch}_{Path(f.filename).name}"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(dest))

    job = job_store.create_job(saved, instruction=instruction, source_urls=urls)
    background.add_task(orchestrator.run_job, job["id"])
    return {"job_id": job["id"], "stage": job["stage"]}


@router.get("/jobs")
async def get_jobs():
    return [
        {"id": j["id"], "stage": j["stage"], "created_at": j["created_at"],
         "error": j["error"], "user_choice": j["user_choice"]}
        for j in job_store.list_jobs()
    ]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        return job_store.load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "job not found")


@router.get("/jobs/{job_id}/video/{label}")
async def get_video(job_id: str, label: str):
    job = job_store.load_job(job_id)
    path = job["outputs"].get(label)
    if not path or not Path(path).exists():
        raise HTTPException(404, "video not found")
    return FileResponse(path, media_type="video/mp4")


@router.post("/jobs/{job_id}/evaluate")
async def evaluate(
    job_id: str,
    background: BackgroundTasks,
    choice: str = Form(...),
    comment: str = Form(""),
):
    """The pilot user picks the better variant; taste is fine-tuned from it."""
    job = job_store.load_job(job_id)
    if job["stage"] != "awaiting_evaluation":
        raise HTTPException(400, f"job is in stage '{job['stage']}'")
    labels = {v["label"] for v in job["variants"] or []}
    if choice not in labels:
        raise HTTPException(400, f"choice must be one of {sorted(labels)}")

    # Persist the pick immediately so a slow/failed LLM update cannot
    # swallow the creator's choice.
    job["user_choice"] = choice
    job["user_comment"] = comment
    job_store.set_stage(job, "done")
    cleanup.slim_finished_job(job, keep_shorts=True)

    current = rubric_store.load_rubric()

    def _learn() -> None:
        feedback.record_feedback(
            job_id=job_id,
            variants=job.get("variants") or [],
            judge_verdict=job.get("judge_verdict"),
            user_choice=choice,
            user_comment=comment,
        )
        finetune.schedule()

    background.add_task(_learn)
    return {"ok": True, "rubric_version": current["version"]}


@router.get("/rubric")
async def get_rubric():
    return rubric_store.load_rubric()


@router.get("/form")
async def get_form():
    return shorts_form.load_form()


@router.post("/form/refresh")
async def refresh_form(notes: str = Form("")):
    return shorts_form.refresh_form(notes)


# ------------------ Reference-style learning (YouTube links) ------------------ #

@router.post("/references")
async def learn_reference(
    background: BackgroundTasks,
    url: str = Form(...),
    notes: str = Form(""),
):
    """Give xlog a YouTube link whose editing style should be learned."""
    try:
        result = reference_mod.learn_from_youtube(url, notes)
    except reference_mod.ReferenceError as e:
        raise HTTPException(400, str(e))
    background.add_task(finetune.schedule)
    return result


@router.get("/references")
async def get_references():
    return reference_mod.list_references()
