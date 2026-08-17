"""HTTP API for the local xlog site."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app import config
from app.evaluation import feedback, rubric as rubric_store, taste
from app.knowledge import reference as reference_mod
from app.knowledge import shorts_form
from app.pipeline import brief as brief_mod, ingest, orchestrator
from app.storage import cleanup
from app.storage import jobs as job_store

router = APIRouter(prefix="/api")

_CHUNK = 8 * 1024 * 1024


def _load_job_or_404(job_id: str) -> dict:
    try:
        return job_store.load_job(job_id)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        raise HTTPException(404, "job not found") from None


def _save_upload(f: UploadFile, dest: Path) -> None:
    """Stream to disk, aborting past MAX_UPLOAD_MB so one client cannot fill
    the drive. The partial file is removed before the error propagates."""
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := f.file.read(_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        413,
                        f"{f.filename}: over the {config.MAX_UPLOAD_MB}MB limit",
                    )
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


@router.post("/jobs")
async def create_job(
    background: BackgroundTasks,
    files: list[UploadFile] | None = File(default=None),
    youtube_urls: str = Form(""),
    instruction: str = Form(""),
    brief: str = Form(""),
    font: str = Form("malgun"),
    voice: str = Form("auto"),
    source_edited: str = Form(""),
    quality: str = Form("fast"),
):
    """Upload 1~3 raw videos and/or paste YouTube URLs + a structured brief."""
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
    for i, f in enumerate(files):
        ext = Path(f.filename or "video.mp4").suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"unsupported format: {f.filename}")
        # ASCII-only path: Windows cp949 + macOS NFD names break ffmpeg stderr.
        dest = config.UPLOAD_DIR / f"{batch}_{i}{ext}"
        try:
            _save_upload(f, dest)
        except BaseException:
            cleanup.drop_paths(saved)   # don't strand the earlier files of this batch
            raise
        saved.append(str(dest))

    raw_brief: dict = {}
    brief_text = (brief or "").strip()
    if brief_text:
        try:
            parsed = json.loads(brief_text)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"brief is not JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise HTTPException(400, "brief must be an object")
        raw_brief = parsed
    if instruction.strip() and not raw_brief.get("notes"):
        raw_brief["notes"] = instruction
    brief_data = brief_mod.normalize(raw_brief)
    compiled = brief_mod.compile(brief_data)

    job = job_store.create_job(
        saved, instruction=compiled, source_urls=urls, brief=brief_data,
        font=font, voice=voice,
        source_edited=str(source_edited).lower() in ("1", "true", "on", "yes"),
        quality=quality,
    )
    background.add_task(orchestrator.run_job, job["id"])
    return {"job_id": job["id"], "stage": job["stage"]}


@router.get("/brief")
async def get_brief_presets():
    return brief_mod.presets_for_ui()


@router.get("/style")
async def get_style_presets():
    from app.pipeline import captions as captions_mod
    from app.pipeline import narrate, quality as quality_mod
    return {
        "fonts": captions_mod.available_fonts(),
        "voices": narrate.available_voices(),
        **quality_mod.for_ui(),
    }


@router.get("/jobs")
async def get_jobs():
    return [
        {"id": j["id"], "stage": j["stage"], "created_at": j["created_at"],
         "error": j["error"], "user_choice": j["user_choice"],
         "progress": j.get("progress")}
        for j in job_store.list_jobs()
    ]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    return _load_job_or_404(job_id)


@router.get("/jobs/{job_id}/video/{label}")
async def get_video(job_id: str, label: str):
    job = _load_job_or_404(job_id)
    raw = (job.get("outputs") or {}).get(label)
    if not raw:
        raise HTTPException(404, "video not found")
    # job.json is editable on disk: only serve files that live inside the job
    path = Path(raw).resolve()
    if not path.is_relative_to(job_store.job_dir(job_id)) or not path.is_file():
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
    job = _load_job_or_404(job_id)
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

    background.add_task(_learn)
    return {"ok": True, "rubric_version": current["version"]}


@router.get("/learning")
async def learning_status():
    """What the creator's picks have actually changed inside xlog."""
    return taste.status()


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
async def learn_reference(url: str = Form(...), notes: str = Form("")):
    """Give xlog a YouTube link whose editing style should be learned."""
    try:
        return reference_mod.learn_from_youtube(url, notes)
    except reference_mod.ReferenceError as e:
        raise HTTPException(400, str(e))


@router.get("/references")
async def get_references():
    return reference_mod.list_references()
