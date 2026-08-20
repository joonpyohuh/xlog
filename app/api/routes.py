"""HTTP API for the local xlog site."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from app import config
from app.evaluation import feedback, rubric as rubric_store, taste
from app.knowledge import reference as reference_mod
from app.knowledge import shorts_form
from app.pipeline import brief as brief_mod, ingest, index as index_mod, orchestrator
from app.pipeline import premiere as premiere_mod
from app.storage import cleanup
from app.storage import jobs as job_store
from app.storage import traces as traces_store

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
    editor: str = Form(""),
    consent: str = Form(""),
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

    editor_data: dict = {}
    if (editor or "").strip():
        try:
            parsed_ed = json.loads(editor)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"editor is not JSON: {e}") from e
        if not isinstance(parsed_ed, dict):
            raise HTTPException(400, "editor must be an object")
        editor_data = parsed_ed

    agreed = str(consent).lower() in ("1", "true", "on", "yes")
    if not agreed:
        raise HTTPException(
            400,
            "Consent is required to start.",
        )

    job = job_store.create_job(
        saved, instruction=compiled, source_urls=urls, brief=brief_data,
        font=font, voice=voice,
        source_edited=str(source_edited).lower() in ("1", "true", "on", "yes"),
        quality=quality,
        editor=editor_data,
        consent=True,
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
    """Pick which starting timeline to work from. Not a finished-short contest."""
    job = _load_job_or_404(job_id)
    if job["stage"] not in ("awaiting_evaluation", "done"):
        raise HTTPException(400, f"job is in stage '{job['stage']}'")
    labels = {v["label"] for v in job["variants"] or []}
    if choice not in labels:
        raise HTTPException(400, f"choice must be one of {sorted(labels)}")

    job["user_choice"] = choice
    job["user_comment"] = comment
    job_store.save_job(job)

    traces_store.record({
        "consent": bool(job.get("consent")),
        "job_id": job_id,
        "kind": "pick",
        "pick": choice,
        "comment": comment,
        "brief": job.get("brief") or {},
        "ai_plan": job.get("ai_plan") or traces_store.compact_plan(job.get("variants") or []),
        "tighten": job.get("tighten") or {},
    })

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
    return {"ok": True, "rubric_version": current["version"], "working": choice}


@router.post("/jobs/{job_id}/recut")
async def recut(
    job_id: str,
    background: BackgroundTasks,
    drops: str = Form("[]"),
):
    """Strike transcript spans and re-encode. No new LLM analysis."""
    job = _load_job_or_404(job_id)
    if job["stage"] not in ("awaiting_evaluation", "done"):
        raise HTTPException(400, f"job is in stage '{job['stage']}'")
    try:
        parsed = json.loads(drops or "[]")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"drops is not JSON: {e}") from e
    if not isinstance(parsed, list):
        raise HTTPException(400, "drops must be a list")
    job_store.set_stage(job, "rendering", detail="대본대로 다시 컷 대기")
    traces_store.record({
        "consent": bool(job.get("consent")),
        "job_id": job_id,
        "kind": "recut",
        "pick": job.get("user_choice") or "",
        "ai_plan": job.get("ai_plan") or [],
        "actions": [{"op": "drop_span", **d} for d in parsed if isinstance(d, dict)],
        "brief": job.get("brief") or {},
        "tighten": job.get("tighten") or {},
    })
    background.add_task(orchestrator.rerender, job_id, parsed)
    return {"ok": True, "job_id": job_id}


@router.post("/jobs/{job_id}/timeline")
async def save_timeline(
    job_id: str,
    background: BackgroundTasks,
    shots: str = Form("[]"),
    actions: str = Form("[]"),
    comment: str = Form(""),
    rerender: str = Form(""),
):
    """Human-edited starting timeline. This is the training pair."""
    job = _load_job_or_404(job_id)
    if job["stage"] not in ("awaiting_evaluation", "done", "rendering"):
        raise HTTPException(400, f"job is in stage '{job['stage']}'")
    try:
        parsed_shots = json.loads(shots or "[]")
        parsed_actions = json.loads(actions or "[]")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"timeline is not JSON: {e}") from e
    if not isinstance(parsed_shots, list):
        raise HTTPException(400, "shots must be a list")
    label = job.get("user_choice") or (job.get("variants") or [{}])[0].get("label")
    for v in job.get("variants") or []:
        if v.get("label") == label:
            v["shots"] = parsed_shots
            v["total_sec"] = round(sum(
                float(s.get("end_sec") or 0) - float(s.get("start_sec") or 0)
                for s in parsed_shots
            ), 2)
    human = traces_store.compact_plan(job.get("variants") or [], label)
    job["human_plan"] = human
    job["edit_actions"] = parsed_actions
    if comment:
        job["user_comment"] = comment
    traces_store.record({
        "consent": bool(job.get("consent")),
        "job_id": job_id,
        "kind": "edit",
        "pick": label,
        "comment": comment or job.get("user_comment") or "",
        "brief": job.get("brief") or {},
        "ai_plan": job.get("ai_plan") or [],
        "human_plan": human,
        "actions": parsed_actions,
        "tighten": job.get("tighten") or {},
    })
    want_render = str(rerender).lower() in ("1", "true", "on", "yes")
    if want_render:
        job_store.set_stage(job, "rendering", detail="고친 타임라인 미리보기")
        background.add_task(orchestrator.rerender, job_id, [])
    else:
        job_store.set_stage(job, "done", detail="시작 타임라인 저장됨")
    return {"ok": True, "shots": len(parsed_shots), "rerender": want_render}


@router.get("/jobs/{job_id}/timeline.json")
async def export_timeline(job_id: str):
    job = _load_job_or_404(job_id)
    label = job.get("user_choice")
    body = {
        "job_id": job_id,
        "brief": job.get("brief") or {},
        "pick": label,
        "ai_plan": job.get("ai_plan") or traces_store.compact_plan(job.get("variants") or []),
        "human_plan": job.get("human_plan") or traces_store.compact_plan(
            job.get("variants") or [], label,
        ),
        "actions": job.get("edit_actions") or [],
    }
    from fastapi.responses import JSONResponse
    return JSONResponse(
        body,
        headers={"Content-Disposition": f'attachment; filename="xlog_{job_id}_timeline.json"'},
    )


@router.post("/jobs/{job_id}/index")
async def save_index(
    job_id: str,
    actions: str = Form("[]"),
    comment: str = Form(""),
):
    """Keep / discard / handle / reorder. Source files stay intact."""
    job = _load_job_or_404(job_id)
    if job["stage"] not in ("awaiting_evaluation", "done"):
        raise HTTPException(400, f"job is in stage '{job['stage']}'")
    try:
        parsed = json.loads(actions or "[]")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"actions is not JSON: {e}") from e
    if not isinstance(parsed, list):
        raise HTTPException(400, "actions must be a list")
    footage = job.get("index") or {"clips": []}
    job["index"] = index_mod.apply_actions(footage, parsed)
    job["edit_actions"] = (job.get("edit_actions") or []) + parsed
    if comment:
        job["user_comment"] = comment
    traces_store.record({
        "consent": bool(job.get("consent")),
        "job_id": job_id,
        "kind": "index",
        "comment": comment or job.get("user_comment") or "",
        "brief": job.get("brief") or {},
        "ai_plan": job.get("ai_plan") or [],
        "human_plan": job["index"].get("clips") or [],
        "actions": parsed,
    })
    job_store.set_stage(job, "done", detail="인덱스 결정 저장. 원본은 그대로입니다")
    return {"ok": True, "clips": len(job["index"].get("clips") or [])}


@router.get("/jobs/{job_id}/premiere.xml")
async def export_premiere(job_id: str):
    job = _load_job_or_404(job_id)
    xml = premiere_mod.fcpxml(
        job.get("index") or {"clips": []},
        job.get("video_infos") or [],
        name=job_id,
    )
    return Response(
        xml,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="xlog_{job_id}.fcpxml"'},
    )


@router.get("/jobs/{job_id}/markers.csv")
async def export_markers(job_id: str):
    job = _load_job_or_404(job_id)
    csv = premiere_mod.marker_csv(job.get("index") or {"clips": []})
    return Response(
        csv,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="xlog_{job_id}_markers.csv"'},
    )


@router.get("/traces")
async def get_traces():
    return {"stats": traces_store.stats(), "recent": traces_store.list_traces(20)}


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
