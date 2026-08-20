"""End-to-end job runner, optimized for turnaround time.

ingest -> analyze (coarse then dense refine) -> write 2 shot plans
-> render A/B -> AI judge watches the rendered pixels -> await evaluation.
"""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config
from app.pipeline import highlight, ingest, index as index_mod, polish, render, tighten
from app.pipeline import quality as quality_mod
from app.storage import cleanup
from app.storage import jobs as job_store


def run_job(job_id: str) -> None:
    job = job_store.load_job(job_id)
    work_dir = job_store.job_dir(job_id)
    instruction = job.get("instruction", "")
    brief = job.get("brief") or {}
    try:
        cleanup.reclaim()
        urls = job.get("source_urls") or []
        job_store.set_stage(
            job, "ingesting",
            detail="Downloading from YouTube" if urls else "Checking duration and codec",
        )
        for i, url in enumerate(urls):
            job_store.set_stage(
                job, "ingesting",
                detail=f"YouTube download {i + 1}/{len(urls)}",
                pct=8 + int(8 * i / max(len(urls), 1)),
            )
            dest = ingest.download_youtube(
                url, config.UPLOAD_DIR, stem=f"{job_id}_yt{i}"
            )
            job["videos"].append(str(dest))
            job_store.save_job(job)
        nvid = len(job["videos"])
        job_store.set_stage(
            job, "ingesting",
            detail=f"Checking {nvid} file(s) — large files take a minute",
            pct=15,
        )
        video_infos = ingest.validate_inputs([Path(p) for p in job["videos"]])
        flags = tighten.editor_flags(job)
        if flags["silence"]:
            job_store.set_stage(
                job, "ingesting",
                detail="Detecting silence (ffmpeg, no API cost)",
                pct=16,
            )
            tighten.attach_silence(video_infos)
        job["video_infos"] = _slim_infos(video_infos)
        job_store.save_job(job)
        mins = sum(v["duration_sec"] for v in video_infos) / 60
        job_store.set_stage(
            job, "analyzing",
            detail=f"Pulling frames · {mins:.0f} min of footage",
        )

        def _analyze_progress(detail: str, frac: float) -> None:
            job_store.set_stage(
                job, "analyzing",
                detail=detail,
                pct=18 + int(40 * max(0.0, min(1.0, frac))),
            )

        source_edited = bool(job.get("source_edited"))
        spec = quality_mod.resolve(job.get("quality"))
        analysis = highlight.analyze_all(
            video_infos, work_dir, brief=brief or instruction,
            on_progress=_analyze_progress,
            source_edited=source_edited,
            quality=spec["id"],
        )
        theme = {}
        if flags["theme"]:
            jpgs = list(work_dir.glob("**/f_*.jpg"))[:8]
            theme = polish.sample_theme(jpgs)
        job_store.set_stage(job, "analyzing", detail="Indexing scenes, lines, action", pct=88)
        footage = index_mod.build(analysis, video_infos)
        job["analysis"] = analysis
        job["caption_theme"] = theme
        job["index"] = footage
        job["ai_plan"] = footage.get("clips") or []
        job["video_infos"] = _slim_infos(video_infos)
        job_store.save_job(job)
        cleanup.drop_intermediates(work_dir)
        job_store.set_stage(
            job, "awaiting_evaluation",
            detail="Index ready. Source is untouched — keep, dump, or open the handles",
        )
    except Exception as e:  # noqa: BLE001 — surface any stage failure on the job
        traceback.print_exc()
        job_store.set_stage(
            job, "failed",
            error=f"{type(e).__name__}: {e}",
            detail=str(e),
        )
        cleanup.slim_finished_job(job)


def _slim_infos(infos: list[dict]) -> list[dict]:
    keys = (
        "path", "name", "duration_sec", "has_audio", "silences", "keeps",
        "saved_silence_sec", "filler_n",
    )
    return [{k: i.get(k) for k in keys if k in i or k in ("path", "duration_sec")} for i in infos]


def _apply_polish(
    variants: list[dict],
    infos: list[dict],
    theme: dict,
    flags: dict,
    work: Path,
) -> None:
    if flags.get("theme") and theme:
        polish.lock_caption_styles(variants, theme)
    if flags.get("zoom"):
        polish.apply_motion(variants, infos, work)
    if flags.get("sfx"):
        polish.assign_sfx(variants)
        kit = polish.ensure_kit(work)
        for v in variants:
            for s in v.get("shots") or []:
                name = s.get("sfx")
                if name and name in kit:
                    s["sfx_path"] = str(kit[name])
    if flags.get("stickers"):
        polish.place_stickers(variants, theme, work)
    if flags.get("qa"):
        polish.clamp_caption_layout(variants)


def rerender(job_id: str, drops: list[dict] | None = None) -> None:
    """Re-cut from the transcript without another LLM pass."""
    job = job_store.load_job(job_id)
    work_dir = job_store.job_dir(job_id)
    try:
        infos = job.get("video_infos") or ingest.validate_inputs(
            [Path(p) for p in job["videos"]]
        )
        flags = tighten.editor_flags(job)
        if drops:
            tighten.apply_drops(infos, drops)
            payload = job.get("transcript") or {}
            words = list(payload.get("words") or [])
            for w in words:
                for d in drops:
                    if int(w.get("video_index") or 0) != int(d.get("video_index") or 0):
                        continue
                    if float(w["t1"]) < float(d["start_sec"]) or float(w["t0"]) > float(d["end_sec"]):
                        continue
                    w["keep"] = False
            job["transcript"] = {**payload, "words": words}
        variants = job.get("variants") or []
        tighten.snap_variants(variants, infos)
        _apply_polish(variants, infos, job.get("caption_theme") or {}, flags, work_dir)
        job["variants"] = variants
        job["video_infos"] = _slim_infos(infos)
        job_store.set_stage(job, "rendering", detail="Recutting from the transcript")

        def _render(v: dict) -> tuple[str, str]:
            out = work_dir / f"short_{v['label']}.mp4"
            render.render_variant(v, infos, work_dir, out)
            return v["label"], str(out)

        outputs: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(variants) or 1) as pool:
            for label, path in pool.map(lambda v: _render(v), variants):
                outputs[label] = path
        job["outputs"] = outputs
        nxt = "done" if job.get("user_choice") else "awaiting_evaluation"
        job_store.set_stage(job, nxt, detail="Check the recut")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job_store.set_stage(
            job, "failed",
            error=f"{type(e).__name__}: {e}",
            detail=str(e),
        )

