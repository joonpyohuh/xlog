"""End-to-end job runner, optimized for turnaround time.

ingest -> analyze (coarse then dense refine) -> write 2 shot plans
-> render A/B -> AI judge watches the rendered pixels -> await evaluation.
"""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config
from app.evaluation import judge
from app.evaluation import rubric as rubric_store
from app.pipeline import highlight, ingest, render, screenwriter, verify
from app.storage import cleanup
from app.storage import jobs as job_store


def run_job(job_id: str) -> None:
    job = job_store.load_job(job_id)
    work_dir = job_store.job_dir(job_id)
    instruction = job.get("instruction", "")
    try:
        cleanup.reclaim()
        # 1. ingest / validate (1~3 videos, optional YouTube URLs)
        job_store.set_stage(job, "ingesting")
        for i, url in enumerate(job.get("source_urls") or []):
            dest = ingest.download_youtube(
                url, config.UPLOAD_DIR, stem=f"{job_id}_yt{i}"
            )
            job["videos"].append(str(dest))
            job_store.save_job(job)
        video_infos = ingest.validate_inputs([Path(p) for p in job["videos"]])

        # 2. core-moment extraction (chunks fan out concurrently)
        job_store.set_stage(job, "analyzing")
        analysis = highlight.analyze_all(video_infos, work_dir)
        job["analysis"] = analysis
        job_store.save_job(job)
        cleanup.drop_intermediates(work_dir)

        # 3. two competing shot plans (user instruction honored), then
        # deterministic validation: clamp ranges into real video bounds,
        # drop degenerate shots (code-level hallucination guard).
        job_store.set_stage(job, "writing_plans")
        variants = screenwriter.write_plans(analysis, instruction)
        variants = verify.validate_variants(variants, video_infos)
        job["variants"] = variants
        job_store.save_job(job)

        job_store.set_stage(job, "rendering")
        summary = " ".join(v["summary"] for v in analysis["videos"])

        def _render(v: dict) -> tuple[str, str]:
            out = work_dir / f"short_{v['label']}.mp4"
            render.render_variant(v, video_infos, work_dir, out)
            return v["label"], str(out)

        with ThreadPoolExecutor(max_workers=len(variants)) as pool:
            render_futures = [pool.submit(_render, v) for v in variants]
            job["outputs"] = dict(f.result() for f in render_futures)
        job_store.save_job(job)
        cleanup.drop_intermediates(work_dir)

        # Judge AFTER render so it watches the actual pixels.
        job_store.set_stage(job, "judging")
        verdict = judge.judge_variants(
            variants, summary, instruction,
            outputs=job["outputs"], work_dir=work_dir,
        )
        opinion = verify.second_opinion(
            variants, summary, instruction, rubric_store.rubric_as_prompt(),
            outputs=job["outputs"], work_dir=work_dir,
        )
        if opinion is not None:
            verdict["second_opinion"] = opinion
            verdict["models_agree"] = opinion.get("winner") == verdict["winner"]
        job["judge_verdict"] = verdict

        job_store.set_stage(job, "awaiting_evaluation")
    except Exception as e:  # noqa: BLE001 — surface any stage failure on the job
        traceback.print_exc()
        job_store.set_stage(job, "failed", error=f"{type(e).__name__}: {e}")
        cleanup.slim_finished_job(job)
