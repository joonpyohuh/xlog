"""End-to-end job runner, optimized for turnaround time.

ingest -> analyze (coarse then dense refine) -> write 2 shot plans
-> render A/B -> AI judge watches the rendered pixels -> await evaluation.
"""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app import config
from app.evaluation import judge
from app.evaluation import rubric as rubric_store
from app.llm.grok_client import research_trends
from app.pipeline import highlight, ingest, render, screenwriter, verify
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
            detail="YouTube 다운로드 중" if urls else "원본 길이·코덱 검사 중",
        )
        for i, url in enumerate(urls):
            job_store.set_stage(
                job, "ingesting",
                detail=f"YouTube 다운로드 {i + 1}/{len(urls)}",
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
            detail=f"원본 {nvid}개 검사 중 (큰 파일은 조금 걸립니다)",
            pct=15,
        )
        video_infos = ingest.validate_inputs([Path(p) for p in job["videos"]])
        mins = sum(v["duration_sec"] for v in video_infos) / 60
        job_store.set_stage(
            job, "analyzing",
            detail=f"프레임 추출 중 · 원본 {mins:.0f}분",
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
        job["analysis"] = analysis
        job_store.save_job(job)
        cleanup.drop_intermediates(work_dir)

        research = ""
        if screenwriter.will_research(instruction, analysis):
            job_store.set_stage(
                job, "writing_plans", detail="X에서 최신 감각 검색 중",
            )
            research = research_trends(
                instruction, screenwriter.moments_blurb(analysis),
            )
        job_store.set_stage(job, "writing_plans", detail="편집안 A/B 작성 중")
        variants = screenwriter.write_plans(
            analysis, instruction, research=research, quality=spec["id"],
        )
        variants = verify.validate_variants(variants, video_infos)
        font = job.get("font") or "malgun"
        voice = job.get("voice") or "auto"
        channel = (brief or {}).get("channel") or ""
        for v in variants:
            v["font"] = font
            v["voice"] = voice
            v["channel"] = channel
            v["source_edited"] = source_edited
            v["crf"] = spec["crf"]
            v["render_preset"] = spec["render_preset"]
        job["variants"] = variants
        job_store.save_job(job)

        job_store.set_stage(job, "rendering", detail="숏츠 인코딩 시작")
        summary = " ".join(v["summary"] for v in analysis["videos"])

        def _render(v: dict) -> tuple[str, str]:
            out = work_dir / f"short_{v['label']}.mp4"
            render.render_variant(v, video_infos, work_dir, out)
            return v["label"], str(out)

        outputs: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(variants)) as pool:
            futs = [pool.submit(_render, v) for v in variants]
            for i, fut in enumerate(as_completed(futs), 1):
                label, path = fut.result()
                outputs[label] = path
                job_store.set_stage(
                    job, "rendering",
                    detail=f"편집 {label} 인코딩 완료 ({i}/{len(variants)})",
                    pct=70 + int(16 * i / max(len(variants), 1)),
                )
        job["outputs"] = outputs
        job_store.save_job(job)
        cleanup.drop_intermediates(work_dir)

        job_store.set_stage(job, "judging", detail="렌더된 숏츠 심사 중")
        verdict = judge.judge_variants(
            variants, summary, instruction,
            outputs=job["outputs"], work_dir=work_dir,
        )
        job_store.set_stage(job, "judging", detail="교차 검증 중", pct=94)
        opinion = verify.second_opinion(
            variants, summary, instruction, rubric_store.rubric_as_prompt(),
            outputs=job["outputs"], work_dir=work_dir,
        )
        if opinion is not None:
            verdict["second_opinion"] = opinion
            verdict["models_agree"] = opinion.get("winner") == verdict["winner"]
        job["judge_verdict"] = verdict

        job_store.set_stage(
            job, "awaiting_evaluation", detail="A/B 중 골라 주세요",
        )
    except Exception as e:  # noqa: BLE001 — surface any stage failure on the job
        traceback.print_exc()
        job_store.set_stage(
            job, "failed",
            error=f"{type(e).__name__}: {e}",
            detail=str(e),
        )
        cleanup.slim_finished_job(job)
