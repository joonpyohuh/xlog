"""Drop bulky working files. Learned taste lives in xlog_memory.db."""
from __future__ import annotations

import shutil
from pathlib import Path

from app import config
from app.storage import jobs as job_store

_KEEP = {"job.json"}


def _rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def empty_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        _rm(child)


def drop_intermediates(work_dir: Path) -> None:
    """Frames, clips, concat lists — everything except shorts + job.json."""
    if not work_dir.exists():
        return
    for p in work_dir.glob("frames_*"):
        _rm(p)
    for p in work_dir.glob("clips_*"):
        _rm(p)
    for p in work_dir.glob("*_concat.txt"):
        _rm(p)


def purge_work_media(work_dir: Path) -> None:
    """Leave only job.json. Shorts, clips, frames, sources all go."""
    if not work_dir.exists():
        return
    for p in work_dir.iterdir():
        if p.name in _KEEP:
            continue
        _rm(p)


def drop_paths(paths: list[str] | None) -> None:
    for raw in paths or []:
        _rm(Path(raw))


def slim_finished_job(job: dict, *, keep_shorts: bool = False) -> None:
    """Drop source footage. Keep rendered shorts until the next job starts
    (`reclaim` wipes finished jobs fully) so A/B pick still shows on screen."""
    drop_paths(job.get("videos"))
    work = job_store.job_dir(job["id"])
    if keep_shorts:
        drop_intermediates(work)
        job["videos"] = []
        job["analysis"] = None
    else:
        purge_work_media(work)
        job["videos"] = []
        job["analysis"] = None
        job["outputs"] = {}
    job_store.save_job(job)


def after_reference(video: Path | None, frames_dir: Path | None) -> None:
    if video is not None:
        _rm(video)
    if frames_dir is not None:
        _rm(frames_dir)
    empty_dir(config.TMP_DIR)


def reclaim() -> None:
    """Wipe tmp/references and media from jobs that already finished."""
    empty_dir(config.TMP_DIR)
    empty_dir(config.REFERENCE_DIR)
    live_sources: set[str] = set()
    for job in job_store.list_jobs():
        if job.get("stage") in ("done", "failed"):
            slim_finished_job(job)
        else:
            live_sources.update(job.get("videos") or [])
    if config.UPLOAD_DIR.exists():
        for p in config.UPLOAD_DIR.iterdir():
            if str(p) not in live_sources:
                _rm(p)
