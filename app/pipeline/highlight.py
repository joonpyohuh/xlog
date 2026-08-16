"""Core-moment extraction.

Pass 1: cheap 0.5 fps scan of the whole source.
Pass 2: the windows pass 1 marked as worth cutting are re-read at
REFINE_FPS (8 fps) so 1–2s beats are not skipped.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config
from app.llm import claude
from app.pipeline import preprocess, verify

_MOMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "description": {"type": "string"},
                    "mood": {"type": "string"},
                    "intensity": {"type": "integer"},
                    "hook_potential": {"type": "integer"},
                },
                "required": [
                    "start_sec", "end_sec", "description",
                    "mood", "intensity", "hook_potential",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "moments"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a short-form video editor's assistant. You are shown frames "
    "sampled from a raw, unedited video, each labelled with its timestamp. "
    "Identify the moments worth putting in a 30-60 second short: emotional "
    "peaks, actions, reveals, funny or surprising beats. Rate each moment's "
    "intensity (1-10) and hook_potential (1-10, how well it could open a "
    "short). Time ranges must come from the frame timestamps you saw."
)

_REFINE_SYSTEM = (
    "You previously flagged this time window as promising, but you only saw "
    "sparse frames. You now see the SAME window at ~8 fps. Tighten start/end "
    "to the actual 1–2s beat (or keep a longer range if the whole run is "
    "usable). Drop anything that looked good at 0.5 fps but is dead air here. "
    "Timestamps must match the frames you see."
)


def _analyze_chunk(
    video_path: Path,
    video_index: int,
    chunk: list[dict],
    *,
    system: str = _SYSTEM,
    note: str = "",
) -> dict:
    b64s = [preprocess.frame_to_b64(f["path"]) for f in chunk]
    ts = [f["t"] for f in chunk]
    result = claude.analyze_frames(
        system=system,
        prompt=(
            f"Video {video_index} ('{video_path.name}'). Frames from "
            f"{ts[0]:.1f}s to {ts[-1]:.1f}s follow, each labelled with its "
            f"timestamp. {note or 'Extract the key moments in this range.'}"
        ),
        frames_b64=b64s,
        timestamps=ts,
        schema=_MOMENTS_SCHEMA,
        effort=config.ANALYSIS_EFFORT,
    )
    result["moments"] = verify.cross_check_moments(result["moments"], b64s, ts)
    return result


def _score(m: dict) -> int:
    return int(m.get("intensity") or 0) + int(m.get("hook_potential") or 0)


def _candidate_windows(moments: list[dict], duration: float) -> list[tuple[float, float]]:
    """Merge high-value coarse moments into windows to re-scan densely."""
    hot = [
        m for m in moments
        if (m.get("intensity") or 0) >= config.REFINE_MIN_SCORE
        or (m.get("hook_potential") or 0) >= config.REFINE_MIN_SCORE
    ]
    if not hot:
        hot = sorted(moments, key=_score, reverse=True)[:4]
    padded: list[tuple[float, float, int]] = []
    for m in sorted(hot, key=lambda x: x["start_sec"]):
        lo = max(0.0, float(m["start_sec"]) - config.REFINE_PAD_SEC)
        hi = min(duration, float(m["end_sec"]) + config.REFINE_PAD_SEC)
        if hi - lo < 1.0:
            hi = min(duration, lo + 2.0)
        sc = _score(m)
        if padded and lo <= padded[-1][1] + 2.0:
            prev = padded[-1]
            padded[-1] = (prev[0], max(prev[1], hi), max(prev[2], sc))
        else:
            padded.append((lo, hi, sc))
    padded.sort(key=lambda w: w[2], reverse=True)
    return [(a, b) for a, b, _ in padded[: config.REFINE_MAX_WINDOWS]]


def _overlaps(m: dict, lo: float, hi: float) -> bool:
    return not (m["end_sec"] < lo or m["start_sec"] > hi)


def _refine_video(
    video_index: int,
    path: Path,
    duration: float,
    coarse: list[dict],
    work_dir: Path,
) -> list[dict]:
    windows = _candidate_windows(coarse, duration)
    if not windows:
        return coarse
    tasks = []
    for wi, (lo, hi) in enumerate(windows):
        frames = preprocess.extract_frames(
            path,
            work_dir / f"frames_{video_index}_dense_{wi}",
            fps=config.REFINE_FPS,
            start_sec=lo,
            end_sec=hi,
        )
        if not frames:
            continue
        for chunk in preprocess.chunk_frames(frames):
            tasks.append((lo, hi, chunk))
    if not tasks:
        return coarse

    def _run(item: tuple[float, float, list[dict]]) -> tuple[float, float, list[dict]]:
        lo, hi, chunk = item
        r = _analyze_chunk(
            path, video_index, chunk,
            system=_REFINE_SYSTEM,
            note="This is a dense re-scan of a candidate window. Tighten the beats.",
        )
        return lo, hi, r["moments"]

    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_LLM) as pool:
        dense_parts = list(pool.map(_run, tasks))

    kept = list(coarse)
    for lo, hi, densem in dense_parts:
        kept = [m for m in kept if not _overlaps(m, lo, hi)]
        for m in densem:
            m["refined"] = True
            m["start_sec"] = max(lo, min(m["start_sec"], hi))
            m["end_sec"] = max(lo, min(m["end_sec"], hi))
            if m["end_sec"] - m["start_sec"] >= 0.8:
                kept.append(m)
    kept.sort(key=lambda m: m["start_sec"])
    return kept


def analyze_all(video_infos: list[dict], work_dir: Path) -> dict:
    """Coarse scan of every source, then dense refine of the hot windows."""
    tasks: list[tuple[int, Path, list[dict]]] = []
    for i, info in enumerate(video_infos):
        path = Path(info["path"])
        frames = preprocess.extract_frames(path, work_dir / f"frames_{i}")
        for chunk in preprocess.chunk_frames(frames):
            tasks.append((i, path, chunk))

    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_LLM) as pool:
        results = list(
            pool.map(lambda t: (t[0], _analyze_chunk(t[1], t[0], t[2])), tasks)
        )

    per_video = []
    for i, info in enumerate(video_infos):
        coarse = [
            {**m, "video_index": i}
            for vi, r in results if vi == i
            for m in r["moments"]
        ]
        coarse.sort(key=lambda m: m["start_sec"])
        summary = " ".join(r["summary"] for vi, r in results if vi == i)
        refined = _refine_video(
            i, Path(info["path"]), info["duration_sec"], coarse, work_dir,
        )
        for m in refined:
            m["video_index"] = i
        per_video.append(
            {"video_index": i, "name": info["name"],
             "summary": summary, "moments": refined}
        )
    return {
        "videos": per_video,
        "moments": [m for v in per_video for m in v["moments"]],
    }
