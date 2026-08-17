"""Core-moment extraction.

Preferred path: one Gemini call per source (native video). JPEG chunking
is only the fallback when the Files API rejects the file.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from app import config
from app.llm import gemini, grok
from app.pipeline import brief as brief_mod
from app.pipeline import preprocess
from app.pipeline import quality as quality_mod

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
                    "brief_fit": {"type": "integer"},
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
    "You are a senior editor watching a full source video. Use the actual "
    "timecode of the file (seconds from the start). Do not just flag pretty "
    "or busy shots. Infer what is happening: who is there, what changed, "
    "what is funny or at stake, and how beats connect into a story "
    "(setup → turn → payoff). Extract moments that carry that narrative. "
    "Rate intensity (1-10), hook_potential (1-10, could this open a short?), "
    "brief_fit (1-10). Describe each moment so a writer who cannot see the "
    "video could still cut the story. Prefer 8-16 moments spanning early, "
    "middle and late parts of the source."
)

_REFINE_SYSTEM = (
    "You previously flagged this time window as promising, but you only saw "
    "sparse frames. You now see the SAME window at ~2 fps. Tighten start/end "
    "to the actual 1–2s beat (or keep a longer range if the whole run is "
    "usable). Drop anything that looked good at 0.5 fps but is dead air here. "
    "Timestamps must match the frames you see."
)


def _normalize_moments(result: dict) -> dict:
    for m in result.get("moments") or []:
        if m.get("brief_fit") is None:
            m["brief_fit"] = min(10, max(1, int(m.get("hook_potential") or 5)))
    result.setdefault("moments", [])
    result.setdefault("summary", "")
    return result


def _analyze_chunk(
    video_path: Path,
    video_index: int,
    chunk: list[dict],
    *,
    system: str = _SYSTEM,
    note: str = "",
    effort: str = config.ANALYSIS_EFFORT,
) -> dict:
    b64s = [preprocess.frame_to_b64(f["path"]) for f in chunk]
    ts = [f["t"] for f in chunk]
    prompt = (
        f"Video {video_index} ('{video_path.name}'). Frames from "
        f"{ts[0]:.1f}s to {ts[-1]:.1f}s follow, each labelled with its "
        f"timestamp. {note or 'Extract the key moments in this range.'}"
    )
    if gemini.available():
        result = gemini.analyze_frames(
            system=system, prompt=prompt, frames_b64=b64s,
            timestamps=ts, schema=_MOMENTS_SCHEMA,
        )
    else:
        result = grok.analyze_frames(
            system=system, prompt=prompt, frames_b64=b64s,
            timestamps=ts, schema=_MOMENTS_SCHEMA, effort=effort,
        )
    return _normalize_moments(result)


def _analyze_video_native(
    path: Path,
    video_index: int,
    duration: float,
    hunt: str,
    *,
    fps: float = 1.0,
    media: str = "low",
) -> dict:
    prompt = (
        f"Video {video_index} ('{path.name}'), duration {duration:.1f}s. "
        "Watch the whole file. Extract the key narrative moments with "
        "start_sec/end_sec in seconds from the start of this file."
    )
    result = gemini.analyze_video(
        path, system=hunt, prompt=prompt, schema=_MOMENTS_SCHEMA,
        fps=fps, media=media,
    )
    return _normalize_moments(result)


def _score(m: dict) -> int:
    fit = int(m.get("brief_fit") or 0)
    if fit:
        return fit * 2 + int(m.get("hook_potential") or 0)
    return int(m.get("intensity") or 0) + int(m.get("hook_potential") or 0)


def _candidate_windows(moments: list[dict], duration: float) -> list[tuple[float, float]]:
    """Merge high-value coarse moments into windows to re-scan densely."""
    hot = [
        m for m in moments
        if (m.get("brief_fit") or 0) >= config.REFINE_MIN_SCORE
        or (m.get("intensity") or 0) >= config.REFINE_MIN_SCORE
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
    hunt_system: str,
    crop_bottom: float = 0.0,
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
            crop_bottom=crop_bottom,
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
            system=hunt_system or _REFINE_SYSTEM,
            note="This is a dense re-scan of a candidate window. Tighten the beats that match the creator's point.",
            effort=config.ANALYSIS_EFFORT,
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


def analyze_all(
    video_infos: list[dict],
    work_dir: Path,
    brief: dict | str | None = None,
    on_progress: Callable[[str, float], None] | None = None,
    source_edited: bool = False,
    quality: str = "fast",
) -> dict:
    """One Gemini pass per source when possible; JPEG chunks otherwise."""
    spec = quality_mod.resolve(quality)
    def ping(detail: str, frac: float) -> None:
        if on_progress:
            on_progress(detail, max(0.0, min(1.0, frac)))

    hunt = _SYSTEM + (brief_mod.hunt_system_addendum(brief) if brief else "")
    refine = _REFINE_SYSTEM + (brief_mod.hunt_system_addendum(brief) if brief else "")
    must_text = brief_mod.normalize(brief)["must"] if brief else ""
    crop_bottom = config.SOURCE_CAPTION_BAND if source_edited else 0.0
    if source_edited:
        hunt += (
            "\n\nThis source was already edited once and may have burned-in "
            "subtitles on the lower fifth of the frame. Ignore on-screen text; "
            "judge what happens in the picture."
        )

    per_video = []
    nvid = max(len(video_infos), 1)
    for i, info in enumerate(video_infos):
        path = Path(info["path"])
        dur = float(info["duration_sec"] or 1.0)
        ping(f"원본 파일 업로드 · AI 분석 {i + 1}/{nvid} · {dur / 60:.0f}분", 0.08 + 0.7 * i / nvid)
        result = None
        used_native = False
        if gemini.available():
            try:
                result = _analyze_video_native(
                    path, i, dur, hunt,
                    fps=spec["video_fps"], media=spec["gemini_media"],
                )
                used_native = True
            except Exception as e:
                print(f"[highlight] Gemini file path failed, JPEG fallback: {type(e).__name__}: {e}")
                result = None
        if result is None:
            fps = min(config.ANALYSIS_FPS, config.COARSE_MAX_FRAMES / dur)
            frames = preprocess.extract_frames(
                path, work_dir / f"frames_{i}", fps=fps, crop_bottom=crop_bottom,
                long_edge=config.COARSE_FRAME_LONG_EDGE,
            )
            parts = []
            for chunk in preprocess.chunk_frames(frames):
                parts.append(_analyze_chunk(path, i, chunk, system=hunt, effort=config.COARSE_EFFORT))
            result = {
                "summary": " ".join(p["summary"] for p in parts),
                "moments": [m for p in parts for m in p["moments"]],
            }
        coarse = [{**m, "video_index": i} for m in result["moments"]]
        coarse.sort(key=lambda m: m["start_sec"])
        brief_mod.apply_must(coarse, must_text)
        if used_native:
            refined = coarse
        else:
            ping(f"후보 구간 정밀 스캔 {i + 1}/{nvid}", 0.78 + 0.18 * i / nvid)
            refined = _refine_video(
                i, path, dur, coarse, work_dir, refine, crop_bottom=crop_bottom,
            )
        for m in refined:
            m["video_index"] = i
        per_video.append(
            {"video_index": i, "name": info["name"],
             "summary": result["summary"], "moments": refined}
        )
    ping("분석 완료", 1.0)
    moments = [m for v in per_video for m in v["moments"]]
    brief_mod.apply_must(moments, must_text)
    return {
        "videos": per_video,
        "moments": moments,
        "must_note": brief_mod.must_summary(moments),
    }


if __name__ == "__main__":
    # 14 min source must not explode into 400-frame / 50-per-call vision batches
    dur = 829.0
    fps = min(config.ANALYSIS_FPS, config.COARSE_MAX_FRAMES / dur)
    n = int(dur * fps)
    chunks = (n + config.MAX_FRAMES_PER_REQUEST - 1) // config.MAX_FRAMES_PER_REQUEST
    assert n <= config.COARSE_MAX_FRAMES + 1, n
    assert chunks <= 4, chunks
    print("highlight self-check ok", round(fps, 3), "fps", n, "frames", chunks, "calls")
