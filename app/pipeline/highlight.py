"""Core-moment extraction (requirement 2).

Sends sampled frames to Claude and gets back a list of "moments":
time ranges with descriptions, emotional intensity, and hook potential.
These are the raw material the screenwriter builds shorts from.
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
                    "intensity": {"type": "integer"},   # 1-10
                    "hook_potential": {"type": "integer"},  # 1-10
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


def _analyze_chunk(video_path: Path, video_index: int, chunk: list[dict]) -> dict:
    b64s = [preprocess.frame_to_b64(f["path"]) for f in chunk]
    ts = [f["t"] for f in chunk]
    result = claude.analyze_frames(
        system=_SYSTEM,
        prompt=(
            f"Video {video_index} ('{video_path.name}'). Frames from "
            f"{ts[0]:.1f}s to {ts[-1]:.1f}s follow, each labelled with its "
            "timestamp. Extract the key moments in this range."
        ),
        frames_b64=b64s,
        timestamps=ts,
        schema=_MOMENTS_SCHEMA,
        effort=config.ANALYSIS_EFFORT,   # low effort: extraction is speed-critical
    )
    # Hallucination guard: GPT independently re-checks the claimed moments
    # against the SAME frames; rejected moments never reach the screenwriter.
    result["moments"] = verify.cross_check_moments(result["moments"], b64s, ts)
    return result


def analyze_all(video_infos: list[dict], work_dir: Path) -> dict:
    """Analyze every source video (1~3). Frame chunks across ALL videos are
    analyzed concurrently (MAX_PARALLEL_LLM) to minimize turnaround time."""
    # 1. extract frames per video (ffmpeg, fast) and build the chunk task list
    tasks: list[tuple[int, Path, list[dict]]] = []
    for i, info in enumerate(video_infos):
        path = Path(info["path"])
        frames = preprocess.extract_frames(path, work_dir / f"frames_{i}")
        for chunk in preprocess.chunk_frames(frames):
            tasks.append((i, path, chunk))

    # 2. fan out vision calls
    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_LLM) as pool:
        results = list(
            pool.map(lambda t: (t[0], _analyze_chunk(t[1], t[0], t[2])), tasks)
        )

    # 3. regroup per video
    per_video = []
    for i, info in enumerate(video_infos):
        moments = [
            {**m, "video_index": i}
            for vi, r in results if vi == i
            for m in r["moments"]
        ]
        moments.sort(key=lambda m: m["start_sec"])
        summary = " ".join(r["summary"] for vi, r in results if vi == i)
        per_video.append(
            {"video_index": i, "name": info["name"],
             "summary": summary, "moments": moments}
        )
    return {
        "videos": per_video,
        "moments": [m for v in per_video for m in v["moments"]],
    }
