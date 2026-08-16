"""Frame extraction for LLM analysis."""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from app import config


def extract_frames(
    video_path: Path,
    out_dir: Path,
    fps: float | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> list[dict]:
    """Sample frames at `fps`, optionally clipped to [start_sec, end_sec].
    Returns [{"t": seconds (absolute), "path": jpg_path}, ...] in time order."""
    fps = fps or config.ANALYSIS_FPS
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "f_%06d.jpg"
    cmd = [config.FFMPEG_BIN, "-y", "-v", "error"]
    origin = 0.0
    if start_sec is not None:
        origin = float(start_sec)
        cmd += ["-ss", f"{origin:.3f}"]
    cmd += ["-i", str(video_path)]
    if end_sec is not None:
        dur = max(0.05, float(end_sec) - origin)
        cmd += ["-t", f"{dur:.3f}"]
    cmd += [
        "-vf", f"fps={fps},scale='min({config.ANALYSIS_FRAME_LONG_EDGE},iw)':-2",
        "-q:v", "4",
        str(pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    frames = sorted(out_dir.glob("f_*.jpg"))
    return [
        {"t": round(origin + (i + 0.5) / fps, 2), "path": str(p)}
        for i, p in enumerate(frames)
    ]


def frame_to_b64(path: str | Path) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")


def chunk_frames(frames: list[dict], size: int | None = None) -> list[list[dict]]:
    size = size or config.MAX_FRAMES_PER_REQUEST
    return [frames[i:i + size] for i in range(0, len(frames), size)]
