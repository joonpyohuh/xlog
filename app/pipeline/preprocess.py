"""Frame extraction for LLM analysis."""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from app import config


def extract_frames(video_path: Path, out_dir: Path, fps: float | None = None) -> list[dict]:
    """Sample frames at `fps`, downscaled for API cost. Returns
    [{"t": seconds, "path": jpg_path}, ...] in time order."""
    fps = fps or config.ANALYSIS_FPS
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "f_%06d.jpg"
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale='min({config.ANALYSIS_FRAME_LONG_EDGE},iw)':-2",
        "-q:v", "4",
        str(pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    frames = sorted(out_dir.glob("f_*.jpg"))
    # frame N (1-indexed) is sampled at (N - 0.5) / fps seconds (fps filter midpoint)
    return [
        {"t": round((i + 0.5) / fps, 2), "path": str(p)}
        for i, p in enumerate(frames)
    ]


def frame_to_b64(path: str | Path) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")


def chunk_frames(frames: list[dict], size: int | None = None) -> list[list[dict]]:
    size = size or config.MAX_FRAMES_PER_REQUEST
    return [frames[i:i + size] for i in range(0, len(frames), size)]
