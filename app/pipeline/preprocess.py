"""Frame extraction for LLM analysis."""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from app import config


def crop_filter(crop_bottom: float) -> str:
    """ffmpeg filter (with trailing comma) that lops off the bottom band."""
    r = max(0.0, min(0.4, float(crop_bottom or 0.0)))
    return f"crop=iw:ih*{1 - r:.3f}:0:0," if r > 0.01 else ""


def extract_frames(
    video_path: Path,
    out_dir: Path,
    fps: float | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    crop_bottom: float = 0.0,
    long_edge: int | None = None,
) -> list[dict]:
    """Sample frames at `fps`, optionally clipped to [start_sec, end_sec].
    `crop_bottom` drops that fraction of the frame height before sampling, so
    a pre-edited source's burned-in subtitles never reach the model.
    Returns [{"t": seconds (absolute), "path": jpg_path}, ...] in time order."""
    fps = fps or config.ANALYSIS_FPS
    out_dir.mkdir(parents=True, exist_ok=True)
    # Every caller keys out_dir by (video, window, fps), so a populated dir is
    # the same sample we would re-shell out for. Judge + cross-check share one.
    done = sorted(out_dir.glob("f_*.jpg"))
    if done:
        return _index(done, float(start_sec or 0.0), fps)
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
        "-vf", f"{crop_filter(crop_bottom)}fps={fps},"
               f"scale='min({long_edge or config.ANALYSIS_FRAME_LONG_EDGE},iw)':-2",
        "-q:v", "4",
        str(pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return _index(sorted(out_dir.glob("f_*.jpg")), origin, fps)


def _index(frames: list[Path], origin: float, fps: float) -> list[dict]:
    return [
        {"t": round(origin + (i + 0.5) / fps, 2), "path": str(p)}
        for i, p in enumerate(frames)
    ]


def sample_short(path: Path, label: str, tmp: Path) -> list[tuple[str, str]]:
    """Frames of one rendered short as [(tag, base64), ...] for a vision call.

    The judge and the independent cross-check score the same pixels, so they
    share one `tmp` and the second caller reuses the first one's extraction.
    """
    frames = extract_frames(path, tmp / f"judge_{label}", fps=config.JUDGE_FPS)
    if not frames:
        return []
    step = max(1, len(frames) // config.JUDGE_MAX_FRAMES)
    return [
        (f"[edit {label} @ {f['t']:.1f}s]", frame_to_b64(f["path"]))
        for f in frames[::step][: config.JUDGE_MAX_FRAMES]
    ]


def frame_to_b64(path: str | Path) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")


def chunk_frames(frames: list[dict], size: int | None = None) -> list[list[dict]]:
    size = size or config.MAX_FRAMES_PER_REQUEST
    return [frames[i:i + size] for i in range(0, len(frames), size)]


if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    d = tmp / "judge_A"
    d.mkdir()
    for i in range(4):
        (d / f"f_{i:06d}.jpg").write_bytes(b"\xff\xd8\xff")

    # reuse branch must not shell out to ffmpeg — a bogus path proves it
    got = extract_frames(Path("no-such-file.mp4"), d, fps=2.0)
    assert [f["t"] for f in got] == [0.25, 0.75, 1.25, 1.75], got
    assert extract_frames(Path("no-such-file.mp4"), d, fps=2.0, start_sec=10.0)[0]["t"] == 10.25

    pairs = sample_short(Path("no-such-file.mp4"), "A", tmp)
    assert len(pairs) == 4 and pairs[0][0].startswith("[edit A @"), pairs
    assert all(b64 for _, b64 in pairs)

    assert crop_filter(0.18) == "crop=iw:ih*0.820:0:0,"
    assert crop_filter(0) == "" and crop_filter(None) == ""
    assert len(chunk_frames([{"t": i} for i in range(50)])) == 3
    print("preprocess self-check ok")
