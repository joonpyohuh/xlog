"""Input validation (requirement 1): 1~3 raw video files or YouTube URLs."""
from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path

from app import config


class IngestError(ValueError):
    pass


_YT_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu\.be/)[^\s,;]+",
    re.I,
)


def parse_youtube_urls(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    found = _YT_RE.findall(text)
    leftover = _YT_RE.sub(" ", text).strip()
    leftover = re.sub(r"[\s,;]+", " ", leftover).strip()
    if leftover:
        raise IngestError(f"not a YouTube URL: {leftover[:80]}")
    out, seen = [], set()
    for u in found:
        if not u.lower().startswith("http"):
            u = "https://" + u
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def download_youtube(url: str, dest_dir: Path, stem: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"yt_{uuid.uuid4().hex[:8]}"
    out_tpl = dest_dir / f"{stem}.%(ext)s"
    cmd = [
        config.YTDLP_BIN,
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", str(out_tpl),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise IngestError(f"yt-dlp failed: {(proc.stderr or '')[-400:]}")
    matches = list(dest_dir.glob(f"{stem}.*"))
    if not matches:
        raise IngestError("download produced no file")
    return matches[0]


def probe(path: Path) -> dict:
    """ffprobe metadata: duration, resolution, fps."""
    cmd = [
        config.FFPROBE_BIN, "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    meta = json.loads(out)
    vstream = next(
        (s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if vstream is None:
        raise IngestError(f"{path.name}: no video stream found")
    duration = float(meta["format"].get("duration", 0.0))
    return {
        "path": str(path),
        "name": path.name,
        "duration_sec": duration,
        "width": vstream.get("width"),
        "height": vstream.get("height"),
        "has_audio": any(
            s.get("codec_type") == "audio" for s in meta.get("streams", [])
        ),
    }


def validate_inputs(paths: list[Path]) -> list[dict]:
    if not (config.MIN_VIDEOS <= len(paths) <= config.MAX_VIDEOS):
        raise IngestError(
            f"video count must be between {config.MIN_VIDEOS} and "
            f"{config.MAX_VIDEOS}, got {len(paths)}"
        )
    infos = []
    for p in paths:
        if p.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            raise IngestError(f"unsupported format: {p.name}")
        info = probe(p)
        if info["duration_sec"] < config.SHORT_MIN_SEC:
            raise IngestError(
                f"{p.name}: video is shorter ({info['duration_sec']:.0f}s) than "
                f"the minimum short length ({config.SHORT_MIN_SEC}s)"
            )
        if info["duration_sec"] > config.SOURCE_MAX_SEC:
            raise IngestError(
                f"{p.name}: video is {info['duration_sec']:.0f}s — source cap is "
                f"{config.SOURCE_MAX_SEC}s"
            )
        infos.append(info)
    return infos
