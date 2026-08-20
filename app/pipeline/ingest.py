"""Input validation (requirement 1): 1~3 raw video files or YouTube URLs."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
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


def _ytdlp_argv() -> list[str]:
    """Run yt-dlp via the pip module unless a real binary path is configured."""
    bin_ = (config.YTDLP_BIN or "").strip()
    if bin_ and (Path(bin_).exists() or Path(bin_).suffix or "/" in bin_ or "\\" in bin_):
        return [bin_]
    return [sys.executable, "-m", "yt_dlp"]


def _js_runtime_arg() -> list[str]:
    """YouTube extraction needs Node/Deno. Deno is default; this box has Node."""
    for name in ("deno", "node"):
        path = shutil.which(name)
        if path:
            return ["--js-runtimes", f"{name}:{path}"]
    return []


def _cookie_browsers() -> list[str]:
    raw = (config.YTDLP_COOKIES_FROM_BROWSER or "").strip().lower()
    if config.ON_VERCEL or raw in ("none", "off", "0"):
        return []
    return [b.strip() for b in raw.split(",") if b.strip()]


def _ytdlp_extra_args(browser: str | None = None) -> list[str]:
    args = [
        "--extractor-args",
        "youtube:player_client=ios,web,mweb,-android_sdkless",
        *_js_runtime_arg(),
    ]
    if browser:
        args += ["--cookies-from-browser", browser]
    return args


def _yt_blocked(err: str) -> bool:
    return any(
        s in err
        for s in (
            "Sign in to confirm",
            "403: Forbidden",
            "Could not copy",
            "cookie database",
            "Failed to decrypt",
        )
    )


def download_youtube(url: str, dest_dir: Path, stem: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"yt_{uuid.uuid4().hex[:8]}"
    out_tpl = dest_dir / f"{stem}.%(ext)s"
    last_err = ""
    attempts: list[str | None] = [None, *_cookie_browsers()]
    for browser in attempts:
        cmd = _ytdlp_argv() + _ytdlp_extra_args(browser) + [
            "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "-o", str(out_tpl),
            url,
        ]
        try:
            proc = subprocess.run(
                cmd, timeout=config.YTDLP_TIMEOUT_SEC, **config.SUBPROCESS_TEXT,
            )
        except FileNotFoundError as e:
            raise IngestError(
                "yt-dlp is not installed. pip install -r requirements.txt"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise IngestError(
                f"yt-dlp timed out after {config.YTDLP_TIMEOUT_SEC}s. "
                "Upload the source mp4 instead."
            ) from e
        if proc.returncode == 0:
            break
        last_err = (proc.stderr or "")[-800:]
        if _yt_blocked(last_err):
            continue
        if "JavaScript runtime" in last_err:
            raise IngestError(
                "YouTube needs Node or Deno for yt-dlp. Install Node, then retry."
            )
        raise IngestError(f"yt-dlp failed: {last_err}")
    else:
        raise IngestError(
            "YouTube blocked the download (bot check / locked browser cookies). "
            "Quit Chrome or Edge completely and retry, or upload the source mp4."
        )
    matches = list(dest_dir.glob(f"{stem}.*"))
    if not matches:
        raise IngestError("download produced no file")
    return matches[0]


def probe(path: Path) -> dict:
    """ffprobe metadata: duration, resolution, fps."""
    if not path.is_file():
        raise IngestError(f"{path.name}: file not found")
    if path.stat().st_size == 0:
        raise IngestError(f"{path.name}: file is empty (upload may have failed)")
    cmd = [
        config.FFPROBE_BIN, "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, **config.SUBPROCESS_TEXT)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        err = (proc.stderr or "").strip() or f"exit {proc.returncode}"
        raise IngestError(f"{path.name}: ffprobe failed — {err[:300]}")
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
            cap_min = config.SOURCE_MAX_SEC / 60
            dur_min = info["duration_sec"] / 60
            raise IngestError(
                f"{p.name}: video is {dur_min:.0f}min — source cap is {cap_min:.0f}min "
                f"(set XLOG_SOURCE_MAX_SEC in .env to raise)"
            )
        infos.append(info)
    return infos


if __name__ == "__main__":
    import subprocess
    import unicodedata

    upload = config.UPLOAD_DIR
    upload.mkdir(parents=True, exist_ok=True)
    src = upload / "_ingest_selfcheck.mp4"
    subprocess.run(
        [config.FFMPEG_BIN, "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
         "-c:v", "libx264", str(src)],
        check=True, capture_output=True,
    )
    for name in ("첫 브이로그.mov", unicodedata.normalize("NFD", "첫 브이로그") + ".mov"):
        dest = upload / name
        dest.write_bytes(src.read_bytes())
        info = probe(dest)
        assert info["duration_sec"] >= 1.0, info
        dest.unlink(missing_ok=True)
    src.unlink(missing_ok=True)
    print("ingest self-check ok")
