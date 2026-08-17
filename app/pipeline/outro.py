"""Branding outro: custom clip from data/credits/, else a black title card."""
from __future__ import annotations

import subprocess
import traceback
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app import config

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/georgiai.ttf",   # Georgia Italic
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def credit_source() -> Path | None:
    d = config.CREDITS_DIR
    if not d.is_dir():
        return None
    files = [
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    ]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def make_outro_image(out_path: Path) -> Path:
    w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    img = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(img)
    font = _load_font(size=int(w * 0.055))
    text = config.OUTRO_TEXT
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text(((w - tw) / 2, (h - th) / 2), text, fill="white", font=font)
    img.save(out_path)
    return out_path


def _has_audio(path: Path) -> bool:
    out = subprocess.run(
        [
            config.FFPROBE_BIN, "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0", str(path),
        ],
        **config.SUBPROCESS_TEXT,
    ).stdout or ""
    return bool(out.strip())


def _transcode_credit(src: Path, dest: Path, *, crf: str, preset: str) -> Path:
    """Match shot encode so concat -c copy does not break."""
    w, h, fps = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT, config.OUTPUT_FPS
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps},format=yuv420p"
    )
    cmd = [config.FFMPEG_BIN, "-y", "-v", "error", "-i", str(src)]
    if not _has_audio(src):
        cmd += [
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        ]
        audio_map = "1:a:0"
    else:
        audio_map = "0:a:0"
    cmd += [
        "-vf", vf, "-map", "0:v:0", "-map", audio_map,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", "-sn", "-dn",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dest


def _render_card(work_dir: Path, clip: Path, *, crf: str, preset: str) -> Path:
    png = make_outro_image(work_dir / "outro.png")
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-loop", "1", "-framerate", str(config.OUTPUT_FPS), "-i", str(png),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t", str(config.OUTRO_DURATION_SEC),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-r", str(config.OUTPUT_FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest",
        str(clip),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return clip


def make_outro_clip(
    work_dir: Path, crf: str | None = None, preset: str | None = None,
) -> Path:
    """Custom credits clip when present; otherwise the black title card."""
    clip = work_dir / "outro.mp4"
    crf = crf or config.RENDER_CRF
    preset = preset or config.RENDER_PRESET
    src = credit_source()
    if src:
        try:
            return _transcode_credit(src, clip, crf=crf, preset=preset)
        except Exception:
            traceback.print_exc()
            print(f"[outro] credit clip failed, falling back to title card: {src.name}")
    return _render_card(work_dir, clip, crf=crf, preset=preset)


if __name__ == "__main__":
    src = credit_source()
    assert src is not None, "put a video in data/credits/"
    assert src.suffix.lower() in _VIDEO_EXTS, src
    print("outro credit source:", src.name)
