"""Branding outro (requirement 3): black card, white 'directed by xlog'."""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app import config


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


def make_outro_clip(work_dir: Path) -> Path:
    """Render the outro card as a video clip matching the output profile
    (silent stereo AAC audio so concat works)."""
    png = make_outro_image(work_dir / "outro.png")
    clip = work_dir / "outro.mp4"
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-loop", "1", "-framerate", str(config.OUTPUT_FPS), "-i", str(png),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(config.OUTRO_DURATION_SEC),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(config.OUTPUT_FPS),
        "-c:a", "aac", "-shortest",
        str(clip),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return clip
