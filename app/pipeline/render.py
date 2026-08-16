"""ffmpeg rendering: cut shots, normalize to the output ratio, burn captions,
concat, append outro. Captions are overlaid in the same encode pass as the
cut (no extra re-encode) to keep turnaround time down."""
from __future__ import annotations

import subprocess
from pathlib import Path

from app import config
from app.pipeline import captions as captions_mod
from app.pipeline import outro as outro_mod

# Scale to fill then center-crop (content-aware crop is a TODO — CutClaw does
# subject-aware cropping via VLM; pilot version center-crops).
_W, _H, _FPS = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT, config.OUTPUT_FPS
_VF_BASE = (
    f"scale={_W}:{_H}:force_original_aspect_ratio=increase,"
    f"crop={_W}:{_H},fps={_FPS}"
)


def _vf(fx: str) -> str:
    if fx == "punch_in":
        return f"{_VF_BASE},scale=iw*1.18:ih*1.18,crop={_W}:{_H}"
    if fx == "zoom_in":
        # ponytail: zoompan is approximate; upgrade to per-frame affine if needed
        return (
            f"{_VF_BASE},"
            f"zoompan=z='min(1.16,1+0.0012*in)':d=1:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={_W}x{_H}:fps={_FPS}"
        )
    return _VF_BASE


def _cut_shot(
    src: Path,
    start: float,
    end: float,
    out_path: Path,
    has_audio: bool,
    caption_png: Path | None = None,
    fx: str = "none",
) -> Path:
    vf = _vf(fx)
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
    ]
    next_idx = 1
    audio_map = "0:a"
    if not has_audio:
        cmd += ["-f", "lavfi", "-t", f"{end - start:.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
        audio_map = f"{next_idx}:a"
        next_idx += 1

    if caption_png is not None:
        cmd += ["-i", str(caption_png)]
        fc = (
            f"[0:v]{vf}[base];"
            f"[base][{next_idx}:v]overlay=0:0,format=yuv420p[vout]"
        )
        cmd += ["-filter_complex", fc, "-map", "[vout]"]
    else:
        cmd += ["-vf", f"{vf},format=yuv420p", "-map", "0:v"]

    cmd += [
        "-map", audio_map,
        "-c:v", "libx264", "-preset", config.RENDER_PRESET, "-crf", config.RENDER_CRF,
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def _concat(clips: list[Path], out_path: Path, work_dir: Path) -> Path:
    list_file = work_dir / f"{out_path.stem}_concat.txt"
    # absolute paths: ffmpeg resolves relative entries against the list file's dir
    list_file.write_text(
        "".join(f"file '{c.resolve().as_posix()}'\n" for c in clips), encoding="utf-8"
    )
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def render_variant(
    variant: dict,
    video_infos: list[dict],
    work_dir: Path,
    out_path: Path,
) -> Path:
    """Render one shot plan into a final short (captions + branding outro)."""
    clip_dir = work_dir / f"clips_{variant['label']}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    clips: list[Path] = []
    for i, shot in enumerate(variant["shots"]):
        info = video_infos[shot["video_index"]]
        caption_png = None
        text = (shot.get("caption") or "").strip()
        if text and shot.get("caption_style", "none") != "none":
            caption_png = captions_mod.make_caption_png(
                text, shot["caption_style"], clip_dir / f"cap_{i:03d}.png"
            )
        clip = _cut_shot(
            Path(info["path"]),
            shot["start_sec"],
            shot["end_sec"],
            clip_dir / f"shot_{i:03d}.mp4",
            has_audio=info.get("has_audio", True),
            caption_png=caption_png,
            fx=shot.get("fx") or "none",
        )
        clips.append(clip)

    clips.append(outro_mod.make_outro_clip(clip_dir))
    return _concat(clips, out_path, work_dir)
