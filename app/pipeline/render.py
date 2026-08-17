"""ffmpeg rendering: cut shots, normalize to the output ratio, burn captions,
concat, append outro. Captions are overlaid in the same encode pass as the
cut (no extra re-encode) to keep turnaround time down."""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config
from app.pipeline import captions as captions_mod
from app.pipeline import narrate
from app.pipeline import outro as outro_mod
from app.pipeline import preprocess

# Scale to FIT the 9:16 canvas then pad (letterbox). Cover-crop used to
# slice the sides off 16:9 longform; the creator wants the whole frame.
_W, _H, _FPS = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT, config.OUTPUT_FPS


def _canvas() -> str:
    if config.OUTPUT_FIT:
        return (
            f"scale={_W}:{_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={_W}:{_H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={_FPS}"
        )
    return (
        f"scale={_W}:{_H}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={_W}:{_H},fps={_FPS}"
    )


_ZOOM_XY = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"


def _vf(fx: str, crop_bottom: float = 0.0) -> str:
    pre = preprocess.crop_filter(crop_bottom)
    if pre:
        return pre + _vf(fx)
    base = _canvas()
    if fx == "punch_in":
        return f"{base},scale=iw*1.18:ih*1.18:flags=lanczos,crop={_W}:{_H}"
    if fx == "zoom_in":
        # ponytail: zoompan is approximate; upgrade to per-frame affine if needed
        return (
            f"{base},zoompan=z='min(1.16,1+0.0012*in)':d=1:"
            f"{_ZOOM_XY}:s={_W}x{_H}:fps={_FPS}"
        )
    if fx == "zoom_out":
        return (
            f"{base},zoompan=z='max(1.0,1.16-0.0012*in)':d=1:"
            f"{_ZOOM_XY}:s={_W}x{_H}:fps={_FPS}"
        )
    if fx == "shake":
        pad = int(_W * 0.03)
        return (
            f"{base},scale=iw*1.08:ih*1.08:flags=lanczos,"
            f"crop={_W}:{_H}:x='(iw-{_W})/2+{pad}*sin(n/2.4)'"
            f":y='(ih-{_H})/2+{pad}*cos(n/3.1)'"
        )
    if fx == "flash":
        return f"{base},fade=t=in:st=0:d=0.12:color=white"
    if fx == "whip":
        return f"{base},boxblur=luma_radius=24:luma_power=1:enable='lt(t,0.14)'"
    return base


def _cut_shot(
    src: Path,
    start: float,
    end: float,
    out_path: Path,
    has_audio: bool,
    caption_png: Path | None = None,
    fx: str = "none",
    crop_bottom: float = 0.0,
    crf: str | None = None,
    preset: str | None = None,
) -> Path:
    vf = _vf(fx, crop_bottom)
    crf = crf or config.RENDER_CRF
    preset = preset or config.RENDER_PRESET
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
    ]
    next_idx = 1
    audio_map = "0:a"
    if not has_audio:
        cmd += ["-f", "lavfi", "-t", f"{end - start:.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        audio_map = f"{next_idx}:a"
        next_idx += 1

    if caption_png is not None:
        # loop the still into real frames, otherwise the fade below lands on
        # the single frame at t=0 and overlay repeats it fully transparent
        cmd += ["-loop", "1", "-framerate", str(_FPS), "-t", f"{end - start:.3f}",
                "-i", str(caption_png)]
        fc = (
            f"[0:v]{vf}[base];"
            # captions ease in instead of snapping on — the polished look
            f"[{next_idx}:v]format=rgba,fade=t=in:st=0:d=0.22:alpha=1[cap];"
            f"[base][cap]overlay=0:0,format=yuv420p[vout]"
        )
        cmd += ["-filter_complex", fc, "-map", "[vout]"]
    else:
        cmd += ["-vf", f"{vf},format=yuv420p", "-map", "0:v"]

    cmd += [
        "-map", audio_map,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-x264-params", "ref=3",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
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
        # moov atom up front, otherwise a browser <video> stalls on first play
        "-movflags", "+faststart",
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
    crop_bottom = (
        config.SOURCE_CAPTION_BAND if variant.get("source_edited") else 0.0
    )
    crf = str(variant.get("crf") or config.RENDER_CRF)
    preset = variant.get("render_preset") or config.RENDER_PRESET

    def _one(item: tuple[int, dict]) -> Path:
        i, shot = item
        info = video_infos[shot["video_index"]]
        caption_png = None
        text = (shot.get("caption") or "").strip()
        if text and shot.get("caption_style", "none") != "none":
            caption_png = captions_mod.make_caption_png(
                text, shot["caption_style"], clip_dir / f"cap_{i:03d}.png",
                font_id=variant.get("font") or "malgun",
            )
        return _cut_shot(
            Path(info["path"]),
            shot["start_sec"],
            shot["end_sec"],
            clip_dir / f"shot_{i:03d}.mp4",
            has_audio=info.get("has_audio", True),
            caption_png=caption_png,
            fx=shot.get("fx") or "none",
            crop_bottom=crop_bottom,
            crf=crf,
            preset=preset,
        )

    shots = list(enumerate(variant["shots"]))
    with ThreadPoolExecutor(max_workers=config.RENDER_WORKERS) as pool:
        outro = pool.submit(
            outro_mod.make_outro_clip, clip_dir, crf=crf, preset=preset,
        )
        clips = list(pool.map(_one, shots))
        clips.append(outro.result())
    silent = _concat(clips, clip_dir / f"cut_{variant['label']}.mp4", work_dir)
    return narrate.add_narration(silent, variant, clip_dir / "voice", out_path)


if __name__ == "__main__":
    vf = _canvas()
    assert "decrease" in vf and "pad=" in vf, vf
    assert "crop=" not in vf, vf
    print("render fit-canvas self-check ok")
