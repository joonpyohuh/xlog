"""ffmpeg rendering: cut shots, normalize to the output ratio, burn captions,
concat, append outro. Captions are overlaid in the same encode pass as the
cut (no extra re-encode) to keep turnaround time down."""
from __future__ import annotations

import shutil
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


def _anchor_xy(anchor: list[float] | None) -> tuple[str, str]:
    ax = min(0.88, max(0.12, float((anchor or [0.5, 0.42])[0])))
    ay = min(0.72, max(0.18, float((anchor or [0.5, 0.42])[1])))
    return f"(iw-{_W})*{ax:.3f}", f"(ih-{_H})*{ay:.3f}"


def _vf(
    fx: str,
    crop_bottom: float = 0.0,
    anchor: list[float] | None = None,
    scale: float = 1.10,
) -> str:
    pre = preprocess.crop_filter(crop_bottom)
    if pre:
        return pre + _vf(fx, 0.0, anchor, scale)
    base = _canvas()
    x, y = _anchor_xy(anchor)
    z = min(1.18, max(1.08, float(scale or 1.10)))
    if fx == "punch_in":
        return (
            f"{base},scale=iw*{z}:ih*{z}:flags=lanczos,"
            f"crop={_W}:{_H}:x='{x}':y='{y}'"
        )
    if fx == "zoom_in":
        ax = min(0.88, max(0.12, float((anchor or [0.5, 0.42])[0])))
        ay = min(0.72, max(0.18, float((anchor or [0.5, 0.42])[1])))
        return (
            f"{base},zoompan=z='min({z},1+0.0012*in)':d=1:"
            f"x='iw*{ax:.3f}-(iw/zoom/2)':y='ih*{ay:.3f}-(ih/zoom/2)':"
            f"s={_W}x{_H}:fps={_FPS}"
        )
    if fx == "zoom_out":
        return (
            f"{base},zoompan=z='max(1.0,{z}-0.0012*in)':d=1:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={_W}x{_H}:fps={_FPS}"
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


def _encode(
    src: Path,
    start: float,
    end: float,
    out_path: Path,
    has_audio: bool,
    caption_png: Path | None = None,
    sticker_png: Path | None = None,
    fx: str = "none",
    crop_bottom: float = 0.0,
    crf: str | None = None,
    preset: str | None = None,
    anchor: list[float] | None = None,
    scale: float = 1.10,
) -> Path:
    vf = _vf(fx, crop_bottom, anchor, scale)
    crf = crf or config.RENDER_CRF
    preset = preset or config.RENDER_PRESET
    dur = max(0.12, end - start)
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
    ]
    next_idx = 1
    audio_map = "0:a"
    if not has_audio:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        audio_map = f"{next_idx}:a"
        next_idx += 1

    overlays: list[tuple[int, str]] = []
    for png in (caption_png, sticker_png):
        if png is None:
            continue
        cmd += ["-loop", "1", "-framerate", str(_FPS), "-t", f"{dur:.3f}",
                "-i", str(png)]
        overlays.append((next_idx, "cap" if png == caption_png else "stk"))
        next_idx += 1

    if overlays:
        fc = [f"[0:v]{vf}[base]"]
        last = "base"
        for idx, tag in overlays:
            fade = "0.22" if tag == "cap" else "0.12"
            fc.append(
                f"[{idx}:v]format=rgba,fade=t=in:st=0:d={fade}:alpha=1[{tag}]"
            )
            pos = "0:0" if tag == "cap" else f"{int(_W * 0.62)}:{int(_H * 0.14)}"
            fc.append(f"[{last}][{tag}]overlay={pos}[o{tag}]")
            last = f"o{tag}"
        fc.append(f"[{last}]format=yuv420p[vout]")
        cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]"]
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


def _mix_sfx(video: Path, wav: Path, out_path: Path) -> Path:
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-i", str(video), "-i", str(wav),
        "-filter_complex",
        f"[1:a]volume={config.SFX_GAIN},afade=t=out:st=0.12:d=0.08,apad[fx];"
        f"[0:a]volume=1.0[dlg];"
        f"[dlg][fx]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
        "-map", "0:v", "-c:v", "copy",
        "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def _cut_shot(
    src: Path,
    shot: dict,
    out_path: Path,
    has_audio: bool,
    caption_png: Path | None = None,
    sticker_png: Path | None = None,
    crop_bottom: float = 0.0,
    crf: str | None = None,
    preset: str | None = None,
) -> Path:
    ranges = shot.get("ranges") or [
        (float(shot["start_sec"]), float(shot["end_sec"]))
    ]
    fx = shot.get("fx") or "none"
    kwargs = dict(
        has_audio=has_audio, crop_bottom=crop_bottom, crf=crf, preset=preset,
        anchor=shot.get("anchor"), scale=float(shot.get("zoom_scale") or 1.10),
    )
    if len(ranges) == 1:
        a, b = ranges[0]
        raw = _encode(src, a, b, out_path, caption_png=caption_png,
                      sticker_png=sticker_png, fx=fx, **kwargs)
    else:
        pieces = []
        for i, (a, b) in enumerate(ranges):
            p = out_path.with_name(f"{out_path.stem}_p{i}.mp4")
            _encode(src, a, b, p, caption_png=None, sticker_png=None,
                    fx=fx if i == 0 else "none", **kwargs)
            pieces.append(p)
        joined = out_path.with_name(out_path.stem + "_j.mp4")
        _concat(pieces, joined, out_path.parent)
        dur = sum(b - a for a, b in ranges)
        raw = _encode(
            joined, 0.0, dur, out_path, caption_png=caption_png,
            sticker_png=sticker_png, fx="none",
            has_audio=True, crop_bottom=0.0, crf=crf, preset=preset,
        )
    sfx = shot.get("sfx_path")
    if sfx and Path(sfx).is_file():
        mixed = out_path.with_name(out_path.stem + "_sfx.mp4")
        _mix_sfx(raw, Path(sfx), mixed)
        shutil.move(str(mixed), str(out_path))
        return out_path
    return raw


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
    theme = variant.get("caption_theme")
    safe = variant.get("safe")

    def _one(item: tuple[int, dict]) -> Path:
        i, shot = item
        info = video_infos[shot["video_index"]]
        caption_png = None
        text = (shot.get("caption") or "").strip()
        if text and shot.get("caption_style", "none") != "none":
            caption_png = captions_mod.make_caption_png(
                text, shot["caption_style"], clip_dir / f"cap_{i:03d}.png",
                font_id=variant.get("font") or "malgun",
                theme=theme, safe=safe,
            )
        sticker = shot.get("sticker")
        return _cut_shot(
            Path(info["path"]),
            shot,
            clip_dir / f"shot_{i:03d}.mp4",
            has_audio=info.get("has_audio", True),
            caption_png=caption_png,
            sticker_png=Path(sticker) if sticker else None,
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
    punch = _vf("punch_in", anchor=[0.3, 0.4], scale=1.1)
    assert "0.300" in punch and "crop=" in punch, punch
    print("render fit-canvas self-check ok")
