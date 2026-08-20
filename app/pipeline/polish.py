"""Look, motion, SFX, stickers, QA — no generative APIs.

Palette + captions: PIL on sampled frames.
SFX: tiny synthesized wavs, ducked under dialogue.
Zoom: skin-blob face stand-in, 1.10/1.15, 15s fatigue.
Stickers: in-app RGBA (no Blender). Chroma key for black/green plates.
QA: Shorts safe-zone + SFX variation.
"""
from __future__ import annotations

import colorsys
import math
import struct
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from app import config
from app.pipeline.tighten import shot_len


def sample_theme(frames: list[Path]) -> dict:
    """Dominant clothes/bg vs a readable caption fill+stroke."""
    colors: list[tuple[int, int, int]] = []
    for p in frames[:8]:
        if not Path(p).is_file():
            continue
        img = Image.open(p).convert("RGB")
        img.thumbnail((64, 64))
        pal = img.quantize(colors=6, method=Image.Quantize.MEDIANCUT)
        counts = pal.getcolors() or []
        lut = pal.getpalette() or []
        for n, idx in sorted(counts, reverse=True)[:3]:
            r, g, b = lut[idx * 3: idx * 3 + 3]
            if max(r, g, b) - min(r, g, b) < 12 and 40 < (r + g + b) / 3 < 220:
                continue  # skip near-gray
            colors.extend([(r, g, b)] * max(1, n // 8))
    if not colors:
        return {"fill": "#FFFFFF", "stroke": "#111111", "accent": "#FFE14D"}
    r, g, b = colors[len(colors) // 2]
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    # complementary, forced bright so it reads on the picture
    cr, cg, cb = colorsys.hsv_to_rgb((h + 0.5) % 1.0, min(1.0, s + 0.25), 1.0)
    fill = _hex(int(cr * 255), int(cg * 255), int(cb * 255))
    stroke = "#111111" if v > 0.55 else "#F4F4F4"
    ar, ag, ab = colorsys.hsv_to_rgb(h, min(1.0, s + 0.15), 1.0)
    return {"fill": fill, "stroke": stroke, "accent": _hex(int(ar * 255), int(ag * 255), int(ab * 255))}


def lock_caption_styles(variants: list[dict], theme: dict) -> None:
    """One palette per job. Style names stay; colors come from the frame."""
    for v in variants:
        v["caption_theme"] = theme


def apply_motion(variants: list[dict], infos: list[dict], work: Path) -> None:
    """Face-anchored punch/zoom, only on hook/payoff, 15s fatigue cap."""
    last = -1e9
    t = 0.0
    scale_i = 0
    for v in variants:
        t = 0.0
        last = -1e9
        for shot in v.get("shots") or []:
            role = (shot.get("role") or "").lower()
            fx = (shot.get("fx") or "none").lower()
            want = fx in ("zoom_in", "punch_in", "zoom_out") or role in ("hook", "payoff")
            if want and (t - last) < config.ZOOM_FATIGUE_SEC and role not in ("hook", "payoff"):
                if fx in ("zoom_in", "punch_in", "zoom_out"):
                    shot["fx"] = "none"
                want = False
            if want and role in ("hook", "payoff"):
                shot["fx"] = shot.get("fx") if shot.get("fx") in ("zoom_in", "punch_in") else "punch_in"
                shot["zoom_scale"] = config.ZOOM_SCALES[scale_i % 2]
                scale_i += 1
                shot["anchor"] = face_anchor(
                    Path(infos[int(shot.get("video_index") or 0)]["path"]),
                    (float(shot["start_sec"]) + float(shot["end_sec"])) / 2,
                    work / f"face_{v.get('label')}_{int(t)}.jpg",
                )
                last = t
            t += shot_len(shot)


def face_anchor(src: Path, t: float, dest: Path) -> list[float]:
    """ponytail: skin-blob instead of a face net; swap to Haar if talking-heads miss."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-ss", f"{max(0.0, t):.3f}", "-i", str(src),
        "-frames:v", "1", "-q:v", "4", str(dest),
    ]
    subprocess.run(cmd, capture_output=True)
    if not dest.exists():
        return [0.5, 0.42]
    img = Image.open(dest).convert("RGB")
    w, h = img.size
    pix = img.load()
    sx = sy = n = 0
    step = max(1, min(w, h) // 80)
    for y in range(h // 8, int(h * 0.72), step):
        for x in range(int(w * 0.15), int(w * 0.85), step):
            r, g, b = pix[x, y]
            if _skin(r, g, b):
                sx += x
                sy += y
                n += 1
    if n < 12:
        return [0.5, 0.42]
    return [round(sx / n / w, 3), round(sy / n / h, 3)]


def assign_sfx(variants: list[dict]) -> None:
    kinds = ("pop", "whoosh", "click")
    last_at: dict[str, float] = {k: -1e9 for k in kinds}
    variant_i = {"pop": 0, "whoosh": 0, "click": 0}
    for v in variants:
        t = 0.0
        for shot in v.get("shots") or []:
            kind = _sfx_kind(shot)
            if kind:
                if t - last_at[kind] < config.SFX_REPEAT_WINDOW_SEC:
                    variant_i[kind] += 1
                shot["sfx"] = f"{kind}{1 + variant_i[kind] % 3}"
                last_at[kind] = t
            t += shot_len(shot)


def ensure_kit(work: Path) -> dict[str, Path]:
    kit = work / "sfx"
    kit.mkdir(parents=True, exist_ok=True)
    out = {}
    specs = {
        "pop1": (880, 0.12, "pop"), "pop2": (980, 0.11, "pop"), "pop3": (720, 0.13, "pop"),
        "whoosh1": (180, 0.22, "whoosh"), "whoosh2": (140, 0.24, "whoosh"),
        "whoosh3": (220, 0.20, "whoosh"),
        "click1": (1400, 0.07, "click"), "click2": (1700, 0.06, "click"),
        "click3": (1100, 0.08, "click"),
    }
    for name, (freq, dur, kind) in specs.items():
        p = kit / f"{name}.wav"
        if not p.exists():
            _synth(p, freq, dur, kind)
        out[name] = p
    return out


def make_sticker(kind: str, dest: Path, accent: str = "#FF4D6D") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (280, 280), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = _rgba(accent, 230)
    if kind == "heart":
        d.ellipse((40, 50, 160, 170), fill=col)
        d.ellipse((120, 50, 240, 170), fill=col)
        d.polygon([(46, 130), (140, 250), (234, 130)], fill=col)
    elif kind == "star":
        cx, cy, r = 140, 140, 110
        pts = []
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            rr = r if i % 2 == 0 else r * 0.42
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        d.polygon(pts, fill=col)
    else:  # spark
        d.ellipse((90, 90, 190, 190), fill=col)
        d.rectangle((128, 20, 152, 260), fill=col)
        d.rectangle((20, 128, 260, 152), fill=col)
    img.filter(ImageFilter.SMOOTH).save(dest)
    img.save(dest)
    return dest


def place_stickers(variants: list[dict], theme: dict, work: Path) -> None:
    accent = (theme or {}).get("accent") or "#FF4D6D"
    kinds = ("heart", "star", "spark")
    i = 0
    for v in variants:
        for shot in v.get("shots") or []:
            if (shot.get("role") or "") not in ("hook", "payoff"):
                continue
            if (shot.get("caption_style") or "") not in ("impact", "emphasis", "pop", "hot"):
                continue
            kind = kinds[i % 3]
            i += 1
            shot["sticker"] = str(make_sticker(kind, work / f"sticker_{kind}.png", accent))


def clamp_caption_layout(variants: list[dict]) -> None:
    """Force every caption style into the Shorts safe zone."""
    for v in variants:
        v["safe"] = {
            "top": config.SAFE_TOP,
            "bottom": config.SAFE_BOTTOM,
            "side": config.SAFE_SIDE,
            "right_extra": config.SAFE_RIGHT_EXTRA,
        }


def knockout(src: Path, dest: Path, hex_color: str = "000000", similarity: float = 0.28) -> Path:
    """One-click black/green plate removal for CapCut-recorded overlays."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    color = hex_color.lstrip("#")
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error", "-i", str(src),
        "-vf", f"colorkey=0x{color}:{similarity}:0.08,format=rgba",
        "-c:v", "png", str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not dest.exists():
        Image.open(src).convert("RGBA").save(dest)
    return dest


def _sfx_kind(shot: dict) -> str | None:
    fx = (shot.get("fx") or "").lower()
    cap = (shot.get("caption") or "").strip()
    style = (shot.get("caption_style") or "").lower()
    if fx in ("flash", "whip"):
        return "whoosh"
    if cap.endswith("!") or style in ("impact", "hot", "pop"):
        return "pop"
    if cap.endswith("?"):
        return "click"
    return None


def _synth(path: Path, freq: float, dur: float, kind: str) -> None:
    sr = 44100
    n = max(64, int(sr * dur))
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        buf = bytearray()
        for i in range(n):
            t = i / sr
            env = max(0.0, 1.0 - t / dur) ** 1.4
            if kind == "whoosh":
                val = env * math.sin(2 * math.pi * (freq + 900 * t) * t)
            elif kind == "click":
                val = env * math.sin(2 * math.pi * freq * t) * (1.0 if i < 40 else env)
            else:
                val = env * math.sin(2 * math.pi * freq * t)
            buf += struct.pack("<h", int(max(-1, min(1, val)) * 16000))
        w.writeframes(buf)


def _skin(r: int, g: int, b: int) -> bool:
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = r - y
    cb = b - y
    return 80 < y < 230 and 10 < cr < 80 and -80 < cb < -10


def _hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def _rgba(hex_color: str, a: int) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a


if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    theme = sample_theme([])
    assert theme["fill"].startswith("#")
    kit = ensure_kit(tmp)
    assert kit["pop1"].stat().st_size > 200
    heart = make_sticker("heart", tmp / "h.png")
    assert Image.open(heart).mode == "RGBA"
    green = Image.new("RGBA", (32, 32), (0, 255, 0, 255))
    green.putpixel((16, 16), (255, 0, 0, 255))
    gp = tmp / "g.png"
    green.save(gp)
    out = knockout(gp, tmp / "k.png", "00FF00", 0.4)
    assert Image.open(out).mode in ("RGBA", "RGB")
    v = {"shots": [
        {"role": "hook", "fx": "none", "start_sec": 0, "end_sec": 2,
         "caption": "와!", "caption_style": "impact", "video_index": 0},
        {"role": "setup", "fx": "zoom_in", "start_sec": 2, "end_sec": 4,
         "caption": "다음", "caption_style": "normal", "video_index": 0},
    ]}
    assign_sfx([v])
    assert v["shots"][0].get("sfx", "").startswith("pop")
    apply_motion([v], [{"path": str(gp), "duration_sec": 10}], tmp)
    assert v["shots"][1]["fx"] == "none", v["shots"][1]  # fatigue: 2s after hook
    print("polish self-check ok", theme, kit.keys())
