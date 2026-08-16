"""Burned-in caption overlays — color + style, not just white text.

The screenwriter picks a named look per shot. PIL renders a full-frame
transparent PNG; render.py composites it in the same ffmpeg pass.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app import config

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
]

# Viral-shorts looks. `fill`/`stroke` are CSS colors; plate = dark bar behind.
STYLES = {
    "normal":   {"size_ratio": 0.045, "fill": "#FFFFFF", "stroke": "#000000", "y": 0.74},
    "emphasis": {"size_ratio": 0.060, "fill": "#FFE14D", "stroke": "#000000", "y": 0.74},
    "pop":      {"size_ratio": 0.062, "fill": "#FFFFFF", "stroke": "#FF2D55", "y": 0.72, "shadow": True},
    "neon":     {"size_ratio": 0.055, "fill": "#00F0FF", "stroke": "#FF00AA", "y": 0.74, "glow": True},
    "hot":      {"size_ratio": 0.058, "fill": "#FF3B5C", "stroke": "#FFFFFF", "y": 0.74},
    "mint":     {"size_ratio": 0.055, "fill": "#00F5A0", "stroke": "#003322", "y": 0.74},
    "gold":     {"size_ratio": 0.056, "fill": "#FFD60A", "stroke": "#3D2A00", "y": 0.74, "plate": True},
    "plate":    {"size_ratio": 0.048, "fill": "#FFFFFF", "stroke": "#000000", "y": 0.76, "plate": True},
    "box":      {"size_ratio": 0.050, "fill": "#111111", "stroke": None, "y": 0.78, "bar": "#FFE14D"},
    "impact":   {"size_ratio": 0.072, "fill": "#FFFFFF", "stroke": "#E10600", "y": 0.50, "shadow": True},
}

CAPTION_STYLES = tuple(["none", *STYLES.keys()])
SHOT_FX = ("none", "punch_in", "zoom_in")


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        current = ""
        for word in raw_line.split(" "):
            candidate = (current + " " + word).strip()
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = ""
            for ch in word:
                if draw.textlength(current + ch, font=font) <= max_width:
                    current += ch
                else:
                    lines.append(current)
                    current = ch
        lines.append(current)
    return [l for l in lines if l]


def make_caption_png(text: str, style: str, out_path: Path) -> Path:
    """Full-frame transparent PNG with the caption look applied."""
    w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    spec = STYLES.get(style, STYLES["normal"])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    size = int(w * spec["size_ratio"])
    font = _load_font(size)
    stroke = 0 if spec.get("stroke") is None else max(2, size // 9)

    lines = _wrap(draw, text, font, max_width=int(w * 0.86))
    line_h = int(size * 1.28)
    block_h = line_h * len(lines)
    max_tw = max((draw.textlength(line, font=font) for line in lines), default=0)
    y0 = int(h * spec.get("y", 0.74)) - block_h // 2

    pad_x, pad_y = int(size * 0.55), int(size * 0.28)
    box = (
        int((w - max_tw) / 2) - pad_x,
        y0 - pad_y,
        int((w + max_tw) / 2) + pad_x,
        y0 + block_h + pad_y,
    )
    if spec.get("plate"):
        draw.rounded_rectangle(box, radius=int(size * 0.35), fill=(0, 0, 0, 170))
    if spec.get("bar"):
        draw.rectangle(box, fill=spec["bar"])

    y = y0
    for line in lines:
        tw = draw.textlength(line, font=font)
        xy = ((w - tw) / 2, y)
        if spec.get("shadow") or spec.get("glow"):
            off = max(3, size // 14)
            draw.text(
                (xy[0] + off, xy[1] + off),
                line, font=font, fill=(0, 0, 0, 180),
                stroke_width=stroke, stroke_fill=(0, 0, 0, 180),
            )
        draw.text(
            xy, line, font=font,
            fill=spec["fill"],
            stroke_width=stroke,
            stroke_fill=spec.get("stroke") or spec["fill"],
        )
        y += line_h

    if spec.get("glow"):
        glow = img.filter(ImageFilter.GaussianBlur(radius=3))
        img = Image.alpha_composite(glow, img)

    img.save(out_path)
    return out_path


if __name__ == "__main__":
    import tempfile
    out = Path(tempfile.mkdtemp()) / "cap.png"
    make_caption_png("테스트 자막", "neon", out)
    assert out.exists() and out.stat().st_size > 200, out
    print("ok", out, out.stat().st_size)
