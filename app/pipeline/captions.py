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

FONTS = {
    "malgun":   {"label": "맑은 고딕", "paths": ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf"]},
    "headline": {"label": "헤드라인", "paths": ["C:/Windows/Fonts/H2HDRM.TTF"]},
    "gothic":   {"label": "견고딕", "paths": ["C:/Windows/Fonts/H2GTRE.TTF"]},
    "round":    {"label": "둥근고딕", "paths": ["C:/Windows/Fonts/H2PORM.TTF"]},
    "expo":     {"label": "엑스포", "paths": ["C:/Windows/Fonts/HMFMMUEX.TTC"]},
    "old":      {"label": "옛체", "paths": ["C:/Windows/Fonts/HMFMOLD.TTF"]},
    "hangul":   {"label": "HY한글", "paths": ["C:/Windows/Fonts/H2MKPB.TTF"]},
    "gulim":    {"label": "굴림", "paths": ["C:/Windows/Fonts/gulim.ttc"]},
    "batang":   {"label": "바탕", "paths": ["C:/Windows/Fonts/batang.ttc"]},
}


def _first_font(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def available_fonts() -> list[dict[str, str]]:
    out = []
    for fid, spec in FONTS.items():
        if _first_font(spec["paths"]):
            out.append({"id": fid, "label": spec["label"]})
    if not out:
        out = [{"id": "malgun", "label": "기본"}]
    return out


def _load_font(size: int, font_id: str = "malgun") -> ImageFont.ImageFont:
    spec = FONTS.get(font_id) or FONTS["malgun"]
    for candidate in [*spec["paths"], *_FONT_CANDIDATES]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()

# Current short-form looks: big, tight, heavy stroke, sitting mid-low in frame.
# `fill`/`stroke` are CSS colors; pill = rounded dark plate behind the text.
STYLES = {
    # TikTok/Reels default: fat white with a hard black rim.
    "normal":   {"size_ratio": 0.062, "fill": "#FFFFFF", "stroke": "#000000", "y": 0.70, "shadow": True},
    # CapCut "bold" preset — white on a rounded black pill.
    "plate":    {"size_ratio": 0.058, "fill": "#FFFFFF", "stroke": "#000000", "y": 0.72, "pill": True},
    # punchline word: acid yellow, heavy rim
    "emphasis": {"size_ratio": 0.072, "fill": "#FFE14D", "stroke": "#000000", "y": 0.68, "shadow": True},
    # keyword highlighter bar (CapCut "highlight")
    "box":      {"size_ratio": 0.062, "fill": "#111111", "stroke": None, "y": 0.70, "bar": "#D6FF3C"},
    # hyped reaction line
    "pop":      {"size_ratio": 0.070, "fill": "#FFFFFF", "stroke": "#FF2D55", "y": 0.68, "shadow": True},
    "hot":      {"size_ratio": 0.070, "fill": "#FF3B5C", "stroke": "#FFFFFF", "y": 0.68, "shadow": True},
    "mint":     {"size_ratio": 0.066, "fill": "#3CFFB0", "stroke": "#04231A", "y": 0.70, "shadow": True},
    "neon":     {"size_ratio": 0.066, "fill": "#00F0FF", "stroke": "#7A00FF", "y": 0.70, "glow": True},
    "gold":     {"size_ratio": 0.066, "fill": "#FFD60A", "stroke": "#221800", "y": 0.70, "shadow": True},
    # full-bleed shout used on the hook / turn
    "impact":   {"size_ratio": 0.098, "fill": "#FFFFFF", "stroke": "#000000", "y": 0.46, "shadow": True},
    # tiny clean lower line for context beats
    "sub":      {"size_ratio": 0.042, "fill": "#FFFFFF", "stroke": "#000000", "y": 0.84},
}

CAPTION_STYLES = tuple(["none", *STYLES.keys()])
SHOT_FX = (
    "none",
    "punch_in",    # instant crop-in on the beat
    "zoom_in",     # slow push
    "zoom_out",    # slow pull back
    "shake",       # handheld jitter for chaotic beats
    "flash",       # white flash cut-in at the pivot
    "whip",        # motion-blur wipe into the shot
)


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


def make_caption_png(text: str, style: str, out_path: Path, font_id: str = "malgun") -> Path:
    """Full-frame transparent PNG with the caption look applied."""
    w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    spec = STYLES.get(style, STYLES["normal"])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    size = int(w * spec["size_ratio"])
    font = _load_font(size, font_id)
    stroke = 0 if spec.get("stroke") is None else max(3, size // 7)

    lines = _wrap(draw, text, font, max_width=int(w * 0.86))
    line_h = int(size * 1.14)
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
    if spec.get("pill"):
        draw.rounded_rectangle(box, radius=int(size * 0.5), fill=(0, 0, 0, 205))
    if spec.get("bar"):
        draw.rounded_rectangle(box, radius=int(size * 0.16), fill=spec["bar"])

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
    tmp = Path(tempfile.mkdtemp())
    for name in CAPTION_STYLES:
        if name == "none":
            continue
        out = make_caption_png("이게 진짜 말이 돼?", name, tmp / f"{name}.png")
        assert out.exists() and out.stat().st_size > 200, name
    fonts = available_fonts()
    assert fonts and fonts[0]["id"], fonts
    print("ok", len(CAPTION_STYLES) - 1, "styles", [f["id"] for f in fonts], tmp)
