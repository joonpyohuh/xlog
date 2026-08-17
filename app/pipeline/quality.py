"""Job quality tier.

Local: both chips are selectable. Later a paid Pro subscribe sets
XLOG_PRO_UNLOCKED=0 so `pro` silently falls back to `fast`.
"""
from __future__ import annotations

from app import config

PRESETS: dict[str, dict] = {
    "fast": {
        "id": "fast",
        "label": "기본",
        "hint": "빠르고 저렴. 분석은 저해상도, 편집안은 Grok.",
        "gemini_media": "low",
        "video_fps": 1.0,
        "writer": "grok",
        "crf": "18",
        "render_preset": "veryfast",
        "pro": False,
    },
    "pro": {
        "id": "pro",
        "label": "프로",
        "hint": "분석 고화질 + Claude 편집안(키 있으면) + 조금 더 깨끗한 인코딩. 편당 비용↑.",
        "gemini_media": "high",
        "video_fps": 1.0,
        "writer": "claude",
        "crf": "16",
        "render_preset": "faster",
        "pro": True,
    },
}


def resolve(name: str | None) -> dict:
    key = (name or config.QUALITY_DEFAULT or "fast").strip().lower()
    if key not in PRESETS:
        key = "fast"
    spec = dict(PRESETS[key])
    if spec["pro"] and not config.PRO_UNLOCKED:
        return dict(PRESETS["fast"])
    return spec


def for_ui() -> dict:
    return {
        "unlocked": config.PRO_UNLOCKED,
        "claude": bool(config.ANTHROPIC_API_KEY),
        "default": config.QUALITY_DEFAULT if config.QUALITY_DEFAULT in PRESETS else "fast",
        "qualities": [
            {
                "id": p["id"],
                "label": p["label"],
                "hint": p["hint"],
                "pro": p["pro"],
            }
            for p in PRESETS.values()
        ],
    }


if __name__ == "__main__":
    assert resolve("fast")["writer"] == "grok"
    assert resolve("nope")["id"] == "fast"
    print("quality self-check ok", resolve("pro")["gemini_media"])
