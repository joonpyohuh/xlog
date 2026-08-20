"""Job quality tier.

Local: both chips are selectable. Later a paid Pro subscribe sets
XLOG_PRO_UNLOCKED=0 so `pro` silently falls back to `fast`.
"""
from __future__ import annotations

from app import config

PRESETS: dict[str, dict] = {
    "fast": {
        "id": "fast",
        "label": "Standard",
        "hint": "Faster and cheaper. GPT Terra reads frames; Grok searches X.",
        "gemini_media": "low",
        "video_fps": 1.0,
        "writer": "gpt",
        "crf": "18",
        "render_preset": "veryfast",
        "pro": False,
    },
    "pro": {
        "id": "pro",
        "label": "Pro",
        "hint": "Same editor, cleaner encode. Costs more per job.",
        "gemini_media": "high",
        "video_fps": 1.0,
        "writer": "gpt",
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
    assert resolve("fast")["writer"] == "gpt"
    assert resolve("pro")["writer"] == "gpt"
    assert resolve("nope")["id"] == "fast"
    print("quality self-check ok", resolve("pro")["gemini_media"])
