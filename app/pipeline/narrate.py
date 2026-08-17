"""Voice-over: read each shot's caption aloud, in sync with the caption.

One line per captioned shot, delayed to that shot's offset on the rendered
timeline, mixed over ducked source audio. Edge-TTS rate/pitch/volume plus
a Korean neural voice change with the shot's role, style, and FX.
"""
from __future__ import annotations

import json
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import config

VOICES = {
    "off":      {"label": "없음 (원본 소리만)", "id": None},
    "auto":     {"label": "자동 (장면마다 바뀜)", "id": None},
    "sunhi":    {"label": "선히 · 밝은 여성", "id": "ko-KR-SunHiNeural"},
    "yujin":    {"label": "유진 · 또렷한 여성", "id": "ko-KR-YuJinNeural"},
    "jimin":    {"label": "지민 · 부드러운 여성", "id": "ko-KR-JiMinNeural"},
    "seohyeon": {"label": "서현 · 차분한 여성", "id": "ko-KR-SeoHyeonNeural"},
    "injoon":   {"label": "인준 · 힘있는 남성", "id": "ko-KR-InJoonNeural"},
    "hyunsu":   {"label": "현수 · 낮은 남성", "id": "ko-KR-HyunsuNeural"},
    "bongjin":  {"label": "봉진 · 중저음 남성", "id": "ko-KR-BongJinNeural"},
}

_CHANNEL_VOICE = {
    "gaming": "injoon",
    "parenting": "sunhi",
    "pets": "yujin",
    "comedy": "yujin",
    "beauty": "seohyeon",
    "howto": "hyunsu",
    "vlog": "sunhi",
}

_MOOD_VOICE = {
    "hype": "sunhi",
    "punch": "yujin",
    "tense": "injoon",
    "warm": "seohyeon",
    "soft": "jimin",
    "narrate": "sunhi",
}

_MOOD = {
    "hype":    {"rate": "+18%", "pitch": "+8Hz", "volume": "+10%"},
    "punch":   {"rate": "+10%", "pitch": "+5Hz", "volume": "+6%"},
    "tense":   {"rate": "+8%", "pitch": "-5Hz", "volume": "+4%"},
    "warm":    {"rate": "-6%", "pitch": "+2Hz", "volume": "+0%"},
    "soft":    {"rate": "-12%", "pitch": "-3Hz", "volume": "-8%"},
    "narrate": {"rate": "+2%", "pitch": "+0Hz", "volume": "+0%"},
}


def available_voices() -> list[dict[str, str]]:
    return [{"id": k, "label": v["label"]} for k, v in VOICES.items()]


def mood_for_shot(shot: dict) -> str:
    role = (shot.get("role") or "").lower()
    style = (shot.get("caption_style") or "").lower()
    fx = (shot.get("fx") or "").lower()
    if style in ("impact", "hot") or fx in ("flash", "whip") or role == "hook":
        return "hype"
    if style in ("emphasis", "pop") or role == "payoff":
        return "punch"
    if fx == "shake" or role == "development":
        return "tense"
    if style == "sub" or role == "ending":
        return "soft"
    if style == "plate" or role == "setup":
        return "warm"
    return "narrate"


def _voice_id(preset: str, mood: str, channel: str) -> str:
    if preset and preset != "auto" and preset in VOICES and VOICES[preset]["id"]:
        return VOICES[preset]["id"]
    pick = _MOOD_VOICE.get(mood) or _CHANNEL_VOICE.get(channel) or "sunhi"
    return VOICES.get(pick, VOICES["sunhi"])["id"] or config.EDGE_TTS_VOICE


def _spoken(text: str, mood: str) -> str:
    t = text.strip()
    if mood == "hype" and t[-1:] not in "!?…":
        return t + "!"
    if mood == "soft" and t[-1:] not in ".…":
        return t + "…"
    return t


def _edge_tts(text: str, out_path: Path, *, voice: str, mood: str) -> bool:
    try:
        import edge_tts

        prosody = _MOOD.get(mood, _MOOD["narrate"])
        edge_tts.Communicate(
            _spoken(text, mood),
            voice,
            rate=prosody["rate"],
            pitch=prosody["pitch"],
            volume=prosody["volume"],
        ).save_sync(str(out_path))
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return False


def synthesize(
    text: str,
    out_path: Path,
    *,
    shot: dict | None = None,
    voice: str = "auto",
    channel: str = "",
) -> Path | None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mood = mood_for_shot(shot or {})
    vid = _voice_id(voice, mood, channel)
    return out_path if _edge_tts(text, out_path, voice=vid, mood=mood) else None


def _duration(path: Path) -> float:
    out = subprocess.run(
        [config.FFPROBE_BIN, "-v", "error", "-print_format", "json",
         "-show_format", str(path)],
        check=True,
        **config.SUBPROCESS_TEXT,
    ).stdout or ""
    if not out.strip():
        return 0.0
    return float(json.loads(out)["format"].get("duration", 0.0))


def _lines(variant: dict) -> list[tuple[float, float, dict]]:
    """(timeline offset, shot length, shot) for every spoken shot."""
    out: list[tuple[float, float, dict]] = []
    offset = 0.0
    for shot in variant.get("shots") or []:
        length = float(shot["end_sec"]) - float(shot["start_sec"])
        text = (shot.get("caption") or "").strip()
        if text and shot.get("caption_style", "none") != "none":
            out.append((offset, length, shot))
        offset += length
    return out


def wants_narration(voice: str | None) -> bool:
    if not config.NARRATION:
        return False
    return (voice or "auto").strip().lower() not in {"off", "none", "no", "0", "silent"}


def add_narration(video: Path, variant: dict, voice_dir: Path, out_path: Path) -> Path:
    """Mix a synced voice-over into `video`. Falls back to the silent cut."""
    voice = variant.get("voice") or "auto"
    lines = _lines(variant)
    if not wants_narration(voice) or not lines:
        return video.replace(out_path)
    channel = variant.get("channel") or ""
    try:
        with ThreadPoolExecutor(max_workers=config.RENDER_WORKERS) as pool:
            clips = list(pool.map(
                lambda item: synthesize(
                    item[1][2].get("caption", ""),
                    voice_dir / f"vo_{item[0]:03d}.mp3",
                    shot=item[1][2],
                    voice=voice,
                    channel=channel,
                ),
                enumerate(lines),
            ))

        cmd = [config.FFMPEG_BIN, "-y", "-v", "error", "-i", str(video)]
        chains, labels = [], []
        for (offset, length, _), clip in zip(lines, clips):
            if clip is None:
                continue
            spoken = _duration(clip)
            if spoken <= 0.05:
                continue
            idx = len(labels) + 1
            cmd += ["-i", str(clip)]
            tempo = min(config.NARRATION_MAX_TEMPO, spoken / length) if spoken > length else 1.0
            speed = f"atempo={tempo:.3f}," if tempo > 1.0 else ""
            chains.append(
                f"[{idx}:a]{speed}adelay={int(offset * 1000)}:all=1,"
                f"aresample=48000[vo{idx}]"
            )
            labels.append(f"[vo{idx}]")
        if not labels:
            return video.replace(out_path)

        chains.append(f"[0:a]volume={config.NARRATION_DUCK}[bg]")
        chains.append(
            "[bg]" + "".join(labels)
            + f"amix=inputs={len(labels) + 1}:duration=first:normalize=0[aout]"
        )
        cmd += [
            "-filter_complex", ";".join(chains),
            "-map", "0:v", "-c:v", "copy",
            "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path
    except Exception:  # noqa: BLE001 — a missing voice must not lose the edit
        traceback.print_exc()
        return video.replace(out_path)


if __name__ == "__main__":
    v = {"shots": [
        {"start_sec": 0, "end_sec": 3, "caption": "first", "caption_style": "pop",
         "role": "hook", "fx": "flash"},
        {"start_sec": 10, "end_sec": 12.5, "caption": "", "caption_style": "pop"},
        {"start_sec": 20, "end_sec": 24, "caption": "third", "caption_style": "none"},
        {"start_sec": 30, "end_sec": 34, "caption": "fourth", "caption_style": "plate",
         "role": "setup", "fx": "none"},
    ]}
    got = _lines(v)
    assert [round(o, 2) for o, _, _ in got] == [0.0, 9.5], got
    assert [s["caption"] for _, _, s in got] == ["first", "fourth"], got
    assert mood_for_shot(got[0][2]) == "hype", mood_for_shot(got[0][2])
    assert mood_for_shot(got[1][2]) == "warm", mood_for_shot(got[1][2])
    assert _voice_id("auto", "hype", "vlog").endswith("Neural")
    assert _spoken("와", "hype") == "와!"
    assert wants_narration("auto") and wants_narration("sunhi")
    assert not wants_narration("off")
    print("narrate self-check ok")
