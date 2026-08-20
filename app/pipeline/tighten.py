"""Text-based cut: drop dead air and filler without a round-trip to Vrew.

Silence = ffmpeg silencedetect (free). Filler words = whisper-1 on the
planned shot spans only, not the whole source. Editors then strike words
in the transcript and recut without re-analyzing.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from app import config

_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")
_FILLER = re.compile(
    r"^(음+|어+|그|저|막|일단|uh+|um+|hmm+|er+|ah+)$", re.I,
)


def editor_flags(job: dict | None) -> dict:
    raw = (job or {}).get("editor") or {}
    on = True if raw == {} else bool(raw)
    if isinstance(raw, dict):
        def _g(k: str) -> bool:
            return raw.get(k, True) is not False and str(raw.get(k, "1")) not in (
                "0", "false", "off", "no",
            )
        return {
            "silence": _g("silence"),
            "theme": _g("theme"),
            "sfx": _g("sfx"),
            "zoom": _g("zoom"),
            "stickers": _g("stickers"),
            "qa": _g("qa"),
        }
    return {k: on for k in ("silence", "theme", "sfx", "zoom", "stickers", "qa")}


def shot_len(shot: dict) -> float:
    ranges = shot.get("ranges")
    if ranges:
        return sum(max(0.0, float(e) - float(s)) for s, e in ranges)
    return max(0.0, float(shot.get("end_sec", 0)) - float(shot.get("start_sec", 0)))


def detect_silences(path: Path, duration: float) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    cmd = [
        config.FFMPEG_BIN, "-i", str(path),
        "-af", f"silencedetect=noise={config.SILENCE_NOISE_DB}dB:d={config.SILENCE_MIN_SEC}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, **config.SUBPROCESS_TEXT)
    log = (proc.stderr or "") + (proc.stdout or "")
    starts, ends = _SILENCE_START.findall(log), _SILENCE_END.findall(log)
    out: list[tuple[float, float]] = []
    for i, s in enumerate(starts):
        a = float(s)
        b = float(ends[i]) if i < len(ends) else duration
        if b - a >= config.SILENCE_MIN_SEC:
            out.append((max(0.0, a), min(duration, b)))
    return _merge(out)


def keep_windows(
    duration: float,
    drop: list[tuple[float, float]],
    pad: float | None = None,
) -> list[tuple[float, float]]:
    """Invert drop ranges. Leave `pad` of breath inside each cut."""
    pad = config.BREATH_KEEP_SEC if pad is None else pad
    cuts = _merge(
        (max(0.0, a + pad), min(duration, b - pad)) for a, b in drop
    )
    cuts = [(a, b) for a, b in cuts if b - a >= 0.04]
    keep: list[tuple[float, float]] = []
    t = 0.0
    for a, b in cuts:
        if a > t + 0.04:
            keep.append((round(t, 3), round(a, 3)))
        t = max(t, b)
    if duration > t + 0.04:
        keep.append((round(t, 3), round(duration, 3)))
    return keep or [(0.0, duration)]


def pieces_in(
    start: float, end: float, keeps: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    out = []
    for a, b in keeps:
        lo, hi = max(start, a), min(end, b)
        if hi - lo >= 0.12:
            out.append((round(lo, 3), round(hi, 3)))
    return out


def attach_silence(infos: list[dict]) -> None:
    for info in infos:
        if not info.get("has_audio", True):
            info["silences"] = []
            info["keeps"] = [(0.0, float(info["duration_sec"]))]
            continue
        sil = detect_silences(Path(info["path"]), float(info["duration_sec"]))
        info["silences"] = sil
        info["keeps"] = keep_windows(float(info["duration_sec"]), sil)
        info["saved_silence_sec"] = round(
            sum(b - a for a, b in sil), 2,
        )


def filter_moments(moments: list[dict], infos: list[dict]) -> list[dict]:
    """Drop beats that sit mostly in detected silence."""
    keeps = [info.get("keeps") or [(0.0, float(info["duration_sec"]))] for info in infos]
    out = []
    for m in moments:
        idx = int(m.get("video_index") or 0)
        if idx >= len(keeps):
            out.append(m)
            continue
        start, end = float(m["start_sec"]), float(m["end_sec"])
        dur = max(0.01, end - start)
        held = sum(hi - lo for lo, hi in pieces_in(start, end, keeps[idx]))
        if held / dur >= 0.35:
            out.append(m)
    return out or moments


def extract_wav(src: Path, dest: Path, start: float, end: float) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.FFMPEG_BIN, "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{max(0.2, end - start):.3f}",
        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    return dest if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 400 else None


def transcribe_spans(infos: list[dict], spans: list[tuple[int, float, float]], work: Path) -> list[dict]:
    from app.llm import openai_client
    words: list[dict] = []
    seen: list[tuple[int, float, float]] = []
    for idx, start, end in _merge_spans(spans):
        if idx >= len(infos) or not infos[idx].get("has_audio", True):
            continue
        if any(i == idx and abs(s - start) < 0.15 and abs(e - end) < 0.15 for i, s, e in seen):
            continue
        seen.append((idx, start, end))
        wav = extract_wav(
            Path(infos[idx]["path"]), work / f"stt_{idx}_{int(start)}_{int(end)}.wav",
            start, end,
        )
        if wav is None:
            continue
        chunk = openai_client.transcribe_words(wav, offset=start)
        for w in chunk:
            w["video_index"] = idx
            w["keep"] = not _is_filler(w["w"])
            words.append(w)
    return words


def drop_from_words(infos: list[dict], words: list[dict]) -> None:
    by_v: dict[int, list[tuple[float, float]]] = {}
    for w in words:
        if w.get("keep", True):
            continue
        idx = int(w.get("video_index") or 0)
        by_v.setdefault(idx, []).append((float(w["t0"]), float(w["t1"])))
    for idx, info in enumerate(infos):
        extra = by_v.get(idx) or []
        sil = list(info.get("silences") or []) + extra
        info["keeps"] = keep_windows(float(info["duration_sec"]), sil)
        info["filler_n"] = len(extra)


def snap_variants(variants: list[dict], infos: list[dict]) -> None:
    keeps = [info.get("keeps") or [(0.0, float(info["duration_sec"]))] for info in infos]
    for v in variants:
        shots = []
        for shot in v.get("shots") or []:
            idx = int(shot.get("video_index") or 0)
            k = keeps[idx] if idx < len(keeps) else [(0.0, 1e9)]
            pieces = pieces_in(float(shot["start_sec"]), float(shot["end_sec"]), k)
            if not pieces:
                continue
            shot = dict(shot)
            shot["ranges"] = pieces
            shots.append(shot)
        v["shots"] = shots
        v["total_sec"] = round(sum(shot_len(s) for s in shots), 2)


def apply_drops(infos: list[dict], drops: list[dict]) -> None:
    extra: dict[int, list[tuple[float, float]]] = {}
    for d in drops:
        idx = int(d.get("video_index") or 0)
        extra.setdefault(idx, []).append((float(d["start_sec"]), float(d["end_sec"])))
    for idx, info in enumerate(infos):
        sil = list(info.get("silences") or []) + (extra.get(idx) or [])
        info["keeps"] = keep_windows(float(info["duration_sec"]), sil)


def transcript_payload(words: list[dict], variants: list[dict]) -> dict:
    used: list[tuple[int, float, float]] = []
    for v in variants:
        for s in v.get("shots") or []:
            used.append((
                int(s.get("video_index") or 0),
                float(s["start_sec"]), float(s["end_sec"]),
            ))
    visible = []
    for w in words:
        idx = int(w.get("video_index") or 0)
        t0, t1 = float(w["t0"]), float(w["t1"])
        if any(i == idx and t1 >= a and t0 <= b for i, a, b in used):
            visible.append(w)
    return {"words": visible or words, "count": len(visible or words)}


def _is_filler(token: str) -> bool:
    t = (token or "").strip().strip(".,!?…~")
    return bool(_FILLER.match(t))


def _merge(ranges) -> list[tuple[float, float]]:
    items = sorted((float(a), float(b)) for a, b in ranges if b > a)
    if not items:
        return []
    out = [items[0]]
    for a, b in items[1:]:
        if a <= out[-1][1] + 0.02:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _merge_spans(spans: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    by: dict[int, list[tuple[float, float]]] = {}
    for i, a, b in spans:
        by.setdefault(i, []).append((a, b))
    out = []
    for i, rs in by.items():
        for a, b in _merge(rs):
            out.append((i, a, b))
    return out


if __name__ == "__main__":
    sil = [(1.0, 2.5), (2.4, 3.0), (10.0, 10.1)]
    k = keep_windows(12.0, sil, pad=0.0)
    assert k[0] == (0.0, 1.0), k
    assert any(abs(a - 3.0) < 0.05 for a, _ in k), k
    assert pieces_in(0.5, 4.0, k)[0][0] == 0.5
    shot = {"start_sec": 0, "end_sec": 5, "ranges": [(0, 1), (3, 4.5)]}
    assert abs(shot_len(shot) - 2.5) < 1e-6
    assert _is_filler("음") and _is_filler("uh") and not _is_filler("그래서")
    flags = editor_flags({"editor": {}})
    assert flags["silence"] and flags["sfx"]
    assert editor_flags({"editor": {"silence": False}})["silence"] is False
    v = {"shots": [{"video_index": 0, "start_sec": 0.5, "end_sec": 4.0}]}
    snap_variants([v], [{"duration_sec": 12.0, "keeps": k}])
    assert v["shots"][0]["ranges"]
    print("tighten self-check ok", k, v["total_sec"])
