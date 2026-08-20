"""Footage index: scene / line / action / emotion, never a finished cut.

Eddie-style auto-assembly is out of its depth on vérité. What we can do
is surface every usable beat, flag the dead ones, keep alt takes, and
hand source timecode to Premiere. The editor still chooses.
"""
from __future__ import annotations

import re
from pathlib import Path

_DIALOGUE = re.compile(r"[\"「『」』]|says |said |대사|말한다|얘기")
_ACTION = re.compile(
    r"뛰|걷|던|잡|넘어|먹|찍|앉|일어|달리|안|키스|때리|run|walk|fall|eat|hit|grab",
    re.I,
)
_EMOTION = re.compile(
    r"웃|울|화|당황|놀|감동|짜증|기쁨|슬픔|부끄|폭주|laugh|cry|angry|shock|joy",
    re.I,
)
_MISTAKE = re.compile(r"\bNG\b|엔지|실수|잘못|컷해|다시 찍|take 2|리테이크", re.I)
_WORD = re.compile(r"[가-힣]{2,}|[a-zA-Z]{3,}")


def build(analysis: dict, infos: list[dict]) -> dict:
    clips = []
    for m in analysis.get("moments") or []:
        clips.append(_clip(m, infos))
    _group(clips)
    _alts(clips)
    stringout = [c["id"] for c in clips if "mistake" not in c["flags"]]
    selects = [
        c["id"] for c in clips
        if c.get("select") and c.get("verdict") != "discard"
    ]
    return {
        "clips": clips,
        "stringout": stringout,
        "selects": selects,
        "groups": _group_map(clips),
        "note": "candidates only — nothing deleted from source",
    }


def apply_actions(index: dict, actions: list[dict]) -> dict:
    by_id = {c["id"]: c for c in index.get("clips") or []}
    for act in actions:
        cid = act.get("id")
        clip = by_id.get(cid)
        if not clip:
            continue
        op = act.get("op")
        if op == "keep":
            clip["verdict"] = "keep"
        elif op == "discard":
            clip["verdict"] = "discard"
        elif op == "maybe":
            clip["verdict"] = "maybe"
        elif op == "handle":
            if act.get("start_sec") is not None:
                clip["start_sec"] = float(act["start_sec"])
            if act.get("end_sec") is not None:
                clip["end_sec"] = float(act["end_sec"])
        elif op == "reorder" and isinstance(act.get("order"), list):
            pass
    order = next((a.get("order") for a in actions if a.get("op") == "reorder"), None)
    if order:
        index["selects"] = [i for i in order if i in by_id]
    else:
        index["selects"] = [
            c["id"] for c in index["clips"]
            if c.get("select") and c.get("verdict") != "discard"
        ]
    index["stringout"] = [
        c["id"] for c in index["clips"] if "mistake" not in c["flags"]
    ]
    return index


def _clip(m: dict, infos: list[dict]) -> dict:
    idx = int(m.get("video_index") or 0)
    info = infos[idx] if idx < len(infos) else {}
    start = round(float(m.get("start_sec") or 0), 3)
    end = round(float(m.get("end_sec") or start + 1), 3)
    desc = (m.get("description") or "").strip()
    tags = _tags(desc, m.get("mood") or "")
    flags = _flags(m, info, start, end)
    brief = int(m.get("brief_fit") or 0)
    return {
        "id": f"v{idx}_{start:.2f}_{end:.2f}",
        "video_index": idx,
        "source": info.get("name") or Path(info.get("path") or "").name,
        "path": info.get("path") or "",
        "start_sec": start,
        "end_sec": end,
        "source_in": _tc(start),
        "source_out": _tc(end),
        "description": desc,
        "mood": m.get("mood") or "",
        "tags": tags,
        "intensity": int(m.get("intensity") or 0),
        "hook_potential": int(m.get("hook_potential") or 0),
        "brief_fit": brief,
        "flags": flags,
        "select": brief >= 7 or bool(m.get("must")),
        "group": None,
        "hero": False,
        "alts": [],
        "verdict": "keep" if brief >= 7 else "maybe",
        "must": bool(m.get("must")),
    }


def _tags(desc: str, mood: str) -> list[str]:
    text = f"{desc} {mood}"
    tags = []
    if _DIALOGUE.search(text):
        tags.append("dialogue")
    if _ACTION.search(text):
        tags.append("action")
    if _EMOTION.search(text) or mood:
        tags.append("emotion")
    if not tags:
        tags.append("scene")
    return tags


def _flags(m: dict, info: dict, start: float, end: float) -> list[str]:
    flags = []
    dur = max(0.01, end - start)
    if dur < 0.7 or _MISTAKE.search(m.get("description") or ""):
        flags.append("mistake")
    intensity = int(m.get("intensity") or 0)
    hook = int(m.get("hook_potential") or 0)
    if intensity <= 3 and hook <= 3:
        flags.append("boring")
    sil = info.get("silences") or []
    overlap = 0.0
    for a, b in sil:
        overlap += max(0.0, min(end, float(b)) - max(start, float(a)))
    if overlap / dur >= 0.55:
        flags.append("dead")
    return flags


def _group(clips: list[dict]) -> None:
    bags = [_tokens(c["description"]) for c in clips]
    gid = 0
    for i, clip in enumerate(clips):
        if clip["group"] is not None:
            continue
        clip["group"] = gid
        for j in range(i + 1, len(clips)):
            if clips[j]["group"] is not None:
                continue
            if clip["video_index"] != clips[j]["video_index"]:
                continue
            if _jaccard(bags[i], bags[j]) >= 0.34:
                clips[j]["group"] = gid
        gid += 1


def _alts(clips: list[dict]) -> None:
    by: dict[int, list[dict]] = {}
    for c in clips:
        by.setdefault(c["group"], []).append(c)
    for members in by.values():
        members.sort(key=lambda c: (
            -int(c.get("must") or 0),
            -c["brief_fit"],
            -c["hook_potential"],
        ))
        hero = members[0]
        hero["hero"] = True
        for other in members[1:]:
            other["hero"] = False
            hero["alts"].append(other["id"])
            if _jaccard(_tokens(hero["description"]), _tokens(other["description"])) >= 0.7:
                if "duplicate" not in other["flags"]:
                    other["flags"].append("duplicate")


def _group_map(clips: list[dict]) -> list[dict]:
    by: dict[int, list[str]] = {}
    for c in clips:
        by.setdefault(c["group"], []).append(c["id"])
    return [{"group": g, "clips": ids} for g, ids in sorted(by.items())]


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tc(sec: float) -> str:
    s = max(0.0, float(sec))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec2 = s % 60
    frames = int(round((sec2 - int(sec2)) * 30))
    if frames >= 30:
        frames = 0
        sec2 = int(sec2) + 1
    return f"{h:02d}:{m:02d}:{int(sec2):02d}:{frames:02d}"


if __name__ == "__main__":
    analysis = {"moments": [
        {"video_index": 0, "start_sec": 10, "end_sec": 14, "description": "웃으며 말한다 괜찮아",
         "mood": "warm", "intensity": 8, "hook_potential": 8, "brief_fit": 9},
        {"video_index": 0, "start_sec": 40, "end_sec": 44, "description": "웃으며 말한다 괜찮아 진짜",
         "mood": "warm", "intensity": 7, "hook_potential": 7, "brief_fit": 6},
        {"video_index": 0, "start_sec": 90, "end_sec": 91, "description": "빈 복도",
         "mood": "", "intensity": 1, "hook_potential": 1, "brief_fit": 1},
    ]}
    idx = build(analysis, [{"name": "a.mov", "path": "a.mov", "silences": [(89, 95)]}])
    assert idx["clips"][0]["select"] and idx["clips"][0]["hero"]
    assert idx["clips"][1]["id"] in idx["clips"][0]["alts"]
    assert "dead" in idx["clips"][2]["flags"]
    assert idx["clips"][0]["source_in"].count(":") == 3
    apply_actions(idx, [{"op": "discard", "id": idx["clips"][0]["id"]}])
    assert idx["clips"][0]["verdict"] == "discard"
    print("index self-check ok", len(idx["clips"]), "clips")
