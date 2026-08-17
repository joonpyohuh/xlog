"""The creator's reason for pulling a short — structured, not a blank box.

A parenting channel cuts the cutest beat. A gaming channel cuts the clutch.
If xlog hunts generic 'peaks' instead of that reason, the short is unused.
This module is the brief: channel archetype + why + hunt + emphasis + skip,
plus `must` — the one part the creator wants carried as the main beat.
"""
from __future__ import annotations

import re
from typing import Any

CHANNELS: dict[str, dict[str, str]] = {
    "parenting": {
        "label": "육아·가족",
        "hunt": "가장 귀엽거나 마음이 움직이는 장면 — 웃음, 첫 말·첫 걸음, 안아주는 순간, 실수해도 사랑스러운 표정. 평범한 일과·정리 장면은 버려라.",
        "emphasize": "감정 정점에서 punch_in 또는 zoom_in으로 얼굴에 붙고, 그 샷은 조금 더 길게 둔다. whip/flash/shake는 거의 쓰지 마라. 자막은 짧은 감탄.",
        "skip": "이동, 준비, 잔소리, 설명, 빈 방.",
    },
    "gaming": {
        "label": "게임·매드무비",
        "hunt": "클러치, 킬, 역전, 콤보, 리액션이 터지는 순간 — 명장면만. 로밍·상점·로딩은 버려라.",
        "emphasize": "킬/클러치에 punch_in + flash 또는 whip. 자막은 짧은 반응(와/미쳤다/개꿀). 페이오프를 훅으로 열어라.",
        "skip": "로딩, 설정, 조용한 이동, 패배 변명.",
    },
    "vlog": {
        "label": "브이로그·일상",
        "hunt": "오늘을 한 줄로 말할 수 있는 장면 — 장소 한 컷, 사람 표정, 그 날의 사건. 비슷한 풍경을 반복하지 마라.",
        "emphasize": "훅은 결과/한 줄 상황 선언. 컷은 부드럽게, 자막은 상황 설명 한 줄.",
        "skip": "이동 과정, 가방 정리, 반복되는 풍경.",
    },
    "comedy": {
        "label": "예능·리액션",
        "hunt": "웃음이 터지는 비트, 드립, 표정 붕괴, 반전. 설명이 긴 세팅은 최소만.",
        "emphasize": "펀치라인에 flash/whip, 리액션 얼굴에 punch_in. 자막은 말맛 살린 한 줄.",
        "skip": "웃음 없는 설명, 같은 농담의 재탕.",
    },
    "howto": {
        "label": "정보·하우투",
        "hunt": "결과물 먼저, 그다음 핵심 한 단계. 시청자가 따라 할 수 있는 동작만.",
        "emphasize": "결과 샷으로 훅. 단계마다 짧은 자막. 효과는 절제, 중요한 숫자/재료만 box/emphasis.",
        "skip": "인사, 구독 유도, 실패 테이크, 같은 동작 반복.",
    },
    "beauty": {
        "label": "뷰티·패션",
        "hunt": "비포/애프터, 가장 예쁜 각도, 색·질감이 살아있는 클로즈업.",
        "emphasize": "애프터를 훅으로. zoom_in on 디테일. 자막은 제품/포인트 한 단어.",
        "skip": "준비, 말만 하는 토크, 흐린 조명.",
    },
    "pets": {
        "label": "반려동물",
        "hunt": "가장 귀엽거나 웃긴 행동 — 고개 기울임, 점프, 실패, 주인 보는 눈. 앉아있기만 한 컷은 버려라.",
        "emphasize": "얼굴/발에 punch_in. 자막은 의성어·짧은 대사. 정점은 홀드.",
        "skip": "빈 방, 잠만 자는 긴 컷, 손만 보이는 정돈.",
    },
    "other": {
        "label": "기타",
        "hunt": "이 채널이 구독자에게 약속하는 그 장면 — 시청자가 스크롤을 멈추는 이유.",
        "emphasize": "그 장면을 훅에 두고 punch_in. 나머지는 그 장면을 설명하는 데만 쓴다.",
        "skip": "약속과 상관없는 과정, 반복.",
    },
}

FIELDS = ("channel", "why", "hunt", "emphasize", "skip", "must", "notes")

# Phrasing that carries no signal for matching a moment description.
_STOPWORDS = {
    "부분", "장면", "구간", "여기", "거기", "그거", "이거", "저거", "그것", "이것",
    "메인", "강조", "중심", "위주", "중요", "핵심", "가져", "가져가", "가져가고",
    "싶다", "싶어", "해줘", "해라", "하고", "그리고", "그부분", "이부분", "정도",
    "쯤에", "쯤", "부터", "까지", "에서", "으로", "무조건", "반드시", "꼭",
}


def normalize(raw: Any) -> dict[str, str]:
    if isinstance(raw, str):
        raw = {"notes": raw}
    src = raw if isinstance(raw, dict) else {}
    out = {k: str(src.get(k) or "").strip() for k in FIELDS}
    ch = CHANNELS.get(out["channel"])
    if ch:
        for k in ("hunt", "emphasize", "skip"):
            if not out[k]:
                out[k] = ch[k]
    return out


def compile(brief: dict[str, str]) -> str:
    """One prompt block the writer, judge, and analyzer all share."""
    b = normalize(brief)
    ch = CHANNELS.get(b["channel"])
    lines = ["## Why this short exists (the creator's point — outrank generic peaks)"]
    if ch:
        lines.append(f"Channel: {ch['label']}")
    if b["why"]:
        lines.append(f"Why pull this short: {b['why']}")
    else:
        lines.append("Why pull this short: not stated — infer from channel and hunt.")
    if b["hunt"]:
        lines.append(f"Hunt these scenes (ignore the rest): {b['hunt']}")
    if b["emphasize"]:
        lines.append(f"How to punch those scenes: {b['emphasize']}")
    if b["skip"]:
        lines.append(f"Do NOT include: {b['skip']}")
    if b["must"]:
        lines.append(
            f"MAIN BEAT — the creator explicitly asked for this: {b['must']}\n"
            "This part is mandatory. Build the short around it: it is the hook "
            "or the payoff, it gets the longest screen time and the strongest "
            "emphasis. Everything else exists to set it up or react to it."
        )
    if b["notes"]:
        lines.append(f"Extra request: {b['notes']}")
    lines.append(
        "A shot that does not serve this point is a failed shot, even if it is pretty."
    )
    return "\n".join(lines)


def hunt_system_addendum(brief: dict[str, str]) -> str:
    text = compile(brief)
    must = normalize(brief)["must"] if isinstance(brief, (dict, str)) else ""
    extra = (
        "\nIf a moment matches the MAIN BEAT above, score brief_fit 10 and say "
        "so in its description.\n"
        if must
        else ""
    )
    return (
        "\n\n" + text + "\n" + extra
        + "Score brief_fit 1-10: 10 = this IS why the creator would post, "
        "1 = filler even if visually busy. Prefer high brief_fit over raw intensity."
    )


# ------------------ MAIN BEAT matching (deterministic) ------------------ #

_HMS = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)")
_KOR_MIN = re.compile(r"(\d+)\s*분(?:\s*(\d+)\s*초)?")
_KOR_SEC = re.compile(r"(\d+)\s*초")
_RANGE_SEP = re.compile(r"\s*(?:~|-|–|부터|에서)\s*")
_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")


def _timecodes(text: str) -> list[float]:
    """Every clock reference in the text, as absolute seconds."""
    out: list[float] = []
    for m in _HMS.finditer(text):
        a, b, c = m.group(1), m.group(2), m.group(3)
        out.append(
            float(a) * 3600 + float(b) * 60 + float(c) if c
            else float(a) * 60 + float(b)
        )
    # consume "3분 20초" whole, so the trailing "20초" is not read again as 20s
    rest = _HMS.sub(" ", text)
    for m in _KOR_MIN.finditer(rest):
        out.append(float(m.group(1)) * 60 + float(m.group(2) or 0))
    for m in _KOR_SEC.finditer(_KOR_MIN.sub(" ", rest)):
        out.append(float(m.group(1)))
    return out


def must_ranges(text: str, pad_sec: float = 20.0) -> list[tuple[float, float]]:
    """Time windows the creator pointed at. `12:30~14:00` stays a span;
    a lone `12:30` becomes a window padded on both sides, since a human
    quoting a timestamp means "around there", not that exact frame."""
    text = (text or "").strip()
    if not text:
        return []
    spans: list[tuple[float, float]] = []
    for part in _RANGE_SEP.split(text):
        codes = _timecodes(part)
        if codes:
            spans.append((min(codes), max(codes)))
    if not spans:
        return []
    merged: list[tuple[float, float]] = []
    # Two adjacent fragments each holding one code = the two ends of a range.
    if len(spans) == 2 and spans[0][1] == spans[0][0] and spans[1][1] == spans[1][0]:
        lo, hi = sorted([spans[0][0], spans[1][0]])
        if hi > lo:
            return [(lo, hi)]
    for lo, hi in spans:
        if hi - lo < 1.0:
            lo, hi = max(0.0, lo - pad_sec), hi + pad_sec
        merged.append((lo, hi))
    return merged


def must_keywords(text: str) -> list[str]:
    """Content words worth matching against a moment description."""
    words = []
    for w in _TOKEN.findall(text or ""):
        if len(w) < 2 or w.isdigit() or w in _STOPWORDS:
            continue
        words.append(w)
    return words


def _mentions(description: str, word: str) -> bool:
    # Korean inflects the tail: 강아지가/강아지를, 넘어지는/넘어짐. Match on the
    # stem by trimming up to two trailing chars.
    # ponytail: prefix heuristic, not a morphological analyzer — swap in kiwipiepy
    # if false positives show up.
    for n in range(len(word), max(2, len(word) - 2) - 1, -1):
        if word[:n] in description:
            return True
    return False


def apply_must(moments: list[dict], must_text: str) -> list[dict]:
    """Mark the moments the creator pointed at. A timecode hit is decisive;
    keyword hits rank by how much of the request they cover. Marked moments
    get brief_fit 10 so every downstream ranker treats them as the main beat."""
    must_text = (must_text or "").strip()
    if not must_text or not moments:
        return moments
    ranges = must_ranges(must_text)
    keywords = must_keywords(must_text)
    scored: list[tuple[float, dict]] = []
    for m in moments:
        start, end = float(m.get("start_sec", 0)), float(m.get("end_sec", 0))
        score = 0.0
        if any(not (end < lo or start > hi) for lo, hi in ranges):
            score += 10.0
        desc = f"{m.get('description', '')} {m.get('mood', '')}"
        hits = sum(1 for w in keywords if _mentions(desc, w))
        if hits:
            score += 6.0 * hits / max(len(keywords), 1)
        m["must_score"] = round(score, 2)
        m["must"] = False
        if score > 0:
            scored.append((score, m))
    if not scored:
        return moments
    best = max(s for s, _ in scored)
    for score, m in scored:
        # Anything close to the best match is part of the same beat.
        if score >= max(best - 1.0, best * 0.6):
            m["must"] = True
            m["brief_fit"] = 10
    return moments


def must_summary(moments: list[dict]) -> str:
    """The marked beats, as a line the screenwriter cannot miss."""
    picked = [m for m in moments if m.get("must")]
    if not picked:
        return ""
    picked.sort(key=lambda m: m.get("start_sec", 0))
    items = "; ".join(
        f"video {m.get('video_index', 0)} {m['start_sec']:.1f}-{m['end_sec']:.1f}s "
        f"({m.get('description', '')[:60]})"
        for m in picked
    )
    return (
        "MANDATORY MAIN BEAT — the creator pointed at these exact moments: "
        f"{items}. At least one of them MUST appear in every variant, as the "
        "hook or the payoff, held longer than any other shot and carrying the "
        "strongest caption/fx. A variant that omits them is rejected."
    )


def presets_for_ui() -> list[dict[str, str]]:
    return [{"id": k, **v} for k, v in CHANNELS.items()]


if __name__ == "__main__":
    b = normalize({"channel": "parenting", "why": "첫 걸음"})
    assert b["hunt"].startswith("가장 귀엽"), b
    text = compile(b)
    assert "첫 걸음" in text and "육아" in text
    raw = normalize("재밌는 자막")
    assert "재밌는 자막" in compile(raw)

    assert must_ranges("12:30~14:00") == [(750.0, 840.0)], must_ranges("12:30~14:00")
    assert must_ranges("3분 20초쯤") == [(180.0, 220.0)], must_ranges("3분 20초쯤")
    assert must_ranges("강아지 넘어지는 부분") == []
    assert "강아지" in must_keywords("강아지 넘어지는 부분을 메인으로")
    assert "부분" not in must_keywords("강아지 넘어지는 부분을 메인으로")

    ms = [
        {"start_sec": 10, "end_sec": 14, "description": "강아지가 계단에서 넘어짐"},
        {"start_sec": 700, "end_sec": 704, "description": "밥 먹는 컷"},
        {"start_sec": 760, "end_sec": 766, "description": "케이크 자르기"},
    ]
    apply_must(ms, "12:30~14:00 케이크 부분을 메인으로")
    assert ms[2]["must"] and ms[2]["brief_fit"] == 10, ms[2]
    assert not ms[0]["must"] and not ms[1]["must"], ms
    assert "MANDATORY" in must_summary(ms)

    kw = [{"start_sec": 10, "end_sec": 14, "description": "강아지가 계단에서 넘어짐"},
          {"start_sec": 40, "end_sec": 44, "description": "커피 내리는 컷"}]
    apply_must(kw, "강아지가 넘어지는 걸 강조해줘")
    assert kw[0]["must"] and not kw[1]["must"], kw
    assert "MAIN BEAT" in compile(normalize({"must": "강아지 넘어지는 부분"}))
    print("brief self-check ok")
