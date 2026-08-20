"""Non-destructive handoff to Premiere / Resolve.

FCPXML event with three projects: STRINGOUT, SELECTS, ALTS.
Source files are never rewritten. Clips keep source in/out.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape


def fcpxml(index: dict, infos: list[dict], name: str = "xlog") -> str:
    clips = {c["id"]: c for c in index.get("clips") or []}
    assets = []
    for i, info in enumerate(infos):
        path = Path(info.get("path") or "")
        dur = float(info.get("duration_sec") or 0)
        assets.append(
            f'    <asset id="a{i}" name="{escape(path.name)}" '
            f'src="{_href(path)}" start="0s" duration="{_t(dur)}" '
            f'hasVideo="1" hasAudio="1"/>'
        )
    stringout = _seq("STRINGOUT", index.get("stringout") or [], clips, "s1")
    selects = _seq("SELECTS", index.get("selects") or [], clips, "s2")
    alt_ids = []
    for c in clips.values():
        alt_ids.extend(c.get("alts") or [])
    alts = _seq("ALTS", alt_ids, clips, "s3")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE fcpxml>\n'
        '<fcpxml version="1.9">\n'
        "  <resources>\n"
        '    <format id="r1" name="FFVideoFormat1080p30" '
        'frameDuration="1/30s" width="1920" height="1080"/>\n'
        + "\n".join(assets) + "\n"
        "  </resources>\n"
        f'  <library>\n    <event name="{escape(name)} index">\n'
        f"{stringout}{selects}{alts}"
        "    </event>\n  </library>\n</fcpxml>\n"
    )


def marker_csv(index: dict) -> str:
    lines = ["Name,In,Out,Duration,Tags,Flags,Verdict,Note"]
    for c in index.get("clips") or []:
        dur = max(0.0, float(c["end_sec"]) - float(c["start_sec"]))
        lines.append(",".join([
            _csv(c.get("description") or c["id"]),
            c.get("source_in") or "",
            c.get("source_out") or "",
            f"{dur:.2f}",
            _csv("+".join(c.get("tags") or [])),
            _csv("+".join(c.get("flags") or [])),
            c.get("verdict") or "",
            _csv(c.get("source") or ""),
        ]))
    return "\n".join(lines) + "\n"


def _seq(title: str, ids: list[str], clips: dict, pid: str) -> str:
    seen = set()
    ordered = []
    for i in ids:
        if i in seen or i not in clips:
            continue
        seen.add(i)
        ordered.append(clips[i])
    offset = 0.0
    items = []
    for c in ordered:
        dur = max(0.04, float(c["end_sec"]) - float(c["start_sec"]))
        items.append(
            f'          <asset-clip name="{escape((c.get("description") or c["id"])[:80])}" '
            f'ref="a{int(c["video_index"])}" offset="{_t(offset)}" '
            f'start="{_t(c["start_sec"])}" duration="{_t(dur)}" '
            f'audioRole="dialogue"/>'
        )
        offset += dur
    body = "\n".join(items)
    return (
        f'      <project name="{escape(title)}">\n'
        f'        <sequence format="r1" duration="{_t(offset)}">\n'
        "          <spine>\n"
        f"{body}\n"
        "          </spine>\n"
        "        </sequence>\n"
        "      </project>\n"
    )


def _t(sec: float) -> str:
    frames = max(0, int(round(float(sec) * 30)))
    return f"{frames}/30s"


def _href(path: Path) -> str:
    resolved = path.resolve().as_posix()
    if len(resolved) > 1 and resolved[1] == ":":
        return "file://localhost/" + quote(resolved)
    return path.resolve().as_uri()


def _csv(text: str) -> str:
    t = (text or "").replace('"', "'").replace("\n", " ")
    if "," in t:
        return f'"{t}"'
    return t


if __name__ == "__main__":
    from app.pipeline import index as index_mod
    idx = index_mod.build(
        {"moments": [
            {"video_index": 0, "start_sec": 1, "end_sec": 3, "description": "훅",
             "mood": "hype", "intensity": 8, "hook_potential": 8, "brief_fit": 9},
        ]},
        [{"name": "a.mov", "path": "C:/tmp/a.mov", "duration_sec": 10, "silences": []}],
    )
    xml = fcpxml(idx, [{"path": "C:/tmp/a.mov", "duration_sec": 10}])
    assert "STRINGOUT" in xml and "SELECTS" in xml and "1/30s" in xml
    csv = marker_csv(idx)
    assert "Name,In,Out" in csv
    print("premiere export self-check ok")
