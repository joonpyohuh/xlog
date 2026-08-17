"""Runnable check: python -m app.storage.memory_check

Fails if learning does not persist, learned taste does not reach the
prompts, YouTube URLs do not parse, or disk cleanup leaves media behind.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from app import config
from app.evaluation import rubric as rubric_store, taste
from app.pipeline import ingest
from app.storage import cleanup, memory


def main() -> None:
    tmp = Path(tempfile.mkdtemp()) / "memory.db"
    memory.init(tmp, migrate=False, force_sqlite=True)

    memory.save_rubric(
        {
            "version": 1,
            "owner": "check",
            "criteria": [{"name": "hook_strength", "weight": 25, "description": "hook"}],
            "preferences": ["Open cold on the highest-action moment."],
            "notes": "seed",
        },
        source="seed",
    )
    variants = [
        {
            "label": "A",
            "concept": "facts in frame",
            "shots": [
                {
                    "video_index": 0, "start_sec": 0, "end_sec": 3,
                    "role": "hook", "reason": "face + bruise",
                    "caption": "이게 끝인가", "caption_style": "normal",
                }
            ],
        },
        {
            "label": "B",
            "concept": "atmosphere",
            "shots": [
                {
                    "video_index": 0, "start_sec": 1, "end_sec": 4,
                    "role": "hook", "reason": "empty hallway",
                    "caption": "조명은 좁아지고", "caption_style": "normal",
                }
            ],
        },
    ]
    memory.record_feedback(
        job_id="job_1",
        user_choice="A",
        user_comment="A shows the facts in frame",
        judge_winner="B",
        agreement=False,
        variants=variants,
        judge_verdict={"winner": "B"},
        rubric_version=2,
    )
    memory.save_rubric(
        {
            "version": 2,
            "owner": "check",
            "criteria": [{"name": "hook_strength", "weight": 25, "description": "hook"}],
            "preferences": [
                "Open cold on the highest-action moment.",
            ],
            "notes": "drop one-off rule",
        },
        source="feedback",
    )
    memory.save_rubric(
        {
            "version": 3,
            "owner": "check",
            "criteria": [{"name": "hook_strength", "weight": 27, "description": "hook"}],
            "preferences": [
                "Open cold on the highest-action moment.",
                "Caption only what is visible in the frame.",
            ],
            "notes": "rule confirmed on a second pick",
        },
        source="feedback",
    )
    memory.record_reference(
        url="https://youtube.com/shorts/example",
        notes="자막 타이밍이 좋다",
        file_path=None,
        style={
            "style_summary": "narration-driven letterboxed gameplay",
            "hook_technique": "cold open",
            "pacing": "subtitle turnover",
            "caption_style": "lower-third white",
            "structure": "hook → development → meme → payoff",
            "learned_rules": [
                "Caption only what is visible in the frame.",
                "Swap one short caption line every 1-2 seconds.",
            ],
        },
        rubric_version=3,
    )

    rubric = memory.load_rubric()
    assert rubric is not None and rubric["version"] == 3, rubric
    prefs = {p["rule"]: p["times_seen"] for p in memory.list_preferences()}
    assert prefs.get("Caption only what is visible in the frame.", 0) >= 2, prefs
    assert prefs.get("Open cold on the highest-action moment.") == 1, prefs
    refs = memory.list_references()
    assert refs[0]["notes"] == "자막 타이밍이 좋다", refs[0]

    # one-off rules stay out of the prompt; confirmed rules get in, capped
    rules = taste.taste_rules()
    assert rules, "no learned rules reached the prompt"
    assert len(rules) <= config.TASTE_RULES_IN_PROMPT, rules
    assert "Caption only what is visible in the frame." in rules, rules
    assert "Open cold on the highest-action moment." not in rules, rules
    assert taste.taste_prompt() in taste.writer_system(), "writer lost learned taste"

    learning = taste.status()
    assert learning["engine"] == "gemini+grok+gpt", learning
    assert "grok" in config.GROK_MODEL, config.GROK_MODEL
    assert "gemini" in config.GEMINI_MODEL, config.GEMINI_MODEL
    assert learning["vision_model"] == config.GEMINI_MODEL, learning
    assert learning["model"] == config.GROK_MODEL, learning
    assert learning["picks"] == 1 and learning["rubric_version"] == 3, learning
    assert learning["recent"][0]["choice"] == "A", learning["recent"]
    assert learning["rules_in_prompt"] == ["Caption only what is visible in the frame."], learning

    from app.pipeline import highlight
    windows = highlight._candidate_windows(
        [{"start_sec": 140, "end_sec": 155, "intensity": 9, "hook_potential": 8}],
        300.0,
    )
    assert windows and windows[0][0] <= 140 and windows[0][1] >= 155, windows

    from app.pipeline import verify
    tiny = [{
        "label": "A",
        "shots": [
            {"video_index": 0, "start_sec": s, "end_sec": s + 1.0, "caption": "c",
             "caption_style": "pop", "fx": "none"}
            for s in (5.0, 40.0, 90.0)
        ],
    }]
    fixed = verify.validate_variants(tiny, [{"duration_sec": 300.0}])[0]
    assert fixed["total_sec"] >= config.SHORT_FLOOR_SEC, fixed["total_sec"]
    spans = sorted((s["start_sec"], s["end_sec"]) for s in fixed["shots"])
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start >= prev_end - config.MAX_SHOT_OVERLAP_SEC, spans

    from app.pipeline import narrate
    spoken = narrate._lines({"shots": [
        {"start_sec": 0, "end_sec": 3, "caption": "hi", "caption_style": "pop"},
        {"start_sec": 9, "end_sec": 13, "caption": "", "caption_style": "pop"},
        {"start_sec": 20, "end_sec": 24, "caption": "bye", "caption_style": "plate"},
    ]})
    assert [round(o, 2) for o, _, _ in spoken] == [0.0, 7.0], spoken

    from app.pipeline import brief as brief_mod, highlight
    parenting = brief_mod.normalize({"channel": "parenting", "why": "첫 걸음"})
    assert "귀엽" in parenting["hunt"], parenting
    compiled = brief_mod.compile(parenting)
    assert "첫 걸음" in compiled and "육아" in compiled
    gaming = brief_mod.normalize({"channel": "gaming", "why": "클러치"})
    assert "클러치" in gaming["hunt"] or "킬" in gaming["hunt"]
    assert highlight._score({"brief_fit": 9, "hook_potential": 2, "intensity": 1}) > highlight._score(
        {"brief_fit": 2, "hook_potential": 9, "intensity": 9}
    )

    prompt = rubric_store.rubric_as_prompt()
    assert "learned, v3" in prompt, prompt
    assert "essay" not in prompt.lower()
    assert "stop the scroll" in prompt or "first 3 seconds" in prompt, prompt

    args = ingest._ytdlp_extra_args()
    if shutil.which("node"):
        joined = " ".join(args)
        assert "--js-runtimes" in joined and "node" in joined, args

    urls = ingest.parse_youtube_urls(
        "https://youtube.com/shorts/abc123\nhttps://youtu.be/xyz789"
    )
    assert len(urls) == 2, urls
    try:
        ingest.parse_youtube_urls("not-a-video")
        raise AssertionError("expected IngestError")
    except ingest.IngestError:
        pass

    work = Path(tempfile.mkdtemp())
    (work / "job.json").write_text("{}", encoding="utf-8")
    (work / "short_A.mp4").write_bytes(b"x")
    (work / "frames_0").mkdir()
    (work / "frames_0" / "f.jpg").write_bytes(b"y")
    cleanup.purge_work_media(work)
    leftover = {p.name for p in work.iterdir()}
    assert leftover == {"job.json"}, leftover

    print("ok", memory.stats(), "rules in prompt", len(rules))


if __name__ == "__main__":
    main()
