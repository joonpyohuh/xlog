"""Runnable check: python -m app.storage.memory_check

Fails if learning does not persist, examples cannot be built for SFT,
YouTube URLs do not parse, or disk cleanup leaves media behind.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.evaluation import finetune, rubric as rubric_store
from app.pipeline import ingest
from app.storage import cleanup, memory


def main() -> None:
    tmp = Path(tempfile.mkdtemp()) / "memory.db"
    memory.init(tmp, migrate=False)

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
                "Caption only what is visible in the frame.",
            ],
            "notes": "learned from pick A",
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
    assert rubric is not None and rubric["version"] == 2, rubric
    prefs = {p["rule"]: p["times_seen"] for p in memory.list_preferences()}
    assert "Caption only what is visible in the frame." in prefs, prefs
    refs = memory.list_references()
    assert refs[0]["notes"] == "자막 타이밍이 좋다", refs[0]

    examples = finetune.rebuild_examples()
    assert len(examples) >= finetune.MIN_EXAMPLES, len(examples)
    kinds = {ex["kind"] for ex in examples}
    assert "judge" in kinds and "writer" in kinds, kinds
    memory.replace_ft_examples(examples)
    assert len(memory.list_ft_examples()) >= finetune.MIN_EXAMPLES

    prompt = rubric_store.rubric_as_prompt()
    assert "Learned editor preferences" not in prompt, prompt
    assert "fine-tuned" in prompt

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

    print("ok", memory.stats(), "examples", len(examples))


if __name__ == "__main__":
    main()
