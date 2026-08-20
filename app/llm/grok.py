"""Grok 4.6 via xAI. Implementation lives in grok_client."""
from app.llm.grok_client import (  # noqa: F401
    analyze_frames,
    available,
    client,
    complete_json,
    complete_with_tools,
    format_research,
    frames_content,
    parse_search_response,
    research_trends,
    should_research,
)

if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    r = complete_json("Reply with ok=true.", "ping", schema, max_tokens=256, effort="low")
    assert r.get("ok") is True, r
    print("grok self-check ok")
