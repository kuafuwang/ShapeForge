"""Chat transcript lines keep hashes by default instead of the full prompt and reply bodies."""

from __future__ import annotations

import json

from taic.llm import chat_transcript_event


def _event(keep: bool) -> dict:
    return chat_transcript_event(
        label="init", model="m", seconds=1.5,
        messages=[{"role": "user", "content": "SECRET PROMPT"}],
        reply_message={"role": "assistant", "content": "SECRET REPLY"},
        finish_reason="stop", usage=None, cost_usd=0.1, total_usd=0.1, keep_bodies=keep,
    )


def test_the_default_event_has_hashes_and_no_bodies():
    event = _event(False)
    assert "messages" not in event and "reply" not in event
    assert "SECRET" not in json.dumps(event)
    assert event["messages_sha256"] and event["reply_sha256"]
    assert event["messages_bytes"] > 0 and event["reply_bytes"] > 0


def test_the_opt_in_event_keeps_the_old_shape():
    event = _event(True)
    assert event["messages"][0]["content"] == "SECRET PROMPT"
    assert event["reply"]["content"] == "SECRET REPLY"
