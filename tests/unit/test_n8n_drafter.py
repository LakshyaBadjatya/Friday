"""Unit tests for :class:`friday.n8n.drafter.WorkflowDrafter` (offline FakeLLM).

Every LLM call runs on a scripted :class:`~friday.providers.llm.FakeLLM` (zero
network). Covered:

* A valid scripted draft -> a workflow dict (name/nodes/connections) + setup_notes.
* A bad-then-good script -> the bounded repair pass recovers it.
* A bad-then-bad script -> the safe single-Manual-Trigger STUB (non-fatal, never
  raises) plus a drafting-failed setup note.
* A provider error -> the stub (drafting is NON-FATAL).
"""

from __future__ import annotations

import json

from friday.errors import ProviderError
from friday.n8n.drafter import WorkflowDrafter
from friday.providers.llm import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolSpec,
    Usage,
)

_VALID_DRAFT = json.dumps(
    {
        "workflow": {
            "name": "Slack on webhook",
            "nodes": [
                {
                    "id": "wh",
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhook",
                    "position": [250, 300],
                    "parameters": {"path": "incoming"},
                },
                {
                    "id": "slack",
                    "name": "Slack",
                    "type": "n8n-nodes-base.slack",
                    "position": [500, 300],
                    "parameters": {"channel": "#general"},
                },
            ],
            "connections": {
                "Webhook": {"main": [[{"node": "Slack", "type": "main", "index": 0}]]}
            },
        },
        "setup_notes": ["Add Slack OAuth2 credentials", "Set the webhook path"],
    }
)


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


class _BoomLLM(LLMProvider):
    """An LLM that always raises a provider error (drafting must stay non-fatal)."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        raise ProviderError("backend down")


async def test_draft_valid_returns_workflow_and_notes() -> None:
    from friday.providers.llm import FakeLLM

    drafter = WorkflowDrafter(FakeLLM(responses=[_resp(_VALID_DRAFT)]))

    result = await drafter.draft("when a webhook fires, post to Slack")

    workflow = result["workflow"]
    assert workflow["name"] == "Slack on webhook"
    assert {n["name"] for n in workflow["nodes"]} == {"Webhook", "Slack"}
    # Every node carries the five required keys.
    for node in workflow["nodes"]:
        assert set(node) >= {"id", "name", "type", "position", "parameters"}
    assert "connections" in workflow
    assert result["setup_notes"] == [
        "Add Slack OAuth2 credentials",
        "Set the webhook path",
    ]


async def test_draft_prose_wrapped_json_is_parsed() -> None:
    from friday.providers.llm import FakeLLM

    wrapped = f"Sure! Here is the workflow:\n```json\n{_VALID_DRAFT}\n```\nEnjoy."
    drafter = WorkflowDrafter(FakeLLM(responses=[_resp(wrapped)]))

    result = await drafter.draft("post to Slack")

    assert result["workflow"]["name"] == "Slack on webhook"


async def test_draft_bad_then_good_recovers_via_repair() -> None:
    from friday.providers.llm import FakeLLM

    # First reply is junk (no valid object); the bounded repair returns valid JSON.
    drafter = WorkflowDrafter(
        FakeLLM(responses=[_resp("not json at all"), _resp(_VALID_DRAFT)])
    )

    result = await drafter.draft("post to Slack")

    assert result["workflow"]["name"] == "Slack on webhook"
    # The recovered draft is NOT the stub.
    assert len(result["workflow"]["nodes"]) == 2


async def test_draft_bad_then_bad_falls_back_to_stub() -> None:
    from friday.providers.llm import FakeLLM

    drafter = WorkflowDrafter(
        FakeLLM(responses=[_resp("garbage"), _resp("still garbage")])
    )

    result = await drafter.draft("do something vague")

    workflow = result["workflow"]
    # Safe stub: exactly one Manual Trigger node.
    assert len(workflow["nodes"]) == 1
    assert workflow["nodes"][0]["type"] == "n8n-nodes-base.manualTrigger"
    assert workflow["connections"] == {}
    # A note explains drafting fell back.
    assert any("stub" in note.lower() for note in result["setup_notes"])


async def test_draft_missing_nodes_is_invalid_then_stub() -> None:
    from friday.providers.llm import FakeLLM

    # A workflow object with an EMPTY nodes list violates the min_length shape;
    # both passes return it, so the drafter falls back to the stub (never raises).
    bad = json.dumps(
        {"workflow": {"name": "empty", "nodes": [], "connections": {}},
         "setup_notes": []}
    )
    drafter = WorkflowDrafter(FakeLLM(responses=[_resp(bad), _resp(bad)]))

    result = await drafter.draft("x")

    assert len(result["workflow"]["nodes"]) == 1
    assert result["workflow"]["nodes"][0]["type"] == "n8n-nodes-base.manualTrigger"


async def test_draft_provider_error_is_non_fatal_stub() -> None:
    drafter = WorkflowDrafter(_BoomLLM())

    result = await drafter.draft("anything")

    # No raise; safe stub returned.
    assert result["workflow"]["nodes"][0]["type"] == "n8n-nodes-base.manualTrigger"
    assert result["setup_notes"]
