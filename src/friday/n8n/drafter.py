"""LLM-driven drafting of a MINIMAL valid n8n workflow JSON (Tier 2).

:class:`WorkflowDrafter` turns a free-text description ("when a webhook fires,
send me a Slack message") into a minimal valid n8n workflow object plus a list of
human setup notes (credentials / nodes the owner must still configure). It depends
only on the typed :class:`~friday.providers.llm.LLMProvider` boundary, so it
imports no SDK and runs fully offline against a scripted
:class:`~friday.providers.llm.FakeLLM` in tests.

Binding rules (mirrors the graph extractor's NON-FATAL contract):

* Exactly one LLM pass asks for a single strict-JSON object of the shape
  ``{"workflow": {...}, "setup_notes": [...]}``. The ``workflow`` must have
  ``name``, ``nodes`` (each ``{id, name, type, position, parameters}``) and
  ``connections``.
* The result is parsed + shape-validated. On a parse/shape failure ONE bounded
  repair pass is attempted (the model is shown its broken output and asked to fix
  it). If the repair also fails, :meth:`draft` returns a SAFE STUB workflow — a
  single Manual Trigger node — plus a note explaining drafting failed.
* :meth:`draft` NEVER raises. Any provider error/timeout, empty output, or
  un-repairable shape degrades to the stub, so a draft can be attempted on every
  turn without risk.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.n8n.drafter")

# Instruction handed to the LLM. It asks for one strict-JSON object so the parse
# is deterministic; any deviation trips the bounded repair, then the stub.
_DRAFT_INSTRUCTION = (
    "You design n8n workflows. Given a description, reply with a SINGLE JSON "
    "object and nothing else, of the exact shape "
    '{"workflow": {"name": str, "nodes": [{"id": str, "name": str, "type": str, '
    '"position": [int, int], "parameters": object}], "connections": object}, '
    '"setup_notes": [str]}. '
    "Keep the workflow MINIMAL but valid: include a trigger node and only the "
    "nodes the description clearly implies. Use real n8n node type strings (e.g. "
    '"n8n-nodes-base.manualTrigger", "n8n-nodes-base.webhook", '
    '"n8n-nodes-base.httpRequest", "n8n-nodes-base.slack"). connections maps a '
    "source node NAME to its outgoing links (an empty object is fine for a "
    "single node). setup_notes lists, in plain language, the credentials or node "
    "settings the user must still configure (e.g. 'Add Slack OAuth2 "
    "credentials'). Return an empty setup_notes list if none are needed.\n\n"
    "Description:\n"
)

_REPAIR_INSTRUCTION = (
    "Your previous reply was not a single valid JSON object of the required "
    "shape {\"workflow\": {\"name\", \"nodes\", \"connections\"}, "
    "\"setup_notes\": [...]}. Reply again with ONLY the corrected JSON object, no "
    "prose, no code fence.\n\nYour previous reply was:\n"
)


class WorkflowNode(BaseModel):
    """One n8n node: the five keys a minimal valid node needs.

    ``extra`` keys are ignored so a chatty model that adds fields still parses;
    the five named fields are required so a malformed node trips the repair/stub.
    """

    model_config = {"extra": "ignore"}

    id: str
    name: str
    type: str
    position: list[float] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class Workflow(BaseModel):
    """The minimal valid n8n workflow shape the drafter guarantees.

    Requires a non-empty ``nodes`` list and a ``connections`` object; ``name``
    defaults so a model that omits it still parses (it is harmless). A malformed
    node raises :class:`~pydantic.ValidationError` and trips the repair/stub.
    """

    model_config = {"extra": "ignore"}

    name: str = "Untitled workflow"
    nodes: list[WorkflowNode] = Field(min_length=1)
    connections: dict[str, Any] = Field(default_factory=dict)


class DraftResult(BaseModel):
    """The strict shape one LLM draft pass is parsed into."""

    model_config = {"extra": "ignore"}

    workflow: Workflow
    setup_notes: list[str] = Field(default_factory=list)


class WorkflowDrafter:
    """Draft a minimal valid n8n workflow + setup notes from a description.

    Args:
        llm: The live LLM provider used for the single draft pass (and one
            bounded repair pass on a shape failure).
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def draft(self, description: str) -> dict[str, Any]:
        """One LLM pass (plus one bounded repair) for a workflow; never raise.

        Returns ``{"workflow": <dict>, "setup_notes": [<str>, ...]}``. On any
        un-repairable failure the workflow is a safe single-Manual-Trigger stub
        and ``setup_notes`` carries a note explaining drafting failed.
        """
        prompt = _DRAFT_INSTRUCTION + description
        raw = await self._complete(prompt)
        parsed = _try_parse(raw)
        if parsed is not None:
            return _to_dict(parsed)

        # One bounded repair: show the model its broken output and ask for a fix.
        repair_prompt = _REPAIR_INSTRUCTION + (raw or "(no output)")
        repaired_raw = await self._complete(repair_prompt)
        repaired = _try_parse(repaired_raw)
        if repaired is not None:
            logger.info("n8n workflow draft recovered on the repair pass")
            return _to_dict(repaired)

        logger.warning(
            "n8n workflow drafting failed after repair; returning safe stub"
        )
        return _stub_result(description)

    async def _complete(self, prompt: str) -> str:
        """Run one completion; return the stripped text, or ``""`` on any error.

        Provider errors/timeouts are swallowed (returning ``""``) so drafting
        stays NON-FATAL: an empty string simply fails the parse and falls through
        to the repair / stub path.
        """
        try:
            response = await self._llm.complete([Message(role="user", content=prompt)])
        except Exception:  # noqa: BLE001 - drafting is NON-FATAL by contract
            logger.warning("n8n workflow draft LLM call failed")
            return ""
        return (response.text or "").strip()


def _try_parse(raw: str) -> DraftResult | None:
    """Parse ``raw`` into a :class:`DraftResult`, or ``None`` on any failure.

    Slices the first ``{...}`` object (tolerating prose / a code fence) and
    validates the strict shape; any JSON or shape error yields ``None`` so the
    caller falls through to the repair / stub path.
    """
    if not raw:
        return None
    try:
        return DraftResult.model_validate_json(_extract_json(raw))
    except (ValidationError, ValueError):
        return None


def _to_dict(result: DraftResult) -> dict[str, Any]:
    """Render a validated :class:`DraftResult` as plain dicts for callers."""
    return {
        "workflow": result.workflow.model_dump(),
        "setup_notes": list(result.setup_notes),
    }


def _stub_result(description: str) -> dict[str, Any]:
    """A safe single-Manual-Trigger workflow + a drafting-failed setup note.

    Used when the LLM could not produce a valid workflow even after the repair
    pass. The stub is a real, importable n8n workflow (one Manual Trigger node),
    so the owner gets a usable starting point rather than an error.
    """
    name = description.strip() or "FRIDAY workflow"
    workflow: dict[str, Any] = {
        "name": f"FRIDAY draft: {name}"[:120],
        "nodes": [
            {
                "id": "manual-trigger",
                "name": "Start",
                "type": "n8n-nodes-base.manualTrigger",
                "position": [250, 300],
                "parameters": {},
            }
        ],
        "connections": {},
    }
    return {
        "workflow": workflow,
        "setup_notes": [
            "Drafting fell back to a safe stub (a single Manual Trigger). Open it "
            "in n8n and add the nodes you need, then connect them."
        ],
    }


def _extract_json(text: str) -> str:
    """Return the first ``{...}`` JSON object substring of ``text``.

    Tolerates a model that wraps the JSON in prose or a ```` ```json ```` fence by
    slicing from the first ``{`` to the last ``}``. When no braces are present the
    original text is returned so the downstream parse fails into the repair/stub
    path rather than silently succeeding on garbage. Mirrors the graph-extractor
    helper.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]
