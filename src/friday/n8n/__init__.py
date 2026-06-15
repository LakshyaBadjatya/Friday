"""n8n integration (Tier 2; default off).

This package wires FRIDAY to a local `n8n <https://n8n.io>`_ instance so the owner
can say "make a workflow on n8n that ..." and get a MINIMAL valid n8n workflow
drafted (by the LLM), optionally imported into a running n8n via its REST API.

The whole feature is gated behind ``FRIDAY_ENABLE_N8N`` (default false): off, the
``/n8n`` routes 404 and the orchestrator hook is inert.

Three seams, each independently testable offline:

* :class:`~friday.n8n.client.N8nClient` — a thin ``httpx`` adapter over the n8n
  REST API (health check, import-workflow, list-workflows). The API key is held
  as a plain ``str | None`` (sourced from a :class:`~pydantic.SecretStr` in
  config so it never logs) and sent as the ``X-N8N-API-KEY`` header.
* :class:`~friday.n8n.drafter.WorkflowDrafter` — one NON-FATAL LLM pass that
  drafts a workflow JSON + setup notes; any failure degrades to a safe
  single-Manual-Trigger stub (never raises).
* :class:`~friday.n8n.service.N8nService` — orchestrates the confirm-gated docker
  auto-start (argv-only ``create_subprocess_exec``), drafting, and best-effort
  import.
"""

from __future__ import annotations

from friday.n8n.client import N8nClient, N8nError
from friday.n8n.drafter import WorkflowDrafter
from friday.n8n.service import N8nService

__all__ = ["N8nClient", "N8nError", "N8nService", "WorkflowDrafter"]
