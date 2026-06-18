"""Siri front-door package: voice-shaping helpers for the ``/siri/ask`` route.

Thin support code for the Siri Shortcuts integration — turning the core
orchestrator's reply into something Siri can speak. The HTTP surface lives in
:mod:`friday.api.routes_siri`; the brain is the same :class:`Orchestrator` that
backs ``/chat``.
"""

from __future__ import annotations

from friday.siri.speech import for_speech

__all__ = ["for_speech"]
