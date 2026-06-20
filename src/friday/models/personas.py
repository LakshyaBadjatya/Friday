# © Lakshya Badjatya — Author
"""Persona → free-model assignment.

Each roster persona (the prime FRIDAY plus the eight specialists) gets its own
*distinct* free model from the :class:`~friday.models.catalog.ModelCatalog`, so
addressing — or auto-delegating to — a specialist runs that turn on a different
brain. The mapping is **pure data + a resolver**: it imports no LLM SDK and does
no I/O, mirroring :mod:`friday.models.catalog`.

:data:`PERSONA_MODEL_PREFERENCES` lists, per persona code-name, an *ordered*
preference of catalog ids (role-matched primary first, then fall-backs).
:func:`resolve_persona_models` walks the roster in declaration order and greedily
hands each persona the first **available** id it prefers that is not already
taken, so the result is distinct whenever the build has enough free models wired
(nine or more). When a persona's whole preference list is unavailable/taken it is
simply omitted from the result — the orchestrator then leaves that turn on the
gateway's active/default model, so the feature degrades gracefully rather than
mis-routing.
"""

from __future__ import annotations

from collections.abc import Iterable

from friday.models.catalog import ModelCatalog
from friday.roster.definitions import ROSTER_PERSONAS, Persona

# Catalog ids referenced below, kept as named constants so a typo surfaces here
# (one place) rather than as a silent "always falls back" at runtime. These mirror
# the ids in :data:`friday.models.catalog.DEFAULT_CATALOG`.
_OR_GPT_OSS_120B = "openrouter:openai/gpt-oss-120b:free"
_OR_GPT_OSS_20B = "openrouter:openai/gpt-oss-20b:free"
_OR_GEMMA_31B = "openrouter:google/gemma-4-31b-it:free"
_OR_NEMOTRON_NANO_9B = "openrouter:nvidia/nemotron-nano-9b-v2:free"
_OR_QWEN3_CODER = "openrouter:qwen/qwen3-coder:free"
_OR_LLAMA_33_70B = "openrouter:meta-llama/llama-3.3-70b-instruct:free"
_OR_HERMES_405B = "openrouter:nousresearch/hermes-3-llama-3.1-405b:free"
_OC_MIMO = "opencode:mimo-v2.5-free"
_OC_NEMOTRON_ULTRA = "opencode:nemotron-3-ultra-free"
_OC_DEEPSEEK = "opencode:deepseek-v4-flash-free"
_NV_LLAMA_8B = "nvidia:meta/llama-3.1-8b-instruct"


#: Ordered model preference per persona code-name. The primary (first) entry is
#: the role-matched ideal; the rest are fall-backs that keep assignments distinct
#: when a provider is missing. When all three free providers are wired the greedy
#: resolver yields nine distinct models (see :func:`resolve_persona_models`).
PERSONA_MODEL_PREFERENCES: dict[str, tuple[str, ...]] = {
    # Prime — the broadest, most capable general model.
    "FRIDAY": (_OR_GPT_OSS_120B, _OR_LLAMA_33_70B, _OC_NEMOTRON_ULTRA),
    # Security: precise, fail-closed.
    "EDITH": (_OR_GPT_OSS_20B, _OR_GEMMA_31B, _NV_LLAMA_8B),
    # Automation/scheduling: fast, idempotent step planning.
    "ORACLE": (_OR_NEMOTRON_NANO_9B, _OR_GPT_OSS_20B, _OC_MIMO),
    # Finance/markets: numerate generalist.
    "GECKO": (_OR_GEMMA_31B, _OR_GPT_OSS_20B, _OC_MIMO),
    # Comms: light, quick drafting.
    "KAREN": (_OC_MIMO, _OR_NEMOTRON_NANO_9B, _OR_GEMMA_31B),
    # Content/outreach: engaging long-form writing.
    "VERONICA": (_OC_DEEPSEEK, _OR_GEMMA_31B, _OR_HERMES_405B),
    # Memory/knowledge/RAG: large model for grounded recall.
    "JOCASTA": (_OR_HERMES_405B, _OR_LLAMA_33_70B, _OC_DEEPSEEK),
    # Research/analysis: strong step-by-step reasoning.
    "VISION": (_OR_LLAMA_33_70B, _OR_GPT_OSS_120B, _OC_NEMOTRON_ULTRA),
    # Development/system: code-tuned model.
    "FORGE": (_OR_QWEN3_CODER, _OR_GPT_OSS_20B, _OC_DEEPSEEK),
}


def resolve_persona_models(
    catalog: ModelCatalog,
    personas: Iterable[Persona] = ROSTER_PERSONAS,
) -> dict[str, str]:
    """Assign every roster operator a distinct available free model.

    Two passes over ``personas`` (declaration order, prime first), so the result
    is distinct and covers **custom operators** the same as the built-ins:

    1. *Preference pass* — each persona that has a role-matched
       :data:`PERSONA_MODEL_PREFERENCES` list (the built-ins) takes the first id
       in that list which is both *available* in ``catalog`` and not already
       taken.
    2. *Leftover pass* — any persona still unassigned (a custom operator with no
       preferences, or a built-in whose whole list was already taken) takes the
       next still-free available id, in catalog order.

    A persona that can be served no free id (the catalog is exhausted) is omitted;
    the caller treats a missing persona as "use the gateway's active/default
    model", so the feature degrades gracefully rather than mis-routing.

    Args:
        catalog: The runtime model catalog. Only its available (listed) ids are
            eligible, so a persona is never assigned a model the build cannot
            serve.
        personas: The operators to assign, in order. Defaults to the canonical
            built-in roster; callers pass the *merged* roster (built-ins + custom
            operators) so custom agents get a distinct model too.

    Returns:
        A mapping of persona code-name -> catalog id, with at most one id per
        persona and no id assigned twice.
    """
    available_order = catalog.ids()  # available ids, in catalog order
    available = set(available_order)
    roster = list(personas)
    assigned: dict[str, str] = {}
    taken: set[str] = set()
    # Pass 1: role-matched preferences (the built-ins).
    for persona in roster:
        for model_id in PERSONA_MODEL_PREFERENCES.get(persona.name, ()):
            if model_id in available and model_id not in taken:
                assigned[persona.name] = model_id
                taken.add(model_id)
                break
    # Pass 2: leftovers for anyone still unassigned (custom operators, or built-ins
    # whose preferences were all taken) — keeps every operator on its own model.
    for persona in roster:
        if persona.name in assigned:
            continue
        for model_id in available_order:
            if model_id not in taken:
                assigned[persona.name] = model_id
                taken.add(model_id)
                break
    return assigned
