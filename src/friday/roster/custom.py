# © Lakshya Badjatya — Author
"""User-declared custom operators that extend the built-in persona roster.

FRIDAY ships with a frozen roster — the prime plus eight least-privilege
specialists (see :mod:`friday.roster.definitions`). This module lets the owner
declare *extra* operators in a compact, pipe-delimited mini-format without
touching code: each entry becomes one ordinary :class:`Persona`, validated by
the same frozen model the built-ins use. There is no new behaviour and no new
dependency — a custom operator is the same pure-data value object, merged into
the roster alongside the built-ins.

The mini-format is five pipe-delimited fields::

    NAME|Title|tool1,tool2|namespace|system prompt text

The third field is a comma-separated tool allow-list; the fourth (``namespace``)
defaults to ``name.lower()`` when left empty. :func:`parse_custom_operators`
turns a list of such entries into :class:`Persona` objects (skipping blank
lines, raising a clear :class:`ValueError` on a malformed one), and
:func:`merge_personas` folds them into the built-in tuple — a custom whose name
collides (case-insensitively) with a built-in is dropped so the built-ins always
win. This module imports nothing from :mod:`friday.config` or :mod:`friday.app`
and reads no settings itself; the caller passes the raw entries in.
"""

from __future__ import annotations

from friday.roster.definitions import Persona

# The mini-format is exactly five pipe-delimited fields.
_FIELD_COUNT = 5


def parse_custom_operators(raw: list[str]) -> list[Persona]:
    """Parse pipe-delimited custom-operator entries into :class:`Persona` objects.

    Each non-blank entry must have exactly five ``|``-delimited fields::

        NAME|Title|tool1,tool2|namespace|system prompt text

    Fields are individually whitespace-trimmed. The tool field is comma-split
    into a non-empty allow-list. The namespace field defaults to
    ``name.lower()`` when empty. Blank entries (empty or whitespace-only) are
    skipped.

    Args:
        raw: The raw entries, one mini-format string per custom operator.

    Returns:
        The parsed personas, in input order.

    Raises:
        ValueError: If an entry is malformed — the wrong field count, an empty
            name, or an empty tool allow-list. The remaining field rules are
            enforced by the :class:`Persona` validators (which raise on, e.g., an
            empty title or system prompt).
    """
    personas: list[Persona] = []
    for entry in raw:
        if not entry.strip():
            continue
        fields = [field.strip() for field in entry.split("|")]
        if len(fields) != _FIELD_COUNT:
            raise ValueError(
                f"custom operator must have {_FIELD_COUNT} pipe-delimited fields "
                f"(NAME|Title|tools|namespace|prompt), got {len(fields)}: {entry!r}"
            )
        name, title, tools_field, namespace_field, system_prompt = fields
        if not name:
            raise ValueError(f"custom operator has an empty name: {entry!r}")
        tools = [tool.strip() for tool in tools_field.split(",") if tool.strip()]
        if not tools:
            raise ValueError(
                f"custom operator {name!r} has an empty tool allow-list: {entry!r}"
            )
        namespace = namespace_field or name.lower()
        personas.append(
            Persona(
                name=name,
                title=title,
                allowed_tools=frozenset(tools),
                memory_namespace=namespace,
                system_prompt=system_prompt,
            )
        )
    return personas


def merge_personas(
    builtins: tuple[Persona, ...], customs: list[Persona]
) -> tuple[Persona, ...]:
    """Merge custom operators into the built-in roster, built-ins winning ties.

    A custom persona whose name collides (case-insensitively) with a built-in is
    dropped — the built-ins are authoritative and can never be shadowed. The
    surviving customs are appended after the built-ins in their given order.

    Args:
        builtins: The canonical built-in personas, in declaration order.
        customs: The parsed custom operators to fold in.

    Returns:
        The built-ins followed by every non-colliding custom, in order.
    """
    taken = {persona.name.casefold() for persona in builtins}
    merged: list[Persona] = list(builtins)
    for persona in customs:
        key = persona.name.casefold()
        if key in taken:
            continue
        taken.add(key)
        merged.append(persona)
    return tuple(merged)
