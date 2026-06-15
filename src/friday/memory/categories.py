"""Memory categorization: category/tier taxonomy and a categorized fact model.

This module is pure and dependency-injection friendly — it imports nothing from
``friday.config`` or ``friday.app``.

Two enumerations classify a stored memory:

* :class:`MemoryCategory` — *what kind* of thing the memory is (a fact, a stated
  preference, a decision, a task, a person, a project, an event).
* :class:`MemoryTier` — *how hot* the memory is in a tiered store
  (``HOT`` recently/frequently used, ``WARM`` aging, ``COLD`` archival).

:class:`CategorizedFact` is the pydantic v2 record carrying a memory's text plus
its category, tier, namespace, soft-delete flag, and outgoing links to related
records.

:func:`soft_delete` marks a record deleted without removing it — *except* for
:attr:`MemoryCategory.DECISION` records, which are an immutable audit trail and
may never be deleted; attempting to delete one raises
:class:`UndeletableDecisionError`.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field

from friday.errors import FridayError


class MemoryCategory(enum.StrEnum):
    """The kind of thing a memory record represents."""

    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    TASK = "task"
    PERSON = "person"
    PROJECT = "project"
    EVENT = "event"


class MemoryTier(enum.StrEnum):
    """Storage tier reflecting how actively a memory is used."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class UndeletableDecisionError(FridayError):
    """Raised when a caller attempts to soft-delete a ``DECISION`` record.

    Decisions form an immutable audit trail; they are never deletable.
    """


class CategorizedFact(BaseModel):
    """A single categorized memory record.

    Args:
        text: The natural-language content of the memory.
        category: What kind of memory this is.
        tier: The storage tier (defaults to :attr:`MemoryTier.HOT`).
        namespace: Logical partition the record belongs to (e.g. a user or
            workspace id). Defaults to ``"default"``.
        deleted: Soft-delete flag. ``True`` means logically removed but still
            present. Defaults to ``False``.
        links: Outgoing links to related record ids/keys. Defaults to empty.
    """

    text: str
    category: MemoryCategory
    tier: MemoryTier = MemoryTier.HOT
    namespace: str = "default"
    deleted: bool = False
    links: list[str] = Field(default_factory=list)


def soft_delete(record: CategorizedFact) -> CategorizedFact:
    """Mark ``record`` as soft-deleted, returning the updated record.

    The record is mutated in place (``deleted`` set to ``True``) and also
    returned for convenience/chaining.

    Args:
        record: The record to soft-delete.

    Returns:
        The same record instance with ``deleted=True``.

    Raises:
        UndeletableDecisionError: If ``record.category`` is
            :attr:`MemoryCategory.DECISION`. Decisions are never deletable.
    """
    if record.category is MemoryCategory.DECISION:
        raise UndeletableDecisionError(
            "DECISION records are an immutable audit trail and cannot be deleted"
        )
    record.deleted = True
    return record


def is_deletable(record: CategorizedFact) -> bool:
    """Return whether ``record`` may be soft-deleted (i.e. is not a DECISION)."""
    return record.category is not MemoryCategory.DECISION
