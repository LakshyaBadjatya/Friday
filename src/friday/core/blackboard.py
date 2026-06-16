# © Lakshya Badjatya — Author
"""A shared blackboard: the scratchpad multi-operator debate writes to and reads.

When several operators draft an answer to the same turn (the ensemble / debate
path in :mod:`friday.core.ensemble`), each draft is *posted* to a shared
:class:`Blackboard` keyed by the operator's name. A later round — or the judge
that synthesizes a final answer — can then read every operator's contribution
back without the drafts having to thread through each other's call signatures.

The blackboard is a pure, in-memory value store: it imports no LLM SDK, reads no
configuration, performs no I/O, and never mutates a draft once posted (a
:class:`Draft` is frozen). It is the ``core/`` debate's only shared mutable
state, deliberately tiny so one debate round stays trivial to reason about and
to test.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Draft(BaseModel):
    """One operator's contribution to a debate round, posted to the blackboard.

    Frozen once created: the blackboard appends drafts and only ever reads them
    back, so a posted draft can never be mutated out from under a later reader.

    Attributes:
        operator: The code-name of the operator that wrote this draft (e.g.
            ``"VISION"``); drafts are grouped and looked up by this name.
        content: The draft answer text.
        round: The zero-based debate round this draft belongs to. Multiple rounds
            let a later draft react to earlier ones; a single-round debate uses
            ``0`` throughout.
    """

    model_config = ConfigDict(frozen=True)

    operator: str
    content: str
    round: int = 0


class Blackboard:
    """An ordered, append-only store of debate :class:`Draft`s.

    Posts preserve insertion order and reads return copies, so iterating the
    board can never mutate it. Lookups by operator are case-insensitive (the
    roster resolves persona names case-insensitively too).
    """

    def __init__(self) -> None:
        self._drafts: list[Draft] = []

    def post(self, operator: str, content: str, *, round: int = 0) -> Draft:
        """Append a draft from ``operator`` (in ``round``) and return it."""
        draft = Draft(operator=operator, content=content, round=round)
        self._drafts.append(draft)
        return draft

    def drafts(self) -> list[Draft]:
        """Return every posted draft in insertion order (a fresh list)."""
        return list(self._drafts)

    def by_operator(self, operator: str) -> list[Draft]:
        """Return drafts posted by ``operator`` (case-insensitive), in order."""
        key = operator.strip().lower()
        return [d for d in self._drafts if d.operator.strip().lower() == key]

    def by_round(self, round: int) -> list[Draft]:
        """Return the drafts posted in ``round``, in insertion order."""
        return [d for d in self._drafts if d.round == round]

    def latest_round(self) -> int:
        """The highest round posted so far, or ``-1`` when the board is empty."""
        return max((d.round for d in self._drafts), default=-1)

    def __len__(self) -> int:
        """The number of drafts posted."""
        return len(self._drafts)
