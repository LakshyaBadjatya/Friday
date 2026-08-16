"""Where the marks are going.

Every solved problem already carries a subject, a chapter, and an independent
SymPy verdict. Rolled up per chapter, that is a real answer to "what should I
revise" — built from what actually happened rather than from self-report.

Mastery is the verified fraction: solves whose arithmetic checked out, over
attempts. Weakest first, because that is the order to revise in.

**A load-bearing caveat about what "verified" means.** :mod:`friday.vault.solver`
only attempts SymPy verification on a bare single-variable equation; word
problems, prose, geometry, chemistry, and multi-variable systems all fall
through to ``Verification(ok=False)`` with no distinct "not verifiable" state
— see that module's docstring. That means a chapter of entirely correct
chemistry answers rolls up to ``mastery == 0.0`` and sorts as the weakest
chapter, indistinguishable from a chapter that is genuinely being gotten
wrong. This rollup reports the verified fraction exactly as specified — it
does not paper over that gap — but a caller surfacing it (dashboard, revision
queue) should not present ``mastery`` as "percent correct" without that
caveat, and a truer metric would need a third, typed verification state
("not attempted") distinct from "attempted and wrong," which does not exist
in :class:`~friday.vault.models.Verification` today.

**Locked items are excluded.** :meth:`~friday.vault.index.VaultIndex.list_items`
defaults to hiding ``Privacy.LOCKED`` items, and this rollup keeps that
default rather than overriding it. :mod:`friday.vault.search` sets the
precedent — its docstring treats "locked items never appear, full stop" as a
security guarantee, not just a model-provider guardrail — and the same
reasoning applies here: a mastery rollup over a locked chapter would still
leak that chapter's existence, its classification, and whether the student
got it right, through an aggregate rather than raw content, which is exactly
what marking something LOCKED is meant to prevent. The alternative — folding
locked items in because mastery stats are meant to reflect what actually
happened — is a real argument, but it is in tension with a privacy signal the
student set deliberately, and this module defers to that signal.

**Known scaling limits, flagged rather than fixed here** (the ``VaultIndex``
protocol has no pagination cursor or batch solve lookup, and changing it is
out of scope for this module):

* ``list_items(..., limit=_MAX_ITEMS)`` truncates silently past ``_MAX_ITEMS``
  items — a vault larger than that undercounts without any error.
* Every item with a ``solve_id`` costs one ``get_solve`` call — N+1 reads. On
  the Firestore backend that is N+1 network round trips per rollup, not one
  bulk read.
* :meth:`~friday.vault.index.VaultIndex.get_solve` can raise on the Firestore
  backend when the service is unreachable (see
  :class:`friday.vault.firestore_index.FirestoreIndexError`, whose docstring
  is explicit that the caller must treat that as a failure, not an empty
  result). This module does not catch it — an exception propagates out of
  :meth:`Analytics.rollup` rather than being swallowed into a quietly
  incomplete rollup. That matches this codebase's existing rule (see
  ``FirestoreIndexError``'s own docstring): it is fine to report short
  results only when the vault genuinely has fewer of them, never when a read
  failed partway through.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from friday.vault.index import VaultIndex

_MAX_ITEMS = 10_000


class ChapterMastery(BaseModel):
    """One chapter's record."""

    subject: str
    chapter: str
    attempts: int
    verified: int
    mastery: float
    last_seen: str


@dataclass
class _Bucket:
    """Mutable accumulator for one (subject, chapter) key while rolling up."""

    attempts: int = 0
    verified: int = 0
    last_seen: str = ""


class Analytics:
    """Mastery rollups over the solved items in the vault."""

    def __init__(self, index: VaultIndex) -> None:
        self._index = index

    def rollup(self, owner_uid: str) -> list[ChapterMastery]:
        """Per-chapter mastery for one owner, weakest first.

        Only items that carry a ``solve_id`` resolving to a stored
        :class:`~friday.vault.models.Solve` count as attempts. An item
        without a solve was never attempted and contributes nothing; an item
        whose ``solve_id`` points at a solve that is missing (deleted, or
        malformed and skipped by the index's own "log and skip" row-parsing
        rule) is also skipped rather than counted, since there is no
        verification verdict to attach to it — counting it as an
        indeterminate attempt would bias mastery without any real signal
        either way.

        Ties (equal ``mastery`` and equal ``attempts``) sort deterministically
        by subject then chapter, rather than depending on dict insertion
        order.
        """
        buckets: dict[tuple[str, str], _Bucket] = {}
        for item in self._index.list_items(owner_uid, include_locked=False, limit=_MAX_ITEMS):
            if not item.solve_id:
                continue
            solve = self._index.get_solve(item.solve_id)
            if solve is None:
                continue
            key = (
                item.classification.subject or "unclassified",
                item.classification.chapter or "unfiled",
            )
            bucket = buckets.setdefault(key, _Bucket())
            bucket.attempts += 1
            if solve.verification.ok:
                bucket.verified += 1
            if item.created_at > bucket.last_seen:
                bucket.last_seen = item.created_at

        rows = [
            ChapterMastery(
                subject=subject,
                chapter=chapter,
                attempts=bucket.attempts,
                verified=bucket.verified,
                # Safe: a bucket only exists once `setdefault` has been
                # followed by an increment, so `attempts` is never zero here.
                mastery=bucket.verified / bucket.attempts,
                last_seen=bucket.last_seen,
            )
            for (subject, chapter), bucket in buckets.items()
        ]
        rows.sort(key=lambda r: (r.mastery, -r.attempts, r.subject, r.chapter))
        return rows
