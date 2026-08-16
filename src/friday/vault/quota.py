"""Two budgets: bytes stored, and model calls spent.

Storage is checked before a signature is ever issued, so an over-quota account
cannot start an upload it has no room for. The solve cap bounds the always-full
ensemble: three model calls per problem is the accepted cost, and this is what
stops a retry loop from turning that into three thousand.

Counts are per-process and per-day. That is deliberate: a personal surface where
losing the count on restart costs nothing worth a table.

Outage handling: :meth:`VaultIndex.total_bytes` can raise ``FirestoreIndexError``
when the backing store is unreachable, precisely so an outage is never mistaken
for an empty vault (see that class's docstring in
:mod:`friday.vault.firestore_index`). :meth:`QuotaGuard.usage` and
:meth:`QuotaGuard.may_upload` do not catch it: they let it propagate. Catching
it here and returning "no room" would look the same as a real fail-closed
refusal, but it would quietly convert a Firestore outage into an ordinary quota
response instead of the 5xx it should be -- exactly the distinction
``FirestoreIndexError``'s docstring assigns to the route layer, not to this
guard. So an outage surfaces as an exception, not as `False`.
"""

from __future__ import annotations

from pydantic import BaseModel

from friday.vault.index import VaultIndex

_BYTES_PER_GB = 1024**3
#: Fraction of quota past which the usage report raises a warning.
_WARN_AT = 0.8


class Usage(BaseModel):
    """A storage report for one owner."""

    used_bytes: int
    quota_bytes: int
    fraction: float
    warning: bool


class QuotaGuard:
    """Storage accounting and the daily solve cap."""

    def __init__(
        self, index: VaultIndex, *, quota_gb: float, daily_solve_cap: int
    ) -> None:
        self._index = index
        self._quota_bytes = int(quota_gb * _BYTES_PER_GB)
        self._cap = daily_solve_cap
        self._solves: dict[str, int] = {}

    def usage(self, owner_uid: str) -> Usage:
        """Bytes used, and whether the owner is past the warning threshold.

        Propagates ``FirestoreIndexError`` (or any other exception
        ``total_bytes`` raises) rather than catching it -- see the module
        docstring on why an outage must never be reported as usage.
        """
        used = self._index.total_bytes(owner_uid)
        fraction = used / self._quota_bytes if self._quota_bytes else 0.0
        return Usage(
            used_bytes=used,
            quota_bytes=self._quota_bytes,
            fraction=fraction,
            warning=fraction >= _WARN_AT,
        )

    def may_upload(self, owner_uid: str) -> bool:
        """Whether there is room for another upload.

        A ``quota_bytes`` of zero or less always refuses (there is never any
        room), which also covers a misconfigured non-positive ``quota_gb``
        without raising. An index outage propagates rather than being read as
        "plenty of room" -- see the module docstring.
        """
        return self.usage(owner_uid).used_bytes < self._quota_bytes

    def may_solve(self, day: str) -> bool:
        """Whether the ensemble may run again on ``day`` (``YYYY-MM-DD``).

        ``day`` is an opaque per-process bucket key: nothing here parses or
        validates it, so callers must pass a consistent, canonical string
        (e.g. always UTC ``YYYY-MM-DD``) for the cap to mean what it says.
        A non-positive ``daily_solve_cap`` means unlimited.
        """
        if self._cap <= 0:
            return True
        return self._solves.get(day, 0) < self._cap

    def record_solve(self, day: str) -> None:
        """Count one solve against ``day``'s budget."""
        self._solves[day] = self._solves.get(day, 0) + 1
