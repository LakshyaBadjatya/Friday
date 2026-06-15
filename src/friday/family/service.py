"""Consent-enforcing family-sharing service (build-spec §18).

:class:`FamilyService` wraps a :class:`~friday.family.store.SQLiteFamilyStore`
and is the single place the NON-NEGOTIABLE consent guardrails of build-spec §18
live IN CODE — not in a doc, not in the UI, but enforced on every call:

1. **Self-opt-in only.** A participant can ONLY be added by themselves: every
   :meth:`opt_in` requires ``self_opt_in=True``. An attempt to add someone from
   another account (``self_opt_in=False``) is rejected with
   :class:`FamilyShareError` and persists nothing.
2. **Sharing is the owner's choice.** Only an opted-in participant may share, and
   only that participant may revoke their own edges.
3. **Unilateral revoke stops sharing INSTANTLY.** :meth:`revoke` deletes the
   share edge, so the very next :meth:`view` by the revoked viewer is denied.
4. **Every view is recorded and visible to the viewed.** :meth:`view` appends an
   audited record; :meth:`views_of` lets the VIEWED participant see who viewed
   them.

**Default granularity is the geofence STATUS** (``home`` / ``work`` / ``away``),
never raw coordinates. Raw latitude/longitude is only ever returned when the
owner has granted that specific viewer an explicit per-viewer ``raw_location``
opt-in (recorded on the share edge); otherwise the view is status-only.
"""

from __future__ import annotations

from typing import Any

from friday.errors import FridayError
from friday.family.store import Participant, SQLiteFamilyStore, ViewRecord


class FamilyShareError(FridayError):
    """A consent guardrail was violated (not-opted-in / not-shared / non-self).

    Raised by :class:`FamilyService` when an operation would breach build-spec
    §18 consent rules — e.g. adding someone from another account, sharing before
    opting in, or viewing a participant who has not shared with the viewer.
    """


#: A fixed, coarse coordinate pair returned ONLY when an explicit per-viewer raw
#: grant exists. The real coordinate source is out of scope for this slice; the
#: privacy contract (status by default, raw only on explicit per-viewer opt-in)
#: is what is enforced here, so a deterministic placeholder keeps the path
#: offline and testable without inventing a location backend.
_RAW_PLACEHOLDER: dict[str, float] = {"latitude": 0.0, "longitude": 0.0}


class FamilyService:
    """Consent-enforcing facade over the family-sharing store."""

    def __init__(self, store: SQLiteFamilyStore) -> None:
        self.store = store

    # -- opt-in (guardrail 1) ---------------------------------------------- #
    def opt_in(self, name: str, *, self_opt_in: bool) -> Participant:
        """Opt ``name`` in to family sharing; ``self_opt_in`` MUST be ``True``.

        Guardrail 1: a participant can only be added by THEMSELVES. A call with
        ``self_opt_in=False`` (i.e. an attempt to add someone from another
        account) is rejected and persists nothing.

        Raises:
            FamilyShareError: when ``self_opt_in`` is ``False``.
        """
        if not self_opt_in:
            raise FamilyShareError(
                f"cannot add {name!r}: a participant can only be added by "
                "themselves (self_opt_in must be True)"
            )
        return self.store.opt_in(name)

    # -- sharing (guardrail 2) --------------------------------------------- #
    def share(
        self, owner: str, viewer: str, *, raw_location: bool = False
    ) -> Participant:
        """Share ``owner``'s location with ``viewer``; only if ``owner`` opted in.

        The default share is the coarse geofence status; ``raw_location=True``
        records an explicit per-viewer grant to RAW coordinates for this viewer
        only. Returns the updated owner participant (with the new edge in
        ``sharing_with``).

        Raises:
            FamilyShareError: when ``owner`` has not opted in.
        """
        if self.store.get(owner) is None:
            raise FamilyShareError(
                f"{owner!r} has not opted in; cannot share before opting in"
            )
        self.store.add_share(owner, viewer, raw_location=raw_location)
        updated = self.store.get(owner)
        assert updated is not None  # owner exists (checked above)
        return updated

    # -- revoke (guardrail 3) ---------------------------------------------- #
    def revoke(self, owner: str, viewer: str) -> Participant:
        """Unilaterally revoke ``owner``'s share with ``viewer`` — stops INSTANTLY.

        Guardrail 3: removing the edge takes effect immediately, so the next
        :meth:`view` by ``viewer`` is denied. Revoking an edge that does not
        exist is a harmless no-op. Returns the updated owner participant.

        Raises:
            FamilyShareError: when ``owner`` has not opted in.
        """
        if self.store.get(owner) is None:
            raise FamilyShareError(
                f"{owner!r} has not opted in; nothing to revoke"
            )
        self.store.remove_share(owner, viewer)
        updated = self.store.get(owner)
        assert updated is not None  # owner exists (checked above)
        return updated

    # -- viewing (guardrails 3 + 4; default geofence status) --------------- #
    def view(self, owner: str, *, viewer: str) -> dict[str, Any]:
        """Return ``owner``'s shared location for ``viewer`` and RECORD the view.

        A view is only permitted when ``owner`` currently shares with ``viewer``
        (so a revoke denies it instantly — guardrail 3). The returned payload is
        the coarse geofence ``status`` by default (``precision="status"``); raw
        ``latitude``/``longitude`` are included ONLY when ``owner`` has granted
        this viewer an explicit per-viewer raw opt-in (``precision="raw"``). The
        view is recorded so the viewed participant can see who viewed them
        (guardrail 4).

        Raises:
            FamilyShareError: when ``owner`` is unknown or has not shared with
                ``viewer`` (no view is recorded for a denied attempt).
        """
        participant = self.store.get(owner)
        if participant is None:
            raise FamilyShareError(f"{owner!r} is not a known participant")
        if not self.store.is_sharing(owner, viewer):
            raise FamilyShareError(
                f"{owner!r} is not sharing with {viewer!r}"
            )

        # Record the view BEFORE returning (auditable even for the raw path).
        self.store.record_view(viewer, owner)

        payload: dict[str, Any] = {
            "name": participant.name,
            "viewer": viewer,
            "status": participant.status,
            "precision": "status",
        }
        # Raw coordinates ONLY on an explicit per-viewer grant.
        if self.store.shares_raw(owner, viewer):
            payload["precision"] = "raw"
            payload.update(_RAW_PLACEHOLDER)
        return payload

    # -- audit read (guardrail 4) ------------------------------------------ #
    def views_of(self, name: str) -> list[dict[str, Any]]:
        """Return the recorded views of ``name`` (most-recent first), as dicts.

        Guardrail 4: this is what lets the VIEWED participant see who viewed them.
        """
        records: list[ViewRecord] = self.store.views_of(name)
        return [record.model_dump() for record in records]
