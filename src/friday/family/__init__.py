"""Consent-enforced family sharing (build-spec §18) — off by default.

This package owns FRIDAY's family-sharing feature, gated behind
``FRIDAY_ENABLE_FAMILY_SHARING`` so the offline default exposes no ``/family``
surface (every route -> 404) and keeps a single-owner scope.

The NON-NEGOTIABLE consent guardrails of build-spec §18 live IN CODE in the
:class:`~friday.family.service.FamilyService`:

* A participant can only be added by THEMSELVES (self-opt-in required).
* A unilateral revoke stops sharing INSTANTLY.
* Every view is recorded and the viewed participant can see who viewed them.
* The default share is the coarse geofence STATUS (home/work/away), never raw
  coordinates; raw location requires an explicit per-viewer opt-in.

The public surface is the typed :class:`~friday.family.store.Participant` /
:class:`~friday.family.store.ViewRecord` models, the
:class:`~friday.family.store.SQLiteFamilyStore` adapter, the
:class:`~friday.family.service.FamilyService` facade with its
:class:`~friday.family.service.FamilyShareError`, and the flagged
:data:`router`. The integration agent wires this slice by including
:data:`friday.family.router`.
"""

from __future__ import annotations

from friday.api.routes_family import router
from friday.family.service import FamilyService, FamilyShareError
from friday.family.store import Participant, SQLiteFamilyStore, ViewRecord

__all__ = [
    "FamilyService",
    "FamilyShareError",
    "Participant",
    "SQLiteFamilyStore",
    "ViewRecord",
    "router",
]
