# © Lakshya Badjatya — Author
"""Approval workflow: hold an irreversible action pending an explicit yes/no.

The broker already confirm-gates side-effecting tools in-band. The *approval*
workflow generalizes that to an out-of-band yes/no: a risky action raises an
:class:`ApprovalRequest` that stays ``pending`` until the owner approves or
denies it (in practice via a phone push — that delivery is wired separately),
and the action only proceeds once :meth:`ApprovalStore.is_approved` is true.

This module is the **pure state machine** behind that: create a request, approve
or deny it exactly once, and let it lazily **expire** after an optional TTL.
Time is injected (``now`` is passed in, never read from the clock) so the whole
lifecycle is deterministic and unit-testable; it imports no LLM SDK, performs no
I/O, and sends nothing — delivery and persistence are the caller's concern.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

#: The lifecycle states of an approval request.
ApprovalStatus = Literal["pending", "approved", "denied", "expired"]


class ApprovalRequest(BaseModel):
    """One request for the owner to approve or deny an action.

    Attributes:
        id: A caller-supplied unique id (kept out of the module so the module
            stays deterministic — no random/clock id generation here).
        action: A human-readable description of what will happen if approved.
        requested_by: The operator that raised the request.
        created_ts: The injected timestamp at creation.
        ttl_seconds: Optional time-to-live; past it a still-``pending`` request is
            treated as ``expired``. ``None`` means it never expires.
        status: The current lifecycle state.
        decided_ts: When it was approved/denied, else ``None``.
    """

    id: str
    action: str
    requested_by: str = "FRIDAY"
    created_ts: float
    ttl_seconds: float | None = None
    status: ApprovalStatus = "pending"
    decided_ts: float | None = None

    def is_expired(self, now: float) -> bool:
        """Whether a TTL is set and ``now`` is at or past the expiry instant."""
        return self.ttl_seconds is not None and now >= self.created_ts + self.ttl_seconds


class ApprovalStore:
    """An in-memory registry of approval requests with a one-shot decision rule.

    A request can be decided (approved or denied) exactly once and only while
    genuinely pending; expiry is applied lazily on decision and on read, so a
    request that outlived its TTL can never be approved.
    """

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}

    def create(
        self,
        action: str,
        *,
        request_id: str,
        now: float,
        requested_by: str = "FRIDAY",
        ttl_seconds: float | None = None,
    ) -> ApprovalRequest:
        """Register a new pending request; raise ``ValueError`` on a duplicate id."""
        if request_id in self._requests:
            raise ValueError(f"approval id {request_id!r} already exists")
        request = ApprovalRequest(
            id=request_id,
            action=action,
            requested_by=requested_by,
            created_ts=now,
            ttl_seconds=ttl_seconds,
        )
        self._requests[request_id] = request
        return request

    def get(self, request_id: str) -> ApprovalRequest | None:
        """Return the request, or ``None`` if unknown."""
        return self._requests.get(request_id)

    def approve(self, request_id: str, *, now: float) -> ApprovalRequest:
        """Approve a pending request (see :meth:`_decide`)."""
        return self._decide(request_id, "approved", now=now)

    def deny(self, request_id: str, *, now: float) -> ApprovalRequest:
        """Deny a pending request (see :meth:`_decide`)."""
        return self._decide(request_id, "denied", now=now)

    def _decide(
        self, request_id: str, outcome: ApprovalStatus, *, now: float
    ) -> ApprovalRequest:
        """Transition a pending, unexpired request to ``outcome`` exactly once.

        Raises ``KeyError`` for an unknown id, and ``ValueError`` if the request
        is already decided or has expired (an expired pending request is marked
        ``expired`` first, so it can never be approved after its TTL).
        """
        request = self._requests.get(request_id)
        if request is None:
            raise KeyError(request_id)
        if request.status != "pending":
            raise ValueError(f"approval {request_id!r} is already {request.status}")
        if request.is_expired(now):
            request.status = "expired"
            raise ValueError(f"approval {request_id!r} has expired")
        request.status = outcome
        request.decided_ts = now
        return request

    def is_approved(self, request_id: str, *, now: float) -> bool:
        """Whether the request is currently approved (expiring it lazily if due)."""
        request = self._requests.get(request_id)
        if request is None:
            return False
        if request.status == "pending" and request.is_expired(now):
            request.status = "expired"
        return request.status == "approved"

    def pending(self, *, now: float) -> list[ApprovalRequest]:
        """Every request still genuinely pending at ``now`` (expiring due ones)."""
        result: list[ApprovalRequest] = []
        for request in self._requests.values():
            if request.status == "pending" and request.is_expired(now):
                request.status = "expired"
            if request.status == "pending":
                result.append(request)
        return result
