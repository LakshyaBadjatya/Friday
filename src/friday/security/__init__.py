# © Lakshya Badjatya — Author
"""Cross-cutting security-spine policy modules (pure, broker-adjacent).

These are the *policy* halves of Wave-2 security features — small, pure,
dependency-light decision modules the broker (and other call sites) consult
before an action. They import no LLM SDK and read no configuration: their policy
inputs (allow-lists, placeholders, rotation windows) are injected by ``app.py``.

* :class:`~friday.security.egress.EgressPolicy` — a fail-closed outbound-host
  allow-list ("nothing phones home unless it's on the list").
* :class:`~friday.security.pii.PIIRedactor` — scrub PII from text before it
  reaches a real provider or an external sink.
"""

from __future__ import annotations

from friday.security.anchor import AuditAnchor, make_anchor, verify_anchor
from friday.security.approvals import ApprovalRequest, ApprovalStatus, ApprovalStore
from friday.security.egress import EgressDecision, EgressPolicy
from friday.security.pii import PIIRedactor, RedactionResult
from friday.security.rbac import WILDCARD, AccessPolicy, Role
from friday.security.rotation import RotationPolicy, RotationStatus, SecretAge

__all__ = [
    "WILDCARD",
    "AccessPolicy",
    "ApprovalRequest",
    "ApprovalStatus",
    "ApprovalStore",
    "AuditAnchor",
    "EgressDecision",
    "EgressPolicy",
    "PIIRedactor",
    "RedactionResult",
    "Role",
    "RotationPolicy",
    "RotationStatus",
    "SecretAge",
    "make_anchor",
    "verify_anchor",
]
