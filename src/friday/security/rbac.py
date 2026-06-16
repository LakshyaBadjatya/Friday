# © Lakshya Badjatya — Author
"""Role-scoped access control: graduate single-owner into real, default-deny RBAC.

Family sharing today is a flat scope; RBAC turns it into named roles with explicit
permission sets and per-user assignments. The policy is **default-deny**: a user
with no role, an unknown role, or a permission not granted to their role is
refused. A role may hold the wildcard permission ``"*"`` to grant everything
(e.g. the owner).

Pure and deterministic — roles and assignments are injected, it reads no
configuration, holds no secrets, and performs no I/O — so the whole access matrix
is trivially unit-testable and safe to consult on the hot path.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

#: The permission that grants every action (held by privileged roles, e.g. owner).
WILDCARD = "*"


class Role(BaseModel):
    """A named role and the set of permissions it grants.

    ``permissions`` is a set of permission strings; the special :data:`WILDCARD`
    (``"*"``) grants every permission.
    """

    name: str
    permissions: frozenset[str]


class AccessPolicy:
    """A default-deny role/permission matrix with per-user role assignments.

    Args:
        roles: The known roles.
        assignments: Optional initial ``user -> role name`` map; an assignment to
            an unknown role is rejected.
    """

    def __init__(
        self, roles: Iterable[Role], assignments: dict[str, str] | None = None
    ) -> None:
        self._roles: dict[str, Role] = {r.name: r for r in roles}
        self._assignments: dict[str, str] = {}
        for user, role_name in (assignments or {}).items():
            self.assign(user, role_name)

    def assign(self, user: str, role_name: str) -> None:
        """Assign ``user`` to ``role_name``; raise ``ValueError`` if role unknown."""
        if role_name not in self._roles:
            raise ValueError(f"unknown role {role_name!r}")
        self._assignments[user] = role_name

    def role_of(self, user: str) -> str | None:
        """The role assigned to ``user``, or ``None`` if unassigned."""
        return self._assignments.get(user)

    def can(self, user: str, permission: str) -> bool:
        """Whether ``user`` is granted ``permission`` (default-deny).

        Denies when the user has no role, the role is unknown, or the permission
        is neither explicitly granted nor covered by the role's :data:`WILDCARD`.
        """
        role_name = self._assignments.get(user)
        if role_name is None:
            return False
        role = self._roles.get(role_name)
        if role is None:
            return False
        return WILDCARD in role.permissions or permission in role.permissions
