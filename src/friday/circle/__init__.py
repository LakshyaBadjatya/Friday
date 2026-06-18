"""The friends-circle platform package.

Home for the multi-user "circle" features (groups, members, presence/status,
care, and the long-distance helpers) that sit on top of FRIDAY's core. Kept
generic — no personal data lives here; identity comes from the auth layer and
group membership at runtime.
"""

from __future__ import annotations
