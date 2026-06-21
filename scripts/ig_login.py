#!/usr/bin/env python3
"""One-time Instagram login → prints the session JSON for the server to reuse.

Run this ONCE on your own machine (a residential IP Instagram trusts), solving any
first-time challenge interactively. Copy the printed JSON into Render's
``FRIDAY_INSTAGRAM_SESSION_JSON`` secret so the server reuses the trusted session
instead of a fresh datacenter login.

    FRIDAY_INSTAGRAM_USERNAME=you FRIDAY_INSTAGRAM_PASSWORD=secret python scripts/ig_login.py
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    username = os.environ.get("FRIDAY_INSTAGRAM_USERNAME", "").strip()
    password = os.environ.get("FRIDAY_INSTAGRAM_PASSWORD", "").strip()
    # Optional 2FA verification code (authenticator app or SMS) for accounts with
    # two-factor enabled. The code is time-sensitive, so run this promptly.
    code = os.environ.get("FRIDAY_INSTAGRAM_2FA_CODE", "").strip()
    if not username or not password:
        print(
            "Set FRIDAY_INSTAGRAM_USERNAME and FRIDAY_INSTAGRAM_PASSWORD first.",
            file=sys.stderr,
        )
        return 2
    try:
        from instagrapi import Client
    except ImportError:
        print("instagrapi is not installed — run: pip install instagrapi", file=sys.stderr)
        return 3

    cl = Client()
    if code:
        cl.login(username, password, verification_code=code)
    else:
        cl.login(username, password)
    print(json.dumps(cl.get_settings()))
    print(
        "\nPaste the JSON line above into FRIDAY_INSTAGRAM_SESSION_JSON.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
