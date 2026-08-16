"""GitHub, for the operators who need it.

This one does not go through the relay. GitHub is a hosted API reachable from
the container, so asking a laptop to proxy calls to it would add a dependency on
that laptop being awake for no benefit at all — "did my build pass" should work
from a phone at midnight.

Read-only by construction. There is no code here that merges, closes, comments,
pushes or deletes; the token can be a fine-grained one with read scopes and
nothing here will notice. A model that can close issues on the strength of a
chat message is a bad trade for the convenience of not having to click.

Two operators use it. FORGE answers "did my build pass" and "what's failing",
which is the workflow-run view. EDITH answers "is my Android app safe", which is
the Dependabot alert view plus whether scanning is even switched on — the second
being the more common real answer, because an alert list that is empty because
nobody enabled the scanner reads exactly like an alert list that is empty
because the code is clean.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import anyio

from friday.logging import get_logger

logger = get_logger("friday.link.github")

_API = "https://api.github.com"
_ACCEPT = "application/vnd.github+json"


def _token(settings: Any) -> str:
    secret = getattr(settings, "github_token", None)
    if secret is None:
        return ""
    getter = getattr(secret, "get_secret_value", None)
    return str(getter() if callable(getter) else secret or "")


async def _get(token: str, path: str) -> tuple[bool, Any]:
    """One read from the GitHub API, off the event loop."""
    request = urllib.request.Request(  # noqa: S310
        f"{_API}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": _ACCEPT,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "FRIDAY (https://friday.sukhma.in)",
        },
    )

    def _fetch() -> tuple[bool, Any]:
        try:
            with urllib.request.urlopen(request, timeout=15) as resp:  # noqa: S310
                return True, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            # The status matters to the caller: 403 on an alerts endpoint means
            # the feature is off, not that the repository is clean.
            return False, {"status": exc.code}
        except Exception:  # noqa: BLE001 - GitHub being down is not a crash
            logger.warning("github: request failed for %s", path)
            return False, {"status": 0}

    return await anyio.to_thread.run_sync(_fetch)


async def repositories(settings: Any, limit: int = 10) -> list[dict[str, Any]]:
    """The owner's repositories, most recently pushed first."""
    token = _token(settings)
    if not token:
        return []
    query = urllib.parse.urlencode({"sort": "pushed", "per_page": limit})
    ok, data = await _get(token, f"/user/repos?{query}")
    if not ok or not isinstance(data, list):
        return []
    return [
        {
            "name": repo.get("full_name", ""),
            "private": bool(repo.get("private")),
            "pushed_at": repo.get("pushed_at", ""),
            "language": repo.get("language") or "",
        }
        for repo in data
    ]


async def build_status(settings: Any, repo: str, limit: int = 5) -> dict[str, Any]:
    """The most recent workflow runs for a repository."""
    token = _token(settings)
    if not token:
        return {"ok": False, "error": "no GitHub token is configured"}
    query = urllib.parse.urlencode({"per_page": limit})
    ok, data = await _get(token, f"/repos/{repo}/actions/runs?{query}")
    if not ok:
        status = (data or {}).get("status")
        if status == 404:
            return {
                "ok": False,
                "error": f"can't see {repo} — check the name, or the token's scope",
            }
        return {"ok": False, "error": f"GitHub said {status or 'nothing useful'}"}
    runs = (data or {}).get("workflow_runs") or []
    return {
        "ok": True,
        "repo": repo,
        "runs": [
            {
                "name": run.get("name", ""),
                "status": run.get("status", ""),
                "conclusion": run.get("conclusion") or "running",
                "branch": run.get("head_branch", ""),
                "when": run.get("updated_at", ""),
                "message": (run.get("head_commit") or {}).get("message", "")[:90],
            }
            for run in runs
        ],
    }


async def security_alerts(settings: Any, repo: str) -> dict[str, Any]:
    """Dependabot alerts, and whether anything is actually watching.

    A 403 here is the interesting case and gets its own answer. It means alerts
    are not enabled for the repository, which looks identical to "no alerts" if
    you only count the list — and telling somebody their app is clean when
    nothing has ever checked is worse than saying nothing at all.
    """
    token = _token(settings)
    if not token:
        return {"ok": False, "error": "no GitHub token is configured"}
    ok, data = await _get(
        token, f"/repos/{repo}/dependabot/alerts?state=open&per_page=20"
    )
    if not ok:
        status = (data or {}).get("status")
        if status == 403:
            return {
                "ok": True, "repo": repo, "scanning_enabled": False, "alerts": [],
                "note": "Dependabot alerts are switched off for this repository, so "
                        "nothing has been checked. Settings → Code security.",
            }
        if status == 404:
            return {
                "ok": False,
                "error": f"can't see {repo} — check the name, or the token's scope",
            }
        return {"ok": False, "error": f"GitHub said {status or 'nothing useful'}"}
    alerts = data if isinstance(data, list) else []
    return {
        "ok": True,
        "repo": repo,
        "scanning_enabled": True,
        "alerts": [
            {
                "severity": ((alert.get("security_advisory") or {})
                             .get("severity", "unknown")),
                "package": ((alert.get("dependency") or {}).get("package") or {})
                            .get("name", ""),
                "summary": ((alert.get("security_advisory") or {})
                            .get("summary", ""))[:120],
            }
            for alert in alerts
        ],
    }
