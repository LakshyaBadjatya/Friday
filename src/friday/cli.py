"""The ``friday`` command-line interface.

A thin, dependency-light operator front door over the runtime FRIDAY already
exposes elsewhere. Every subcommand is a tiny adapter onto an existing,
well-tested module — the CLI owns *no* business logic, only argument parsing,
process exit codes, and stdout:

* ``serve`` — run the ASGI app via ``uvicorn`` on ``--host``/``--port``
  (default ``127.0.0.1:8000``). ``uvicorn`` is imported **lazily** inside the
  handler so merely importing this module (or building the parser in a test)
  never pulls the server in, and the app is passed by its factory path string
  (``friday.app:create_app`` with ``factory=True``) so the worker imports it.
* ``secrets set <name> <value>`` / ``secrets get <name>`` — write/read a secret
  through the backend selected by ``get_settings().secret_vault`` (the same
  :mod:`friday.secrets` backends the runtime wires); a missing secret exits
  non-zero.
* ``audit verify`` — open the hash-chained ledger at
  ``get_settings().audit_ledger_path`` and run :meth:`HashChainedAudit.verify`,
  printing ``ok`` / ``broken at <index>`` and exiting non-zero on tamper so a CI
  check can gate on it.
* ``roster`` — print the canonical persona code-names from :mod:`friday.roster`.
* ``version`` — print the installed package version.

:func:`build_parser` constructs the parser with **no** side effects so tests can
parse arguments in isolation; :func:`main` parses ``argv`` and dispatches to the
selected handler, returning the process exit code (it never calls
``sys.exit`` itself except via argparse's own usage errors).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from friday.config import Settings, get_settings
from friday.secrets import (
    EnvVault,
    FileVault,
    KeyringVault,
    MemoryVault,
    SecretVault,
)

__all__ = ["build_parser", "main"]

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000

# The factory path uvicorn imports for ``serve``. A string (not the function) so
# uvicorn's worker imports it in-process and this module never imports
# :mod:`friday.app` (which would pull the whole runtime graph in at CLI import).
_APP_FACTORY = "friday.app:create_app"


def _build_secret_vault(settings: Settings) -> SecretVault:
    """Construct the secret backend selected by ``settings.secret_vault``.

    Mirrors the runtime selection (``env`` reads ``FRIDAY_<NAME>`` from the
    process environment; ``memory`` is in-process only; ``file`` is a ``0600``
    JSON dev fallback alongside ``data/``; ``keyring`` wraps the OPTIONAL OS
    keychain — lazy-imported only when selected). Constructed here from the same
    :mod:`friday.secrets` backends so the CLI stays decoupled from
    :mod:`friday.app`.
    """
    from pathlib import Path  # noqa: PLC0415 — local: keep module import light

    backend = settings.secret_vault
    if backend == "memory":
        return MemoryVault()
    if backend == "file":
        db_path = settings.memory_db_path
        data_dir = Path(db_path).parent if db_path != ":memory:" else Path("data")
        return FileVault(str(data_dir / "secrets.json"))
    if backend == "keyring":
        return KeyringVault()
    return EnvVault()


def _handle_serve(args: argparse.Namespace) -> int:
    """Run the ASGI app via uvicorn (imported lazily) on ``--host``/``--port``."""
    import uvicorn  # noqa: PLC0415 — lazy so importing the CLI never needs uvicorn

    uvicorn.run(_APP_FACTORY, factory=True, host=args.host, port=args.port)
    return 0


def _handle_secrets(args: argparse.Namespace) -> int:
    """Set or get a secret through the configured vault backend."""
    vault = _build_secret_vault(get_settings())
    if args.secrets_command == "set":
        vault.set(args.name, args.value)
        print(f"secret {args.name!r} set")
        return 0
    # ``get``
    value = vault.get(args.name)
    if value is None:
        print(f"secret {args.name!r} is not set in the configured vault")
        return 1
    print(value)
    return 0


def _handle_audit(args: argparse.Namespace) -> int:
    """Verify the hash-chained audit ledger; exit non-zero on tamper."""
    from friday.broker import HashChainedAudit  # noqa: PLC0415 — local keeps import light

    ledger = HashChainedAudit(get_settings().audit_ledger_path)
    ok, broken_at = ledger.verify()
    if ok:
        print("ok")
        return 0
    print(f"broken at {broken_at}")
    return 1


def _handle_roster(args: argparse.Namespace) -> int:
    """Print the canonical persona code-names, one per line."""
    from friday.roster import ROSTER  # noqa: PLC0415 — local keeps import light

    for name in ROSTER.names():
        print(name)
    return 0


def _handle_version(args: argparse.Namespace) -> int:
    """Print the installed package version."""
    from importlib.metadata import version  # noqa: PLC0415 — local keeps import light

    print(version("friday"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``friday`` argument parser (no side effects).

    Each subcommand binds its handler via ``set_defaults(func=...)`` so
    :func:`main` can dispatch uniformly. Parsing alone (the path tests exercise)
    imports nothing beyond this module's top-level deps and never starts a
    server or touches a vault/ledger.
    """
    parser = argparse.ArgumentParser(
        prog="friday",
        description="FRIDAY operator CLI: serve the app, manage secrets, "
        "verify the audit ledger, and inspect the roster.",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="run the FRIDAY ASGI app via uvicorn")
    serve.add_argument(
        "--host", default=_DEFAULT_HOST, help=f"bind host (default {_DEFAULT_HOST})"
    )
    serve.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_PORT,
        help=f"bind port (default {_DEFAULT_PORT})",
    )
    serve.set_defaults(func=_handle_serve)

    secrets = subparsers.add_parser(
        "secrets", help="read/write secrets via the configured vault"
    )
    secrets_sub = secrets.add_subparsers(dest="secrets_command")
    secrets_set = secrets_sub.add_parser("set", help="store a secret value")
    secrets_set.add_argument("name", help="the logical secret name")
    secrets_set.add_argument("value", help="the secret value to store")
    secrets_set.set_defaults(func=_handle_secrets)
    secrets_get = secrets_sub.add_parser("get", help="print a stored secret value")
    secrets_get.add_argument("name", help="the logical secret name")
    secrets_get.set_defaults(func=_handle_secrets)

    audit = subparsers.add_parser("audit", help="audit-ledger operations")
    audit_sub = audit.add_subparsers(dest="audit_command")
    audit_verify = audit_sub.add_parser(
        "verify", help="verify the tamper-evident audit ledger"
    )
    audit_verify.set_defaults(func=_handle_audit)

    roster = subparsers.add_parser("roster", help="print the persona roster names")
    roster.set_defaults(func=_handle_roster)

    version = subparsers.add_parser("version", help="print the package version")
    version.set_defaults(func=_handle_version)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand.

    Returns the process exit code: ``0`` on success, non-zero on a failed
    verification / missing secret, and ``2`` (with usage) when no subcommand —
    or no second-level subcommand for ``secrets``/``audit`` — is given.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    result: int = func(args)
    return result


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
