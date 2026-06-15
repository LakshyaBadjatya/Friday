"""Boot test (Task 0.8, satisfiable once ``app.py`` lands in Task 1.9).

Asserts the FastAPI factory imports and constructs an app with no network I/O.
This is the Phase-0 deferred boot test, now green.
"""

from __future__ import annotations


def test_app_imports() -> None:
    from friday.app import create_app

    app = create_app()
    assert app is not None


def test_app_has_chat_route() -> None:
    from friday.app import create_app

    app = create_app()
    # The OpenAPI schema is the stable, version-agnostic view of registered
    # paths (route object internals vary across FastAPI/Starlette versions).
    assert "/chat" in app.openapi()["paths"]


def test_orchestrator_wired_on_state() -> None:
    from friday.app import create_app
    from friday.core.orchestrator import Orchestrator

    app = create_app()
    assert isinstance(app.state.orchestrator, Orchestrator)
