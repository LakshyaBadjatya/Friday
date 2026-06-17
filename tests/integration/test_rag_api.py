"""Integration tests for the ``/rag`` personal-RAG API (Tier 1, Stage 1A).

All offline against the deterministic :class:`~friday.providers.embeddings.FakeEmbeddings`
and a :class:`~friday.providers.llm.FakeLLM`, with a ``TestClient`` whose
``FRIDAY_ENABLE_RAG`` flag is forced on/off via a monkeypatched ``get_settings``
(mirroring the studio API tests). No network, no key.

Covered:
* Every ``/rag`` route is ``404`` when the flag is off (the feature does not
  exist for callers).
* ``POST /rag/ingest`` with a JSON ``{source_id, text}`` body ingests and returns
  ``{source_id, chunks}``; a subsequent ``GET /rag/sources`` lists the source.
* ``POST /rag/ingest`` as a multipart file upload derives the source id from the
  filename and ingests.
* ``DELETE /rag/sources/{source_id}`` forgets the document; a re-query of the
  shared vector store afterwards returns nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import Settings


def _enable_rag_settings() -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated so
    # tests never touch the developer's real ``data/`` files or each other.
    return Settings(
        _env_file=None,
        enable_rag=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_rag_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_rag=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose RAG flag is forced via a patched ``get_settings``."""
    import friday.app as app_module

    factory = _enable_rag_settings if enabled else _disable_rag_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_rag_disabled_ingest_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_rag_settings()
        resp = client.post(
            "/rag/ingest", json={"source_id": "n", "text": "hello"}
        )
    assert resp.status_code == 404


def test_rag_disabled_sources_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_rag_settings()
        resp = client.get("/rag/sources")
    assert resp.status_code == 404


def test_rag_disabled_delete_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_rag_settings()
        resp = client.delete("/rag/sources/n")
    assert resp.status_code == 404


def test_rag_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), ingest is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_rag_settings()
        resp = client.post("/rag/ingest", json={"source_id": "n", "text": "x"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> JSON ingest, list, multipart, delete
# --------------------------------------------------------------------------- #
def test_rag_ingest_json_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled JSON ingest returns ``{source_id, chunks}`` and lists the source."""
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_rag_settings()
        resp = client.post(
            "/rag/ingest",
            json={
                "source_id": "notes-friday",
                "text": (
                    "FRIDAY is a local-first assistant. The knowledge agent "
                    "grounds every answer strictly in retrieved sources."
                ),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_id"] == "notes-friday"
        assert body["chunks"] >= 1

        listed = client.get("/rag/sources")
        assert listed.status_code == 200
        assert "notes-friday" in listed.json()["sources"]


def test_rag_ingest_multipart_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multipart file upload derives the source id from the filename."""
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_rag_settings()
        resp = client.post(
            "/rag/ingest",
            files={
                "file": (
                    "meeting.md",
                    b"# Meeting notes\nWe agreed to ship the RAG slice on Friday.",
                    "text/markdown",
                )
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # Source id derived from the filename (extension stripped).
        assert body["source_id"] == "meeting"
        assert body["chunks"] >= 1

        listed = client.get("/rag/sources")
        assert "meeting" in listed.json()["sources"]


def test_rag_ingest_pdf_without_pypdf_is_415_not_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A .pdf upload when the optional ``pypdf`` is absent yields a clean 415, not 500."""
    import importlib.util

    if importlib.util.find_spec("pypdf") is not None:  # pragma: no cover
        pytest.skip("pypdf is installed; the optional-dependency 415 path is not exercised")
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_rag_settings()
        resp = client.post(
            "/rag/ingest",
            files={"file": ("report.pdf", b"%PDF-1.4 not a real pdf", "application/pdf")},
        )
        assert resp.status_code == 415
        assert "pypdf" in resp.json()["detail"].lower()


def test_rag_delete_then_requery_returns_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``DELETE /rag/sources/{id}`` forgets the doc; the vector re-query is empty.

    A tmp-file (not ``":memory:"``) vector store is used here so its connection
    is opened per call, making it safe to query directly from the test thread —
    a ``":memory:"`` store keeps one connection pinned to the route worker thread
    and cannot be touched from here. Still fully offline (FakeEmbeddings/FakeLLM).
    """
    db_path = str(tmp_path / "friday.db")
    secret = "The vault combination for project zephyr is four nine two."

    def _factory() -> Settings:
        return Settings(
            _env_file=None,
            enable_rag=True,
            llm_provider="fake",
            memory_db_path=db_path,
        )

    import friday.app as app_module

    monkeypatch.setattr(app_module, "get_settings", _factory)
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _factory()
        client.post(
            "/rag/ingest", json={"source_id": "notes-secret", "text": secret}
        )
        # The shared vector store is the one the orchestrator/knowledge path uses.
        # Query with the exact ingested text: under the deterministic
        # FakeEmbeddings identical text -> identical vector -> score 1.0, so the
        # chunk reliably surfaces regardless of the fake's (random) cross-text
        # similarity. This isolates the test to the delete behaviour itself.
        vector = client.app.state.orchestrator._vector  # noqa: SLF001
        before = vector.query(secret, k=4)
        assert any(hit.source_id.startswith("notes-secret#") for hit in before)

        deleted = client.delete("/rag/sources/notes-secret")
        assert deleted.status_code == 200

        after = vector.query(secret, k=4)
        assert not any(hit.source_id.startswith("notes-secret#") for hit in after)
        listed = client.get("/rag/sources")
        assert "notes-secret" not in listed.json()["sources"]
