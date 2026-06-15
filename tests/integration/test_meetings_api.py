"""Integration tests for the ``/meetings`` capture API (Tier 1).

All offline against :class:`~friday.providers.stt.FakeSTT` + a scripted
:class:`~friday.providers.llm.FakeLLM` and a deterministic
:class:`~friday.providers.embeddings.FakeEmbeddings`, with a ``TestClient`` whose
``FRIDAY_ENABLE_MEETINGS`` flag is forced on/off via a monkeypatched
``get_settings`` (mirroring the ``/rag`` API tests). No network, no key, no audio.

Because the wired-in LLM is the empty-script :class:`FakeLLM`, every capture's LLM
summary pass fails (no scripted response) and degrades to the NON-FATAL
transcript-only fallback — which is exactly the offline contract: capture must
still succeed and store complete notes even when the LLM cannot summarize.

Covered:
* Every ``/meetings`` route is ``404`` when the flag is off.
* ``POST /meetings/capture`` with a JSON ``{title, audio_b64}`` body captures,
  stores, and returns notes (transcript = FakeSTT output); ``GET /meetings`` and
  ``GET /meetings/{id}`` read them back; ``DELETE`` removes them.
* ``POST /meetings/capture`` as a multipart upload derives the title from the
  filename and captures.
* With RAG enabled too, the captured transcript is ingested under
  ``meeting:<title>`` (the shared vector store gains a matching chunk).
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import Settings


def _settings(*, enabled: bool, enable_rag: bool = False) -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated so
    # tests never touch the developer's real ``data/`` files or each other.
    return Settings(
        _env_file=None,
        enable_meetings=enabled,
        enable_rag=enable_rag,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool, enable_rag: bool = False
) -> TestClient:
    """A ``TestClient`` whose meetings flag is forced via a patched ``get_settings``."""
    import friday.app as app_module

    def _factory() -> Settings:
        return _settings(enabled=enabled, enable_rag=enable_rag)

    monkeypatch.setattr(app_module, "get_settings", _factory)
    app = create_app()
    return TestClient(app)


_AUDIO_B64 = base64.b64encode(b"fake-audio-bytes").decode("ascii")


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_capture_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _settings(enabled=False)
        resp = client.post(
            "/meetings/capture", json={"title": "x", "audio_b64": _AUDIO_B64}
        )
    assert resp.status_code == 404


def test_list_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _settings(enabled=False)
        resp = client.get("/meetings")
    assert resp.status_code == 404


def test_get_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _settings(enabled=False)
        resp = client.get("/meetings/1")
    assert resp.status_code == 404


def test_delete_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _settings(enabled=False)
        resp = client.delete("/meetings/1")
    assert resp.status_code == 404


def test_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), capture is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _settings(enabled=False)
        resp = client.post(
            "/meetings/capture", json={"title": "x", "audio_b64": _AUDIO_B64}
        )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> JSON capture, read back, delete
# --------------------------------------------------------------------------- #
def test_capture_json_then_read_and_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _settings(enabled=True)
        resp = client.post(
            "/meetings/capture",
            json={"title": "Weekly Sync", "audio_b64": _AUDIO_B64},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["id"], int)
        assert body["title"] == "Weekly Sync"
        # Transcript is the FakeSTT output.
        assert body["transcript"] == "fake transcript"
        # LLM is the empty-script FakeLLM -> non-fatal fallback: a summary exists,
        # no action items, and capture did not raise.
        assert body["summary"]
        assert body["action_items"] == []
        meeting_id = body["id"]

        listed = client.get("/meetings")
        assert listed.status_code == 200
        listing = listed.json()
        assert listing["count"] == 1
        assert listing["meetings"][0]["id"] == meeting_id

        fetched = client.get(f"/meetings/{meeting_id}")
        assert fetched.status_code == 200
        assert fetched.json()["title"] == "Weekly Sync"

        deleted = client.delete(f"/meetings/{meeting_id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"id": meeting_id, "removed": 1}

        gone = client.get(f"/meetings/{meeting_id}")
        assert gone.status_code == 404


def test_capture_bad_base64_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _settings(enabled=True)
        resp = client.post(
            "/meetings/capture",
            json={"title": "Bad", "audio_b64": "!!!not-base64!!!"},
        )
    assert resp.status_code == 422


def test_capture_missing_field_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _settings(enabled=True)
        resp = client.post("/meetings/capture", json={"title": "No audio"})
    assert resp.status_code == 422


def test_get_missing_meeting_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _settings(enabled=True)
        resp = client.get("/meetings/999")
    assert resp.status_code == 404


def test_delete_missing_meeting_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _settings(enabled=True)
        resp = client.delete("/meetings/999")
    assert resp.status_code == 200
    assert resp.json() == {"id": 999, "removed": 0}


# --------------------------------------------------------------------------- #
# Enabled -> multipart upload (title from filename)
# --------------------------------------------------------------------------- #
def test_capture_multipart_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _settings(enabled=True)
        resp = client.post(
            "/meetings/capture",
            files={"file": ("standup.wav", b"\x00\x01fake-audio", "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Title derived from the filename (extension stripped).
        assert body["title"] == "standup"
        assert body["transcript"] == "fake transcript"

        listed = client.get("/meetings")
        assert any(m["title"] == "standup" for m in listed.json()["meetings"])


# --------------------------------------------------------------------------- #
# Enabled + RAG -> the transcript is ingested for retrieval
# --------------------------------------------------------------------------- #
def test_capture_ingests_transcript_when_rag_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With RAG on, the captured transcript lands in the shared vector store.

    A tmp-file (not ``":memory:"``) DB path is used so the vector store's
    connection is opened per call and is safe to query directly from the test
    thread. Still fully offline (FakeSTT / FakeLLM / FakeEmbeddings).
    """
    db_path = str(tmp_path / "friday.db")

    def _factory() -> Settings:
        return Settings(
            _env_file=None,
            enable_meetings=True,
            enable_rag=True,
            llm_provider="fake",
            memory_db_path=db_path,
        )

    import friday.app as app_module

    monkeypatch.setattr(app_module, "get_settings", _factory)
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _factory()
        resp = client.post(
            "/meetings/capture",
            json={"title": "Board Meeting", "audio_b64": _AUDIO_B64},
        )
        assert resp.status_code == 200

        # The shared vector store is the one the knowledge path retrieves from.
        vector = client.app.state.orchestrator._vector  # noqa: SLF001
        hits = vector.query("fake transcript", k=4)
        assert any(
            hit.source_id.startswith("meeting:Board Meeting#") for hit in hits
        )
