"""Integration tests for the ``/vault`` REST API (Task 7 — core upload surface).

Offline against :class:`~friday.vault.cloudinary.FakeCloudinary` and a
``:memory:`` SQLite index, with a ``TestClient`` whose ``FRIDAY_ENABLE_VAULT``
flag (and other settings) are forced via a monkeypatched ``get_settings``
(mirroring the study API tests). ``get_settings.cache_clear()`` is called before
every app build — a previously cached auth-on ``Settings`` leaking into a build
is a known trap in this repo (it makes an otherwise-passing test 401 only when
run as part of the full suite). No network, no key.

Covered:
* Every ``/vault`` surface is 404 when the flag is off.
* ``POST /vault/sign`` returns upload params + a pending item; falls back to
  ``FakeCloudinary`` when no Cloudinary credentials are configured.
* ``POST /vault/items/{id}/commit`` is REFUSED with 409 when Cloudinary does not
  have the asset (the anti-forgery guarantee) — commit succeeds and records
  ``secure_url`` once the fake Cloudinary does.
* The ``public_id`` ``sign`` hands back and the one ``commit`` verifies against
  agree (the historical "doubled path" bug class).
* Commit enforces ownership (index lookups are owner-scoped), is safe to call
  twice, and never regresses an item past ``uploaded``.
* List (filters + limit), get, delete (cascades to the Cloudinary asset first),
  quota, and 507 over quota.
* A malformed ``limit``/``include_locked`` query param 422s, never 500s.
* A ``FirestoreIndexError`` from the index surfaces as a 503, not an empty
  result or an unhandled 500.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.api.routes_vault import _LOCAL_OWNER
from friday.config import Settings, get_settings
from friday.providers.llm import FakeLLM, LLMResponse
from friday.vault.exam import ExamRunner
from friday.vault.firestore_index import FirestoreIndexError
from friday.vault.models import (
    CaptureSource,
    Classification,
    Consensus,
    Item,
    ItemStatus,
    Note,
    Privacy,
    Solve,
    Verification,
    VerificationStatus,
)
from friday.vault.notes import NoteWriter
from friday.vault.quota import QuotaGuard
from friday.vault.solver import Solver


def _settings(**overrides: object) -> Settings:
    fields: dict[str, object] = {
        "_env_file": None,
        "llm_provider": "fake",
        "memory_db_path": ":memory:",
        "enable_vault": True,
        "vault_index_backend": "sqlite",
        "vault_sqlite_path": ":memory:",
        "vault_quota_gb": 25.0,
        "vault_daily_solve_cap": 200,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def _disabled_settings() -> Settings:
    return _settings(enable_vault=False)


def _client(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> TestClient:
    """A ``TestClient`` whose settings are forced via a patched ``get_settings``."""
    import friday.app as app_module

    get_settings.cache_clear()
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    app = app_module.create_app()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_vault_disabled_sign_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.post("/vault/sign", json={})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "vault disabled"


def test_vault_disabled_commit_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.post("/vault/items/x/commit", json={})
    assert resp.status_code == 404


def test_vault_disabled_list_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.get("/vault/items")
    assert resp.status_code == 404


def test_vault_disabled_get_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.get("/vault/items/x")
    assert resp.status_code == 404


def test_vault_disabled_delete_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.delete("/vault/items/x")
    assert resp.status_code == 404


def test_vault_disabled_quota_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.get("/vault/quota")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# sign
# --------------------------------------------------------------------------- #
def test_vault_sign_returns_upload_and_pending_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.post("/vault/sign", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["item_id"]
        upload = body["upload"]
        assert "signature" in upload["params"]
        assert upload["params"]["public_id"] == f"vault/{_LOCAL_OWNER}/{body['item_id']}"

        item = client.get(f"/vault/items/{body['item_id']}").json()
        assert item["status"] == "pending"
        assert item["owner_uid"] == _LOCAL_OWNER
        assert item["cloudinary"] is None


def test_vault_sign_unconfigured_cloudinary_falls_back_to_fake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Cloudinary credentials configured -> the fake provider signs it."""
    with _client(monkeypatch, _settings()) as client:
        resp = client.post("/vault/sign", json={})
        upload = resp.json()["upload"]
    assert upload["url"] == "https://api.cloudinary.com/v1_1/fake/image/upload"


def test_vault_sign_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.post(
            "/vault/sign",
            json={"source": "screen", "privacy": "shared", "space": "family", "device_id": "d1"},
        )
        item_id = resp.json()["item_id"]
        item = client.get(f"/vault/items/{item_id}").json()
    assert item["source"] == "screen"
    assert item["privacy"] == "shared"
    assert item["space"] == "family"
    assert item["device_id"] == "d1"


def test_vault_sign_bad_body_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.post("/vault/sign", json={"privacy": "not-a-real-privacy"})
    assert resp.status_code == 422


def test_vault_sign_507_when_over_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings(vault_quota_gb=0.0)) as client:
        resp = client.post("/vault/sign", json={})
    assert resp.status_code == 507


# --------------------------------------------------------------------------- #
# commit — the anti-forgery guarantee
# --------------------------------------------------------------------------- #
def test_vault_commit_refused_409_when_cloudinary_missing_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, _settings()) as client:
        item_id = client.post("/vault/sign", json={}).json()["item_id"]
        resp = client.post(f"/vault/items/{item_id}/commit", json={})
        assert resp.status_code == 409
        # The item stays pending — an unverified claim never mutates state.
        item = client.get(f"/vault/items/{item_id}").json()
        assert item["status"] == "pending"
        assert item["cloudinary"] is None


def test_vault_commit_succeeds_and_records_secure_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, _settings()) as client:
        signed = client.post("/vault/sign", json={}).json()
        item_id = signed["item_id"]
        public_id = signed["upload"]["params"]["public_id"]

        # Simulate the phone having actually uploaded to Cloudinary: seed the
        # fake provider's asset table under the SAME public_id sign() handed out.
        cloudinary = client.app.state.vault_cloudinary
        cloudinary.assets[public_id] = {
            "public_id": public_id,
            "version": 42,
            "format": "jpg",
            "bytes": 123_456,
            "width": 800,
            "height": 600,
            "resource_type": "image",
            "secure_url": f"https://res.cloudinary.com/fake/image/authenticated/{public_id}.jpg",
        }

        resp = client.post(f"/vault/items/{item_id}/commit", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "uploaded"
        assert body["cloudinary"]["public_id"] == public_id
        assert body["cloudinary"]["bytes"] == 123_456
        assert body["cloudinary"]["secure_url"] == (
            f"https://res.cloudinary.com/fake/image/authenticated/{public_id}.jpg"
        )

        # GET serves the SAME stored secure_url (never a hand-rolled one).
        fetched = client.get(f"/vault/items/{item_id}").json()
        assert fetched["cloudinary"]["secure_url"] == body["cloudinary"]["secure_url"]


def test_vault_commit_public_id_agrees_with_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route never re-derives ``public_id`` locally; sign and commit must agree.

    This is the regression test for the historical "doubled folder path" bug: if
    commit's independently-computed public_id ever drifted from sign's, seeding
    the fake asset table under sign's public_id would make commit 409 instead of
    200, because it would be asking Cloudinary about the WRONG id.
    """
    with _client(monkeypatch, _settings()) as client:
        signed = client.post("/vault/sign", json={}).json()
        item_id = signed["item_id"]
        public_id = signed["upload"]["params"]["public_id"]
        client.app.state.vault_cloudinary.assets[public_id] = {
            "public_id": public_id,
            "version": 1,
            "format": "png",
            "bytes": 10,
        }
        resp = client.post(f"/vault/items/{item_id}/commit", json={})
    assert resp.status_code == 200


def test_vault_commit_unknown_item_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.post("/vault/items/does-not-exist/commit", json={})
    assert resp.status_code == 404


def test_vault_commit_item_owned_by_someone_else_is_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller cannot commit an item filed under a different owner_uid.

    Every index lookup here is scoped to the caller's own owner_uid (the local
    owner, absent a real auth layer), so an item that exists but is owned by
    someone else simply never matches — proving ownership is enforced at the
    query, not forgotten.
    """
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_item(
            Item(
                id="foreign-item",
                owner_uid="someone-else",
                source=CaptureSource.CAMERA,
                created_at="2024-01-01T00:00:00Z",
            )
        )
        resp = client.post("/vault/items/foreign-item/commit", json={})
    assert resp.status_code == 404


def test_vault_commit_twice_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        signed = client.post("/vault/sign", json={}).json()
        item_id = signed["item_id"]
        public_id = signed["upload"]["params"]["public_id"]
        client.app.state.vault_cloudinary.assets[public_id] = {
            "public_id": public_id,
            "bytes": 5,
            "format": "jpg",
        }
        # No pipeline here: this test is about commit alone, and with one wired
        # the capture races ahead to READY between the two calls, which would be
        # testing the pipeline's speed rather than commit's idempotency.
        client.app.state.vault_pipeline = None  # type: ignore[attr-defined]
        first = client.post(f"/vault/items/{item_id}/commit", json={})
        second = client.post(f"/vault/items/{item_id}/commit", json={})
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "uploaded"


def test_vault_commit_does_not_regress_ready_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committing an item already past ``uploaded`` never regresses its status."""
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        cloudinary = client.app.state.vault_cloudinary
        public_id = cloudinary.upload_params(
            owner_uid=_LOCAL_OWNER, item_id="ready-item"
        ).params["public_id"]
        cloudinary.assets[public_id] = {"public_id": public_id, "bytes": 7, "format": "jpg"}
        index.put_item(
            Item(
                id="ready-item",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                status=ItemStatus.READY,
                created_at="2024-01-01T00:00:00Z",
            )
        )
        resp = client.post("/vault/items/ready-item/commit", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


# --------------------------------------------------------------------------- #
# list / get
# --------------------------------------------------------------------------- #
def test_vault_list_items_filters_and_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_item(
            Item(
                id="1",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                created_at="2024-01-01T00:00:00Z",
                classification=Classification(subject="math", kind="worksheet"),
            )
        )
        index.put_item(
            Item(
                id="2",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                created_at="2024-01-02T00:00:00Z",
                privacy=Privacy.LOCKED,
                classification=Classification(subject="math", kind="worksheet"),
            )
        )
        index.put_item(
            Item(
                id="3",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                created_at="2024-01-03T00:00:00Z",
                classification=Classification(subject="history", kind="notes"),
            )
        )

        default_math = client.get("/vault/items?subject=math").json()
        assert default_math["count"] == 1
        assert [i["id"] for i in default_math["items"]] == ["1"]

        with_locked = client.get("/vault/items?subject=math&include_locked=true").json()
        assert with_locked["count"] == 2

        by_kind = client.get("/vault/items?kind=notes").json()
        assert [i["id"] for i in by_kind["items"]] == ["3"]

        limited = client.get("/vault/items?include_locked=true&limit=1").json()
        assert limited["count"] == 1


def test_vault_list_items_bad_limit_is_422_not_500(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.get("/vault/items?limit=not-a-number")
    assert resp.status_code == 422


def test_vault_list_items_bad_include_locked_is_422_not_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.get("/vault/items?include_locked=maybe")
    assert resp.status_code == 422


def test_vault_get_item_unknown_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.get("/vault/items/does-not-exist")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# delete
# --------------------------------------------------------------------------- #
def test_vault_delete_removes_cloudinary_asset_then_index_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, _settings()) as client:
        signed = client.post("/vault/sign", json={}).json()
        item_id = signed["item_id"]
        public_id = signed["upload"]["params"]["public_id"]
        cloudinary = client.app.state.vault_cloudinary
        cloudinary.assets[public_id] = {"public_id": public_id, "bytes": 9, "format": "jpg"}
        client.post(f"/vault/items/{item_id}/commit", json={})

        resp = client.delete(f"/vault/items/{item_id}")
        assert resp.status_code == 200
        assert resp.json() == {"id": item_id, "removed": True}
        assert public_id not in cloudinary.assets
        assert client.get(f"/vault/items/{item_id}").status_code == 404


def test_vault_delete_pending_item_without_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A never-committed item has no Cloudinary asset; delete must not crash on it."""
    with _client(monkeypatch, _settings()) as client:
        item_id = client.post("/vault/sign", json={}).json()["item_id"]
        resp = client.delete(f"/vault/items/{item_id}")
    assert resp.status_code == 200
    assert resp.json()["removed"] is True


def test_vault_delete_unknown_item_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.delete("/vault/items/does-not-exist")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# quota
# --------------------------------------------------------------------------- #
def test_vault_quota_reports_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings(vault_quota_gb=1.0)) as client:
        empty = client.get("/vault/quota").json()
        assert empty["used_bytes"] == 0
        assert empty["quota_bytes"] == 1024**3
        assert empty["warning"] is False

        signed = client.post("/vault/sign", json={}).json()
        item_id = signed["item_id"]
        public_id = signed["upload"]["params"]["public_id"]
        client.app.state.vault_cloudinary.assets[public_id] = {
            "public_id": public_id,
            "bytes": 500,
            "format": "jpg",
        }
        client.post(f"/vault/items/{item_id}/commit", json={})

        after = client.get("/vault/quota").json()
        assert after["used_bytes"] == 500


# --------------------------------------------------------------------------- #
# Index outage -> 503, never a quiet empty result or an unhandled 500.
# --------------------------------------------------------------------------- #
class _ErrorIndex:
    """A minimal ``VaultIndex`` whose reads always raise ``FirestoreIndexError``."""

    def put_item(self, item: object) -> None:
        return None

    def get_item(self, owner_uid: str, item_id: str) -> object:
        raise FirestoreIndexError("simulated outage")

    def list_items(self, owner_uid: str, **_: object) -> object:
        raise FirestoreIndexError("simulated outage")

    def delete_item(self, owner_uid: str, item_id: str) -> object:
        raise FirestoreIndexError("simulated outage")

    def put_solve(self, solve: object) -> None:
        return None

    def get_solve(self, solve_id: str) -> None:
        return None

    def put_note(self, note: object) -> None:
        return None

    def get_note(self, owner_uid: str, note_id: str) -> None:
        return None

    def list_notes(self, owner_uid: str, **_: object) -> list[object]:
        return []

    def total_bytes(self, owner_uid: str) -> int:
        raise FirestoreIndexError("simulated outage")


def test_vault_index_outage_surfaces_as_503(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        error_index = _ErrorIndex()
        client.app.state.vault_index = error_index
        client.app.state.vault_quota = QuotaGuard(
            error_index, quota_gb=25.0, daily_solve_cap=200
        )

        assert client.get("/vault/quota").status_code == 503
        assert client.get("/vault/items").status_code == 503
        assert client.get("/vault/items/x").status_code == 503
        assert client.delete("/vault/items/x").status_code == 503
        assert client.post("/vault/sign", json={}).status_code == 503
        assert client.post("/vault/items/x/commit", json={}).status_code == 503


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every Task 8 surface too
# --------------------------------------------------------------------------- #
def test_vault_disabled_solve_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.post("/vault/items/x/solve")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "vault disabled"


def test_vault_disabled_notes_post_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.post("/vault/notes", json={"item_ids": []})
    assert resp.status_code == 404


def test_vault_disabled_notes_list_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.get("/vault/notes")
    assert resp.status_code == 404


def test_vault_disabled_note_get_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.get("/vault/notes/x")
    assert resp.status_code == 404


def test_vault_disabled_search_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.get("/vault/search?q=x")
    assert resp.status_code == 404


def test_vault_disabled_analytics_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.get("/vault/analytics")
    assert resp.status_code == 404


def test_vault_disabled_exam_start_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.post("/vault/exam/start", json={"paper_item_ids": []})
    assert resp.status_code == 404


def test_vault_disabled_exam_grade_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _disabled_settings()) as client:
        resp = client.post("/vault/exam/x/grade", json={"answer_item_ids": []})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# solve
# --------------------------------------------------------------------------- #
_DRAFT_REPLY = (
    '{"subject": "algebra", "statement": "2x+4=10", "latex": "", '
    '"steps": ["subtract 4", "divide by 2"], "final_answer": "x = 3", '
    '"equation": "2*x + 4 = 10", "confidence": 0.9}'
)


def test_vault_solve_persists_and_returns_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_item(
            Item(
                id="i1",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                ocr_text="2*x + 4 = 10",
                created_at="2024-01-01T00:00:00Z",
            )
        )
        # All three operators agree, so no judge call is spent -- three drafts is
        # exactly enough script for the whole panel.
        client.app.state.vault_solver = Solver(
            FakeLLM([LLMResponse(text=_DRAFT_REPLY) for _ in range(3)]),
            operators=["VISION", "ORACLE", "GECKO"],
        )

        resp = client.post("/vault/items/i1/solve")
        assert resp.status_code == 200
        body = resp.json()
        assert body["consensus"]["final_answer"] == "x = 3"
        assert body["verification"]["status"] == "verified"

        item = client.get("/vault/items/i1").json()
        assert item["solve_id"] == body["id"]


def test_vault_solve_locked_item_is_403_and_never_calls_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_item(
            Item(
                id="i1",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                privacy=Privacy.LOCKED,
                ocr_text="2*x + 4 = 10",
                created_at="2024-01-01T00:00:00Z",
            )
        )
        # An empty script: if the route ever reached the solver, the very first
        # provider call would raise -- proving the 403 is returned before that.
        client.app.state.vault_solver = Solver(FakeLLM([]), operators=["VISION"])

        resp = client.post("/vault/items/i1/solve")
        assert resp.status_code == 403

        item = client.get("/vault/items/i1").json()
        assert item["solve_id"] is None


def test_vault_solve_unknown_item_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        client.app.state.vault_solver = Solver(FakeLLM([]), operators=["VISION"])
        resp = client.post("/vault/items/does-not-exist/solve")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# notes
# --------------------------------------------------------------------------- #
_NOTE_REPLY = (
    '{"title": "EMF and internal resistance", "subject": "physics", '
    '"chapter": "circuits", "markdown": "# EMF\\nWork the drop.", '
    '"cards": [{"front": "What is EMF?", "back": "Energy per unit charge."}]}'
)


def test_vault_note_generated_over_items_and_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_item(
            Item(
                id="i1",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                ocr_text="A cell of emf 2V and internal resistance 1 ohm...",
                created_at="2024-01-01T00:00:00Z",
            )
        )
        client.app.state.vault_notes = NoteWriter(
            index=index,
            llm=FakeLLM([LLMResponse(text=_NOTE_REPLY)]),
            ingestor=None,
            study_store=None,
        )

        resp = client.post("/vault/notes", json={"item_ids": ["i1"], "prompt": "keep it short"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "EMF and internal resistance"
        assert body["item_ids"] == ["i1"]
        note_id = body["id"]

        # The source item now points back at the note.
        item = client.get("/vault/items/i1").json()
        assert note_id in item["note_ids"]

        listed = client.get("/vault/notes").json()
        assert listed["count"] == 1
        assert listed["notes"][0]["id"] == note_id

        fetched = client.get(f"/vault/notes/{note_id}").json()
        assert fetched["id"] == note_id


def test_vault_note_missing_item_ids_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.post("/vault/notes", json={})
    assert resp.status_code == 422


def test_vault_note_empty_or_nonexistent_item_ids_still_returns_a_minimal_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No source material -> the model is never called, but a real note id comes back."""
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        client.app.state.vault_notes = NoteWriter(
            index=index, llm=FakeLLM([]), ingestor=None, study_store=None
        )

        empty = client.post("/vault/notes", json={"item_ids": []})
        assert empty.status_code == 200
        assert empty.json()["item_ids"] == []
        assert empty.json()["markdown"] == ""

        missing = client.post("/vault/notes", json={"item_ids": ["does-not-exist"]})
        assert missing.status_code == 200
        assert missing.json()["item_ids"] == []


def test_vault_note_provider_failure_still_returns_a_usable_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead model degrades to a bare-bones note, never a 500 traceback."""
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_item(
            Item(
                id="i1",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                ocr_text="some raw extracted ocr text",
                created_at="2024-01-01T00:00:00Z",
            )
        )
        client.app.state.vault_notes = NoteWriter(
            index=index, llm=FakeLLM([]), ingestor=None, study_store=None
        )

        resp = client.post("/vault/notes", json={"item_ids": ["i1"]})
        assert resp.status_code == 200
        assert "some raw extracted ocr text" in resp.json()["markdown"]


def test_vault_note_owned_by_someone_else_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot read another owner's note by id."""
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_note(
            Note(id="foreign-note", owner_uid="someone-else", created_at="2024-01-01T00:00:00Z")
        )
        resp = client.get("/vault/notes/foreign-note")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def test_vault_search_returns_hits_and_excludes_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_item(
            Item(
                id="1",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                ocr_text="resistance in a series circuit",
                created_at="2024-01-01T00:00:00Z",
            )
        )
        index.put_item(
            Item(
                id="2",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                ocr_text="resistance is futile but locked",
                privacy=Privacy.LOCKED,
                created_at="2024-01-02T00:00:00Z",
            )
        )

        resp = client.get("/vault/search?q=resistance")
        assert resp.status_code == 200
        body = resp.json()
        assert [h["item_id"] for h in body["hits"]] == ["1"]
        assert body["count"] == 1


def test_vault_search_no_query_returns_no_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.get("/vault/search")
    assert resp.status_code == 200
    assert resp.json() == {"hits": [], "count": 0}


# --------------------------------------------------------------------------- #
# analytics
# --------------------------------------------------------------------------- #
def test_vault_analytics_returns_rollup(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_solve(
            Solve(
                id="s1",
                item_ids=["i1"],
                created_at="2024-01-01T00:00:00Z",
                verification=Verification(status=VerificationStatus.VERIFIED),
                consensus=Consensus(final_answer="x = 3", agreement="3/3"),
            )
        )
        index.put_item(
            Item(
                id="i1",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                solve_id="s1",
                classification=Classification(subject="math", chapter="algebra"),
                created_at="2024-01-01T00:00:00Z",
            )
        )

        resp = client.get("/vault/analytics")
        assert resp.status_code == 200
        chapters = resp.json()["chapters"]
        assert len(chapters) == 1
        assert chapters[0]["subject"] == "math"
        assert chapters[0]["chapter"] == "algebra"
        assert chapters[0]["mastery"] == 1.0
        assert chapters[0]["basis"] == "sympy"


# --------------------------------------------------------------------------- #
# exam
# --------------------------------------------------------------------------- #
_GRADING_REPLY = (
    '{"questions": [{"q": "Q1", "marks_awarded": 1, "marks_total": 1, '
    '"feedback": "correct"}], "total": 1}'
)


def test_vault_exam_start_then_grade_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        index = client.app.state.vault_index
        index.put_item(
            Item(
                id="paper",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                ocr_text="Q1: what is 2 + 2?",
                created_at="2024-01-01T00:00:00Z",
            )
        )
        index.put_item(
            Item(
                id="answer",
                owner_uid=_LOCAL_OWNER,
                source=CaptureSource.CAMERA,
                ocr_text="Q1: 4",
                created_at="2024-01-01T00:00:00Z",
            )
        )
        client.app.state.vault_exam = ExamRunner(
            index=index, llm=FakeLLM([LLMResponse(text=_GRADING_REPLY)])
        )

        started = client.post(
            "/vault/exam/start", json={"paper_item_ids": ["paper"], "duration_s": 600}
        )
        assert started.status_code == 200
        session = started.json()
        assert session["status"] == "open"
        session_id = session["id"]

        graded = client.post(
            f"/vault/exam/{session_id}/grade", json={"answer_item_ids": ["answer"]}
        )
        assert graded.status_code == 200
        body = graded.json()
        assert body["status"] == "graded"
        assert body["total"] == 1.0
        assert body["grading"][0]["q"] == "Q1"


def test_vault_exam_grade_unknown_session_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        client.app.state.vault_exam = ExamRunner(
            index=client.app.state.vault_index, llm=FakeLLM([])
        )
        resp = client.post("/vault/exam/does-not-exist/grade", json={"answer_item_ids": []})
    assert resp.status_code == 404


def test_vault_exam_grade_foreign_owner_session_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot grade a session that belongs to a different owner.

    ``ExamRunner.grade`` raises ``KeyError`` for both an unknown session and a
    session owned by someone else, on purpose (see the module docstring) --
    this proves the route maps that to 404, not 403, so the response can never
    be used to distinguish "doesn't exist" from "exists but isn't yours".
    """
    with _client(monkeypatch, _settings()) as client:
        runner = ExamRunner(index=client.app.state.vault_index, llm=FakeLLM([]))
        client.app.state.vault_exam = runner
        session = runner.start("someone-else", ["paper"], duration_s=0)

        resp = client.post(f"/vault/exam/{session.id}/grade", json={"answer_item_ids": []})
    assert resp.status_code == 404


def test_vault_exam_start_missing_paper_item_ids_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        resp = client.post("/vault/exam/start", json={})
    assert resp.status_code == 422


def test_vault_exam_grade_missing_answer_item_ids_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, _settings()) as client:
        runner = ExamRunner(index=client.app.state.vault_index, llm=FakeLLM([]))
        client.app.state.vault_exam = runner
        session = runner.start(_LOCAL_OWNER, ["paper"], duration_s=0)

        resp = client.post(f"/vault/exam/{session.id}/grade", json={})
    assert resp.status_code == 422


def test_vault_exam_grade_nothing_gradable_is_422_not_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every item textless/missing -> a clean 422, not a leaked 500."""
    with _client(monkeypatch, _settings()) as client:
        runner = ExamRunner(index=client.app.state.vault_index, llm=FakeLLM([]))
        client.app.state.vault_exam = runner
        session = runner.start(_LOCAL_OWNER, ["does-not-exist"], duration_s=0)

        resp = client.post(
            f"/vault/exam/{session.id}/grade", json={"answer_item_ids": ["also-missing"]}
        )
    assert resp.status_code == 422


def _seed_uploaded(client: TestClient, source: str = "camera") -> str:
    """Sign, then simulate the phone having actually uploaded, and return the id.

    Commit refuses (409) anything Cloudinary cannot confirm, so a test that
    wants a successful commit has to put the asset where the provider will
    find it.
    """
    signed = client.post("/vault/sign", json={"source": source}).json()
    public_id = signed["upload"]["params"]["public_id"]
    client.app.state.vault_cloudinary.assets[public_id] = {  # type: ignore[attr-defined]
        "public_id": public_id,
        "version": 7,
        "format": "jpg",
        "bytes": 5_774,
        "width": 900,
        "height": 300,
        "resource_type": "image",
        "secure_url": f"https://res.cloudinary.com/fake/{public_id}.jpg",
    }
    return str(signed["item_id"])


# --------------------------------------------------------------------------- #
# Commit hands the capture to the pipeline (Task 10 Step 5)
# --------------------------------------------------------------------------- #
def test_commit_kicks_off_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Commit must start the pipeline, or a filed capture is never read.

    Without this the item stops at ``uploaded`` forever: no OCR, no
    classification, never ``ready`` — which is the whole point of photographing
    it. Verified against production, where exactly that happened.
    """
    seen: list[tuple[str, str]] = []

    class _RecordingPipeline:
        async def process(self, owner_uid: str, item_id: str) -> None:
            seen.append((owner_uid, item_id))

    with _client(monkeypatch, _settings()) as client:
        client.app.state.vault_pipeline = _RecordingPipeline()  # type: ignore[attr-defined]
        item_id = _seed_uploaded(client)
        resp = client.post(f"/vault/items/{item_id}/commit", json={})

    assert resp.status_code == 200
    assert seen == [(_LOCAL_OWNER, item_id)]


def test_commit_survives_a_missing_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A build with no pipeline wired still commits; the capture is not lost."""
    with _client(monkeypatch, _settings()) as client:
        client.app.state.vault_pipeline = None  # type: ignore[attr-defined]
        item_id = _seed_uploaded(client)
        resp = client.post(f"/vault/items/{item_id}/commit", json={})
    assert resp.status_code == 200


def test_vault_pipeline_is_wired_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wiring itself: app.state carries a pipeline when the vault is on."""
    with _client(monkeypatch, _settings()) as client:
        assert getattr(client.app.state, "vault_pipeline", None) is not None  # type: ignore[attr-defined]
