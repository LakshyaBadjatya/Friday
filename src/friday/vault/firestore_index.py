"""The production vault index, over Firestore's REST API.

The whole document is stored as one JSON string in a ``doc`` field, exactly as
the SQLite index does, so both backends decode through the same pydantic models
and there is no second schema to keep in step. Filtering happens client-side on
the small result set a personal vault produces — a structured-query builder
would be a second encoding to maintain for no benefit at this scale.

Transport is injected so the tests need no network: production passes a thin
adapter over :mod:`friday.circle.firestore_rest`, which already solves the
caller-token exchange.

Row parsing (and its "log and skip, never raise" discipline) is imported from
:mod:`friday.vault.index` rather than duplicated here — see ``_parse_row``'s
docstring there. It is the single place that decides how a corrupt stored
document is handled, and both backends must treat model-evolution drift the
same way; two copies of that judgement call would drift out of sync exactly
the way the models it protects against already have, twice.
"""

from __future__ import annotations

from typing import Any, Protocol

from friday.vault.index import _parse_row
from friday.vault.models import Item, Note, Privacy, Solve


class FirestoreTransport(Protocol):
    """One method: issue a REST call against a document path."""

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any: ...


def _wrap(doc_json: str) -> dict[str, Any]:
    return {"fields": {"doc": {"stringValue": doc_json}}}


def _unwrap(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get("fields", {}).get("doc", {}).get("stringValue")
    return value if isinstance(value, str) else None


def _doc_id(document: Any, fallback: str) -> str:
    """Best-effort id for a log line: the last segment of Firestore's ``name``.

    A real list response's documents carry a ``name`` like
    ``projects/.../documents/vaults/u1/items/i1``; falls back to a positional
    label when that is absent (as in tests, or a malformed response), since a
    row that fails to parse should still log *something* identifiable.
    """
    name = document.get("name") if isinstance(document, dict) else None
    if isinstance(name, str) and name:
        return name.rsplit("/", 1)[-1]
    return fallback


class FirestoreVaultIndex:
    """Firestore-backed :class:`~friday.vault.index.VaultIndex`."""

    def __init__(self, transport: FirestoreTransport) -> None:
        self._t = transport

    # ---------------------------------------------------------------- items
    def put_item(self, item: Item) -> None:
        self._t.request(
            "PATCH",
            f"vaults/{item.owner_uid}/items/{item.id}",
            _wrap(item.model_dump_json()),
        )

    def get_item(self, owner_uid: str, item_id: str) -> Item | None:
        doc = _unwrap(self._t.request("GET", f"vaults/{owner_uid}/items/{item_id}"))
        if not doc:
            return None
        return _parse_row(Item, item_id, doc, "item")

    def list_items(
        self,
        owner_uid: str,
        *,
        subject: str = "",
        kind: str = "",
        space: str = "",
        include_locked: bool = False,
        limit: int = 100,
    ) -> list[Item]:
        raw = self._t.request("GET", f"vaults/{owner_uid}/items")
        documents = raw.get("documents", []) if isinstance(raw, dict) else []
        items: list[Item] = []
        for index, document in enumerate(documents):
            doc = _unwrap(document)
            if not doc:
                continue
            item = _parse_row(Item, _doc_id(document, str(index)), doc, "item")
            if item is None:
                continue
            if subject and item.classification.subject != subject:
                continue
            if kind and item.classification.kind != kind:
                continue
            if space and item.space != space:
                continue
            if not include_locked and item.privacy is Privacy.LOCKED:
                continue
            items.append(item)
        items.sort(key=lambda i: i.created_at, reverse=True)
        return items[:limit]

    def delete_item(self, owner_uid: str, item_id: str) -> bool:
        self._t.request("DELETE", f"vaults/{owner_uid}/items/{item_id}")
        return True

    # --------------------------------------------------------- solves/notes
    def put_solve(self, solve: Solve) -> None:
        self._t.request("PATCH", f"solves/{solve.id}", _wrap(solve.model_dump_json()))

    def get_solve(self, solve_id: str) -> Solve | None:
        doc = _unwrap(self._t.request("GET", f"solves/{solve_id}"))
        if not doc:
            return None
        return _parse_row(Solve, solve_id, doc, "solve")

    def put_note(self, note: Note) -> None:
        self._t.request(
            "PATCH",
            f"vaults/{note.owner_uid}/notes/{note.id}",
            _wrap(note.model_dump_json()),
        )

    def get_note(self, owner_uid: str, note_id: str) -> Note | None:
        doc = _unwrap(self._t.request("GET", f"vaults/{owner_uid}/notes/{note_id}"))
        if not doc:
            return None
        return _parse_row(Note, note_id, doc, "note")

    def list_notes(self, owner_uid: str, *, limit: int = 100) -> list[Note]:
        raw = self._t.request("GET", f"vaults/{owner_uid}/notes")
        documents = raw.get("documents", []) if isinstance(raw, dict) else []
        notes: list[Note] = []
        for index, document in enumerate(documents):
            doc = _unwrap(document)
            if not doc:
                continue
            note = _parse_row(Note, _doc_id(document, str(index)), doc, "note")
            if note is not None:
                notes.append(note)
        notes.sort(key=lambda n: n.created_at, reverse=True)
        return notes[:limit]

    def total_bytes(self, owner_uid: str) -> int:
        """Sum committed Cloudinary asset bytes, including locked items.

        Unlike SQLite there is no promoted ``bytes`` column to sum in a single
        query; this re-lists and re-parses every item. That is deliberate per
        the module docstring's filtering trade-off — acceptable at personal
        vault scale, and it must include locked items, which still consume
        storage, matching :meth:`SQLiteVaultIndex.total_bytes`.
        """
        return sum(
            item.cloudinary.bytes
            for item in self.list_items(owner_uid, include_locked=True, limit=10_000)
            if item.cloudinary is not None
        )
