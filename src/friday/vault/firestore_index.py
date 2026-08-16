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

Two failure modes get careful handling, both stemming from the same rule:
it is fine to report an empty vault only when the vault is genuinely empty,
never when we could not find out.

* Firestore's REST list endpoint pages at ~300 documents (see
  ``friday.circle.firestore_rest``'s hardcoded ``pageSize=300``) and returns
  ``nextPageToken`` past that. ``_list_all_documents`` loops until there is no
  token left, so a vault past one page doesn't silently lose its tail.
* An error-shaped response (``{"error": {...}}``) or an unusable response
  shape from a collection listing is raised as :class:`FirestoreIndexError`
  rather than treated as "no documents" — see that class's docstring for why.
"""

from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import quote

from friday.vault.index import _parse_row
from friday.vault.models import Item, Note, Privacy, Solve


class FirestoreTransport(Protocol):
    """One method: issue a REST call against a document or collection path.

    ``path`` may carry a query string — most notably ``?pageToken=...`` when
    :meth:`FirestoreVaultIndex._list_all_documents` pages through a collection
    larger than one page. Implementations MUST pass ``path`` through to the
    underlying HTTP call exactly as given: do not strip, re-parse, or
    re-encode a query string that's already present. This binds the
    production adapter over :mod:`friday.circle.firestore_rest` (written in a
    later task) as much as it binds the test stubs here.
    """

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any: ...


class FirestoreIndexError(RuntimeError):
    """Firestore returned something other than data or a legitimate "not found".

    Raised for an explicit error-shaped payload (``{"error": {...}}``, which
    Firestore's REST API can return with a JSON body even on a non-2xx
    response) or, for a collection listing, any response that isn't a valid
    page — a genuinely empty Firestore collection is still a dict (``{}``),
    so anything else (``None``, a string, a list...) means the transport
    could not get a real answer, not that the vault is empty.

    This distinction matters most for :meth:`FirestoreVaultIndex.total_bytes`,
    which runs on every upload signature request to enforce the storage
    quota: silently treating "Firestore is down" as "zero items" would let
    the quota check pass when it should refuse. The route layer calling into
    this index is responsible for catching this and returning a 5xx, not for
    treating it as any kind of empty or missing result.

    Genuine transport-level exceptions (network errors, timeouts, etc. raised
    by the injected :class:`FirestoreTransport`) are deliberately never
    caught anywhere in this module and propagate as-is — this exception is
    only for responses that came back but cannot be trusted.
    """


def _wrap(doc_json: str) -> dict[str, Any]:
    return {"fields": {"doc": {"stringValue": doc_json}}}


def _unwrap(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get("fields", {}).get("doc", {}).get("stringValue")
    return value if isinstance(value, str) else None


def _raise_if_error_payload(raw: Any, path: str) -> None:
    if isinstance(raw, dict) and "error" in raw:
        raise FirestoreIndexError(f"Firestore returned an error for {path}: {raw['error']!r}")


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

    def _list_all_documents(self, base_path: str) -> list[Any]:
        """Page through a whole collection, following ``nextPageToken`` to the end.

        Stopping after one page would silently drop everything past
        Firestore's ~300-document page cap — see the module docstring. Raises
        :class:`FirestoreIndexError` rather than returning a partial result
        for any response that isn't a genuine (possibly empty) page, per the
        "never mistake an outage for an empty vault" rule.
        """
        documents: list[Any] = []
        path = base_path
        while True:
            raw = self._t.request("GET", path)
            _raise_if_error_payload(raw, path)
            if not isinstance(raw, dict):
                raise FirestoreIndexError(
                    f"Firestore returned an unusable response listing {path}: {raw!r}"
                )
            documents.extend(raw.get("documents", []))
            token = raw.get("nextPageToken")
            if not token:
                return documents
            path = f"{base_path}?pageToken={quote(str(token), safe='')}"

    # ---------------------------------------------------------------- items
    def put_item(self, item: Item) -> None:
        self._t.request(
            "PATCH",
            f"vaults/{item.owner_uid}/items/{item.id}",
            _wrap(item.model_dump_json()),
        )

    def get_item(self, owner_uid: str, item_id: str) -> Item | None:
        path = f"vaults/{owner_uid}/items/{item_id}"
        raw = self._t.request("GET", path)
        _raise_if_error_payload(raw, path)
        doc = _unwrap(raw)
        if not doc:
            return None
        return _parse_row(Item, item_id, doc, "item")

    def _fetch_items(
        self,
        owner_uid: str,
        *,
        subject: str = "",
        kind: str = "",
        space: str = "",
        include_locked: bool = False,
    ) -> list[Item]:
        """Every matching item for ``owner_uid``, newest first, unpaginated and unlimited.

        Shared by :meth:`list_items` (which then slices to ``limit``) and
        :meth:`total_bytes` (which must not slice at all — see its
        docstring).
        """
        documents = self._list_all_documents(f"vaults/{owner_uid}/items")
        items: list[Item] = []
        for position, document in enumerate(documents):
            doc = _unwrap(document)
            if not doc:
                continue
            item = _parse_row(Item, _doc_id(document, str(position)), doc, "item")
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
        # Secondary key on id keeps ordering deterministic when two items
        # share a created_at, matching SQLite's `ORDER BY created_at DESC, id
        # DESC` — a collection listing has no inherent order to fall back on.
        items.sort(key=lambda i: (i.created_at, i.id), reverse=True)
        return items

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
        return self._fetch_items(
            owner_uid, subject=subject, kind=kind, space=space, include_locked=include_locked
        )[:limit]

    def delete_item(self, owner_uid: str, item_id: str) -> bool:
        self._t.request("DELETE", f"vaults/{owner_uid}/items/{item_id}")
        return True

    # --------------------------------------------------------- solves/notes
    def put_solve(self, solve: Solve) -> None:
        self._t.request("PATCH", f"solves/{solve.id}", _wrap(solve.model_dump_json()))

    def get_solve(self, solve_id: str) -> Solve | None:
        path = f"solves/{solve_id}"
        raw = self._t.request("GET", path)
        _raise_if_error_payload(raw, path)
        doc = _unwrap(raw)
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
        path = f"vaults/{owner_uid}/notes/{note_id}"
        raw = self._t.request("GET", path)
        _raise_if_error_payload(raw, path)
        doc = _unwrap(raw)
        if not doc:
            return None
        return _parse_row(Note, note_id, doc, "note")

    def list_notes(self, owner_uid: str, *, limit: int = 100) -> list[Note]:
        documents = self._list_all_documents(f"vaults/{owner_uid}/notes")
        notes: list[Note] = []
        for position, document in enumerate(documents):
            doc = _unwrap(document)
            if not doc:
                continue
            note = _parse_row(Note, _doc_id(document, str(position)), doc, "note")
            if note is not None:
                notes.append(note)
        notes.sort(key=lambda n: (n.created_at, n.id), reverse=True)
        return notes[:limit]

    def total_bytes(self, owner_uid: str) -> int:
        """Sum every committed Cloudinary asset's bytes for ``owner_uid``, locked included.

        Reads and parses *every* document in the owner's item collection —
        all pages, no limit — because this feeds the storage quota check on
        every upload signature request, and undercounting here (e.g. by
        capping the read) would let an upload through that should have been
        refused. Locked items still occupy storage, so they count too,
        matching :meth:`SQLiteVaultIndex.total_bytes`, which sums
        unconditionally.

        Known scaling limit, flagged rather than fixed here: unlike SQLite's
        promoted ``bytes`` column, there is no server-side sum available, so
        this is one Firestore document read per item in the vault, on every
        photo upload. At a few thousand items that starts to burn the
        free-tier read quota. The correct fix is a per-owner aggregate
        counter document maintained on write, not a full scan on every read —
        deliberately not built in this task; it belongs to the quota-guard
        work that follows.
        """
        return sum(
            item.cloudinary.bytes
            for item in self._fetch_items(owner_uid, include_locked=True)
            if item.cloudinary is not None
        )
