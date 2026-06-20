"""Firestore-backed implementations of the circle's storage protocols.

:class:`FirestoreCircleStore` satisfies
:class:`~friday.circle.store.CircleStore` and :class:`FirestoreChatStore` satisfies
:class:`~friday.circle.chat.ChatStore`, so the services swap from in-memory to
persistent without any change to the guardrail layer or the routes. The Firestore
client is passed in (built lazily in :mod:`friday.circle.firebase`); this module
never imports ``firebase-admin`` at module load, so the offline build is unaffected.

Documents store the pydantic models in JSON mode (ISO-8601 timestamps), which sort
lexically — so ``order_by`` on a timestamp field is chronological. A ``member_uids``
array on each group doc powers the ``array_contains`` membership query.
"""

from __future__ import annotations

from typing import Any

from friday.circle.chat import ChatMessage
from friday.circle.models import Group, Invite, Member


class FirestoreCircleStore:
    """A Firestore-backed :class:`~friday.circle.store.CircleStore`."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def _groups(self) -> Any:
        return self._db.collection("groups")

    def _members(self, group_id: str) -> Any:
        return self._groups().document(group_id).collection("members")

    def create_group(self, group: Group) -> None:
        data = group.model_dump(mode="json")
        data["member_uids"] = []
        self._groups().document(group.id).set(data)

    def get_group(self, group_id: str) -> Group | None:
        snap = self._groups().document(group_id).get()
        return Group(**snap.to_dict()) if snap.exists else None

    def add_member(self, group_id: str, member: Member) -> None:
        self._members(group_id).document(member.uid).set(member.model_dump(mode="json"))
        ref = self._groups().document(group_id)
        snap = ref.get()
        uids = list(snap.to_dict().get("member_uids", [])) if snap.exists else []
        if member.uid not in uids:
            uids.append(member.uid)
            ref.set({"member_uids": uids}, merge=True)

    def get_member(self, group_id: str, uid: str) -> Member | None:
        snap = self._members(group_id).document(uid).get()
        return Member(**snap.to_dict()) if snap.exists else None

    def list_members(self, group_id: str) -> list[Member]:
        docs = self._members(group_id).order_by("joined_at").stream()
        return [Member(**doc.to_dict()) for doc in docs]

    def remove_member(self, group_id: str, uid: str) -> bool:
        ref = self._members(group_id).document(uid)
        existed = bool(ref.get().exists)
        ref.delete()
        gref = self._groups().document(group_id)
        snap = gref.get()
        if snap.exists:
            kept = [u for u in snap.to_dict().get("member_uids", []) if u != uid]
            gref.set({"member_uids": kept}, merge=True)
        return existed

    def groups_of(self, uid: str) -> set[str]:
        docs = self._groups().where("member_uids", "array_contains", uid).stream()
        return {doc.id for doc in docs}

    def save_invite(self, invite: Invite) -> None:
        self._db.collection("invites").document(invite.code).set(
            invite.model_dump(mode="json")
        )

    def get_invite(self, code: str) -> Invite | None:
        snap = self._db.collection("invites").document(code).get()
        return Invite(**snap.to_dict()) if snap.exists else None


class FirestoreChatStore:
    """A Firestore-backed :class:`~friday.circle.chat.ChatStore`."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def _messages(self, group_id: str) -> Any:
        return self._db.collection("groups").document(group_id).collection("messages")

    def save(self, message: ChatMessage) -> None:
        self._messages(message.group_id).document(message.id).set(
            message.model_dump(mode="json")
        )

    def history(self, group_id: str, limit: int = 200) -> list[ChatMessage]:
        docs = (
            self._messages(group_id)
            .order_by("created_at", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
        messages = [ChatMessage(**doc.to_dict()) for doc in docs]
        messages.reverse()
        return messages
