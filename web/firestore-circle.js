// Client-direct Firestore data layer for the circle (keyless — no backend service
// account). The browser reads/writes Firestore using the signed-in user's login;
// access is enforced entirely by the security rules in firestore.rules. Messages
// are end-to-end encrypted before they get here, so only ciphertext is stored.

import {
  collection,
  doc,
  getDoc,
  getDocs,
  setDoc,
  updateDoc,
  addDoc,
  query,
  orderBy,
  limit,
  onSnapshot,
  serverTimestamp,
  Timestamp,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-firestore.js";

const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

function newCode() {
  return crypto.randomUUID().replace(/-/g, "");
}

export function createCircleData(db) {
  return {
    async createGroup(uid, name, displayName, tz) {
      const gid = doc(collection(db, "groups")).id;
      await setDoc(doc(db, "groups", gid), {
        name,
        adminUid: uid,
        createdAt: serverTimestamp(),
      });
      await setDoc(doc(db, "groups", gid, "members", uid), {
        uid,
        displayName,
        role: "admin",
        tz,
        joinedAt: serverTimestamp(),
      });
      await setDoc(doc(db, "users", uid, "memberships", gid), {
        groupId: gid,
        groupName: name,
        role: "admin",
      });
      return { id: gid, name };
    },

    async myGroups(uid) {
      const snap = await getDocs(collection(db, "users", uid, "memberships"));
      return snap.docs.map((d) => d.data());
    },

    async members(gid) {
      const snap = await getDocs(
        query(collection(db, "groups", gid, "members"), orderBy("joinedAt")),
      );
      return snap.docs.map((d) => d.data());
    },

    async createInvite(uid, gid, groupName) {
      const code = newCode();
      await setDoc(doc(db, "invites", code), {
        groupId: gid,
        groupName,
        createdBy: uid,
        revoked: false,
        accepted: false,
        createdAt: serverTimestamp(),
        expiresAt: Timestamp.fromMillis(Date.now() + WEEK_MS),
      });
      return { code };
    },

    async previewInvite(code) {
      const snap = await getDoc(doc(db, "invites", code));
      return snap.exists()
        ? { groupName: snap.data().groupName, groupId: snap.data().groupId }
        : null;
    },

    async acceptInvite(uid, code, displayName, tz) {
      const snap = await getDoc(doc(db, "invites", code));
      if (!snap.exists()) throw new Error("invite not found");
      const inv = snap.data();
      await setDoc(doc(db, "groups", inv.groupId, "members", uid), {
        uid,
        displayName,
        role: "member",
        tz,
        joinedAt: serverTimestamp(),
        inviteCode: code,
      });
      await setDoc(doc(db, "users", uid, "memberships", inv.groupId), {
        groupId: inv.groupId,
        groupName: inv.groupName,
        role: "member",
      });
      // Best-effort single-use marker; not required for correctness.
      try {
        await updateDoc(doc(db, "invites", code), {
          accepted: true,
          acceptedBy: uid,
        });
      } catch {
        /* the join already succeeded */
      }
      return { groupId: inv.groupId, groupName: inv.groupName };
    },

    async sendMessage(uid, gid, ciphertext, nonce) {
      await addDoc(collection(db, "groups", gid, "messages"), {
        senderUid: uid,
        ciphertext,
        nonce,
        createdAt: serverTimestamp(),
      });
    },

    // Live chat: returns an unsubscribe fn; calls onAdded for each new message.
    subscribeMessages(gid, onAdded) {
      const q = query(
        collection(db, "groups", gid, "messages"),
        orderBy("createdAt"),
      );
      return onSnapshot(
        q,
        (snap) => {
          snap.docChanges().forEach((change) => {
            if (change.type !== "added") return;
            const data = change.doc.data();
            onAdded({
              id: change.doc.id,
              senderUid: data.senderUid,
              ciphertext: data.ciphertext,
              nonce: data.nonce,
            });
          });
        },
        () => {
          /* permission/transient errors — UI shows the status separately */
        },
      );
    },

    // Live activity feed: the nudges / reminders / SOS alerts Siri writes. Merges
    // three collections, newest first. Returns an unsubscribe.
    subscribeActivity(gid, onItems) {
      const kinds = ["nudges", "reminders", "alerts"];
      const buckets = { nudges: [], reminders: [], alerts: [] };
      const emit = () => {
        const merged = [
          ...buckets.nudges,
          ...buckets.reminders,
          ...buckets.alerts,
        ]
          .sort((a, b) => (b.createdAt || "").localeCompare(a.createdAt || ""))
          .slice(0, 25);
        onItems(merged);
      };
      const unsubs = kinds.map((k) =>
        onSnapshot(
          query(
            collection(db, "groups", gid, k),
            orderBy("createdAt", "desc"),
            limit(20),
          ),
          (snap) => {
            buckets[k] = snap.docs.map((d) => ({ id: d.id, group: k, ...d.data() }));
            emit();
          },
          () => {},
        ),
      );
      return () => unsubs.forEach((u) => u());
    },
  };
}
