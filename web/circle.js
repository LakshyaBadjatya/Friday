// Circle UI: create a group, invite a friend (E2EE link), list members, and a live
// end-to-end-encrypted chat. Keyless/client-direct — talks to Firestore via
// firestore-circle.js; live chat uses Firestore's realtime listener (onSnapshot).
// The encryption key lives only in the browser + the invite link fragment; Firestore
// stores ciphertext only. Imported and started by app.js once Firebase is ready.

import { createCircleData } from "./firestore-circle.js";
import {
  generateKey,
  exportKeyB64,
  importKeyB64,
  encrypt,
  decrypt,
  saveGroupKeyB64,
  loadGroupKeyB64,
} from "./crypto.js";

// Where the HUD lives (for the "Open chat in HUD" cast).
const HUD_ORIGIN = "https://friday-backend-oj8h.onrender.com";
const TZ = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
const $ = (id) => document.getElementById(id);

export function initCircle(auth, db) {
  const data = createCircleData(db);
  const state = { groupId: null, name: null, key: null, unsub: null, seen: new Set() };

  const me = () => auth.currentUser;
  const myUid = () => (me() ? me().uid : null);
  const myName = () => {
    const u = me();
    return (u && (u.displayName || u.email)) || "Me";
  };

  function note(msg, isError = false) {
    const el = $("circle-status");
    if (!el) return;
    el.textContent = msg;
    el.className = isError ? "err" : "ok";
  }

  async function refreshGroups() {
    const uid = myUid();
    if (!uid) return;
    let groups = [];
    try {
      groups = await data.myGroups(uid);
    } catch (err) {
      note("Couldn't load your circle: " + err.message, true);
      return;
    }
    const list = $("groups-list");
    list.innerHTML = "";
    if (!groups.length) {
      list.innerHTML = "<p class='muted'>No groups yet — create one above.</p>";
    }
    for (const g of groups) {
      const btn = document.createElement("button");
      btn.className = "ghost group-pill";
      btn.textContent = g.groupName;
      btn.addEventListener("click", () => openGroup(g.groupId, g.groupName));
      list.appendChild(btn);
    }
  }

  async function createGroup() {
    const name = $("group-name").value.trim();
    if (!name) return note("Enter a group name.", true);
    try {
      const key = await generateKey();
      const keyB64 = await exportKeyB64(key);
      const group = await data.createGroup(myUid(), name, myName(), TZ);
      saveGroupKeyB64(group.id, keyB64);
      $("group-name").value = "";
      await refreshGroups();
      await openGroup(group.id, group.name);
      note(`Created “${group.name}”. Now invite your friend.`);
    } catch (err) {
      note("Create failed: " + err.message, true);
    }
  }

  async function openGroup(groupId, name) {
    if (state.unsub) {
      state.unsub();
      state.unsub = null;
    }
    state.groupId = groupId;
    state.name = name;
    state.seen = new Set();
    $("group-panel").hidden = false;
    $("group-title").textContent = name;
    $("invite-out").hidden = true;
    $("messages").innerHTML = "";

    const keyB64 = loadGroupKeyB64(groupId);
    state.key = keyB64 ? await importKeyB64(keyB64) : null;
    note(
      state.key
        ? ""
        : "No encryption key on this device for this group — open the invite link here to unlock chat.",
      !state.key,
    );

    await renderMembers(groupId);
    state.unsub = data.subscribeMessages(groupId, (msg) => {
      if (state.groupId === groupId) renderMessage(msg);
    });
  }

  function localTimeLabel(tz) {
    if (!tz) return "";
    try {
      const now = new Date();
      const time = new Intl.DateTimeFormat(undefined, {
        timeZone: tz,
        hour: "numeric",
        minute: "2-digit",
      }).format(now);
      const hour = Number(
        new Intl.DateTimeFormat("en-US", {
          timeZone: tz,
          hour: "numeric",
          hour12: false,
        }).format(now),
      );
      const asleep = hour >= 23 || hour < 7;
      return `${time} their time${asleep ? " · 😴 likely asleep" : ""}`;
    } catch {
      return "";
    }
  }

  async function renderMembers(groupId) {
    const box = $("members");
    box.innerHTML = "";
    try {
      const members = await data.members(groupId);
      for (const m of members) {
        const row = document.createElement("div");
        row.className = "member-row";
        const name = document.createElement("div");
        name.className = "member-name";
        const dot = m.presence === "active" ? "🟢" : "⚪";
        name.textContent = `${dot} ${m.displayName}${m.role === "admin" ? " ★" : ""}`;
        row.appendChild(name);
        const meta = document.createElement("div");
        meta.className = "member-meta";
        meta.textContent = localTimeLabel(m.tz);
        row.appendChild(meta);
        if (m.location && m.uid !== myUid()) {
          const link = document.createElement("a");
          link.className = "member-map";
          link.href = `https://www.google.com/maps?q=${m.location.lat},${m.location.lng}`;
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = "📍 map";
          row.appendChild(link);
        }
        box.appendChild(row);
      }
    } catch (err) {
      box.textContent = "Couldn't load members: " + err.message;
    }
  }

  async function renderMessage(msg) {
    if (state.seen.has(msg.id)) return;
    state.seen.add(msg.id);
    const text = state.key
      ? await decrypt(state.key, msg.ciphertext, msg.nonce)
      : null;
    const mine = msg.senderUid === myUid();
    const row = document.createElement("div");
    row.className = "msg" + (mine ? " mine" : "");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent =
      text === null ? "🔒 (can't decrypt — missing key)" : text;
    row.appendChild(bubble);
    const box = $("messages");
    box.appendChild(row);
    box.scrollTop = box.scrollHeight;
  }

  async function sendMessage() {
    const input = $("chat-input");
    const text = input.value.trim();
    if (!text || !state.groupId) return;
    if (!state.key) return note("No key on this device — can't encrypt.", true);
    try {
      const { ciphertext, nonce } = await encrypt(state.key, text);
      input.value = "";
      await data.sendMessage(myUid(), state.groupId, ciphertext, nonce);
      // The onSnapshot listener echoes it back (including to us), so no local render.
    } catch (err) {
      note("Send failed: " + err.message, true);
    }
  }

  async function inviteFriend() {
    if (!state.groupId) return;
    const keyB64 = loadGroupKeyB64(state.groupId);
    if (!keyB64) return note("No key for this group on this device.", true);
    try {
      const invite = await data.createInvite(myUid(), state.groupId, state.name);
      const link =
        `${location.origin}/join.html#c=${encodeURIComponent(invite.code)}` +
        `&g=${encodeURIComponent(state.groupId)}&k=${encodeURIComponent(keyB64)}`;
      $("invite-link").value = link;
      $("invite-out").hidden = false;
      note("Invite link ready — send it privately. The key is in the link only.");
    } catch (err) {
      note("Invite failed: " + err.message, true);
    }
  }

  function copyInvite() {
    const el = $("invite-link");
    el.select();
    navigator.clipboard?.writeText(el.value).then(
      () => note("Link copied."),
      () => note("Select the link and copy it manually.", true),
    );
  }

  // "Cast" the open group's live chat into the HUD. The group id, key, and a
  // short-lived ID token ride in the HUD URL's #fragment (never sent to a server;
  // the HUD scrubs them from its address bar on load) so the HUD can read the same
  // encrypted Firestore chat.
  async function openInHud() {
    if (!state.groupId) return;
    const keyB64 = loadGroupKeyB64(state.groupId);
    const u = me();
    if (!keyB64 || !u) return note("Open this group's chat here first.", true);
    const token = await u.getIdToken();
    const url =
      `${HUD_ORIGIN}/hud#chat=${encodeURIComponent(state.groupId)}` +
      `&k=${encodeURIComponent(keyB64)}&token=${encodeURIComponent(token)}`;
    window.open(url, "_blank", "noopener");
  }

  $("create-group").addEventListener("click", createGroup);
  $("invite-friend").addEventListener("click", inviteFriend);
  $("cast-hud").addEventListener("click", openInHud);
  $("copy-invite").addEventListener("click", copyInvite);
  $("chat-send").addEventListener("click", sendMessage);
  $("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  return { refreshGroups };
}
