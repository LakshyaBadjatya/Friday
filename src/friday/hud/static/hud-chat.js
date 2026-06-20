// Circle chat mirror for the HUD — read-only live view. Activates ONLY when the web
// app casts a group via the URL fragment: #chat=<groupId>&k=<base64 key>&token=<ID
// token>. Reads the same end-to-end-encrypted Firestore chat directly over the
// Firestore REST API (the cast token goes in the Authorization HEADER, never a URL;
// the security rules enforce membership), decrypts with the cast key, and polls for
// new messages. No Firebase SDK, no separate sign-in. A module so its scope is its
// own. Sending stays in the web app.

const PROJECT = "lakufriday";
const REST = `https://firestore.googleapis.com/v1/projects/${PROJECT}/databases/(default)/documents`;

const params = new URLSearchParams(location.hash.replace(/^#/, ""));
const groupId = params.get("chat");
const keyB64 = params.get("k");
const token = params.get("token");

if (groupId && keyB64 && token) {
  // Scrub the key + token from the address bar / this history entry immediately.
  try {
    history.replaceState(null, "", location.pathname + location.search);
  } catch {
    /* non-fatal */
  }
  boot().catch(() => {});
}

async function boot() {
  const widget = document.getElementById("cc-widget");
  const msgs = document.getElementById("cc-msgs");
  if (!widget || !msgs) return;

  // Read-only mirror: hide the composer (sending stays in the web app).
  const foot = document.getElementById("cc-foot");
  if (foot) foot.style.display = "none";

  widget.classList.add("live");
  document
    .getElementById("cc-close")
    .addEventListener("click", () => widget.classList.remove("live"));

  const key = await importKey(keyB64);
  const me = decodeUid(token);
  const seen = new Set();
  let alive = true;

  async function poll() {
    if (!alive) return;
    try {
      const url = `${REST}/groups/${groupId}/messages?orderBy=createdAt&pageSize=200`;
      const res = await fetch(url, {
        headers: { Authorization: "Bearer " + token },
      });
      if (res.status === 401 || res.status === 403) {
        alive = false; // token expired / not a member — stop quietly
        return;
      }
      if (res.ok) {
        const body = await res.json();
        for (const docu of body.documents || []) await render(docu);
      }
    } catch {
      /* transient network error — try again next tick */
    }
    if (alive) setTimeout(poll, 2500);
  }

  async function render(docu) {
    const id = (docu.name || "").split("/").pop();
    if (!id || seen.has(id)) return;
    seen.add(id);
    const f = docu.fields || {};
    const ciphertext = (f.ciphertext || {}).stringValue || "";
    const nonce = (f.nonce || {}).stringValue || "";
    const senderUid = (f.senderUid || {}).stringValue || "";
    const text = await decrypt(key, ciphertext, nonce);
    const row = document.createElement("div");
    row.className = "cc-row" + (senderUid === me ? " cc-mine" : "");
    const bubble = document.createElement("div");
    bubble.className = "cc-b";
    bubble.textContent = text === null ? "🔒" : text;
    row.appendChild(bubble);
    msgs.appendChild(row);
    msgs.scrollTop = msgs.scrollHeight;
  }

  poll();
}

// --- minimal AES-GCM decrypt + base64 (same scheme as web/crypto.js) ---
function b64ToBytes(b64) {
  const s = atob(b64);
  const a = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
  return a;
}

function importKey(b64) {
  return crypto.subtle.importKey("raw", b64ToBytes(b64), { name: "AES-GCM" }, true, [
    "decrypt",
  ]);
}

async function decrypt(key, ciphertext, nonce) {
  try {
    const pt = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: b64ToBytes(nonce) },
      key,
      b64ToBytes(ciphertext),
    );
    return new TextDecoder().decode(pt);
  } catch {
    return null;
  }
}

// Decode (not verify) the ID token's payload just to label which bubbles are "mine".
function decodeUid(jwt) {
  try {
    const part = jwt.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const payload = JSON.parse(atob(part));
    return payload.user_id || payload.sub || payload.uid || "";
  } catch {
    return "";
  }
}
