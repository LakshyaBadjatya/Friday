// Thin client for the FRIDAY circle backend (FastAPI on Render). All calls carry
// the Firebase ID token as a bearer header; the backend verifies it and persists
// to Firestore. The token provider is injected by the page (app.js / join.js) so
// this module stays decoupled from Firebase.

export const BACKEND = "https://friday-backend-oj8h.onrender.com";

let tokenProvider = async () => null;

export function setTokenProvider(fn) {
  tokenProvider = fn;
}

async function authed(path, opts = {}) {
  const token = await tokenProvider();
  const headers = Object.assign(
    { "Content-Type": "application/json" },
    opts.headers || {},
  );
  if (token) headers["Authorization"] = "Bearer " + token;
  const res = await fetch(BACKEND + path, Object.assign({}, opts, { headers }));
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).detail || "";
    } catch {
      detail = await res.text();
    }
    const err = new Error(detail || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  createGroup: (name, displayName, tz) =>
    authed("/circle/groups", {
      method: "POST",
      body: JSON.stringify({ name, display_name: displayName, tz }),
    }),
  myGroups: () => authed("/circle/groups"),
  members: (gid) => authed(`/circle/groups/${gid}/members`),
  createInvite: (gid) =>
    authed(`/circle/groups/${gid}/invites`, {
      method: "POST",
      body: JSON.stringify({}),
    }),
  acceptInvite: (code, displayName, tz) =>
    authed(`/circle/invites/${encodeURIComponent(code)}/accept`, {
      method: "POST",
      body: JSON.stringify({ display_name: displayName, tz }),
    }),
  previewInvite: (code) => authed(`/circle/invites/${encodeURIComponent(code)}`),
  postMessage: (gid, ciphertext, nonce) =>
    authed(`/circle/groups/${gid}/messages`, {
      method: "POST",
      body: JSON.stringify({ ciphertext, nonce }),
    }),
  messages: (gid) => authed(`/circle/groups/${gid}/messages`),
  // EventSource can't set headers and a token in the URL would leak into logs, so we
  // mint a single-use, short-lived ticket (authed POST) and open the stream with it.
  streamUrl: async (gid) => {
    const { ticket } = await authed(`/circle/groups/${gid}/stream/ticket`, {
      method: "POST",
      body: "{}",
    });
    return `${BACKEND}/circle/groups/${gid}/stream?ticket=${encodeURIComponent(ticket)}`;
  },
};
