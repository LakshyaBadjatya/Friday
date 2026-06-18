// Accept-invite page (keyless / client-direct Firestore). The invite link looks like:
//   /join.html#c=<code>&g=<groupId>&k=<base64 group key>
// The key is read from the #fragment (never sent to any server), cached locally so
// chat unlocks immediately, then the join is written straight to Firestore (governed
// by the security rules) — no backend involved.

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.5/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
  onAuthStateChanged,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.12.5/firebase-firestore.js";
import { firebaseConfig } from "./firebase-config.js";
import { createCircleData } from "./firestore-circle.js";
import { saveGroupKeyB64 } from "./crypto.js";

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const data = createCircleData(db);
const googleProvider = new GoogleAuthProvider();

const TZ = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
const $ = (id) => document.getElementById(id);

function setStatus(message, isError = false) {
  const el = $("status");
  el.textContent = message;
  el.className = isError ? "err" : "ok";
}

function parseInvite() {
  const params = new URLSearchParams(location.hash.replace(/^#/, ""));
  return {
    code: params.get("c"),
    groupId: params.get("g"),
    key: params.get("k"),
  };
}

const invite = parseInvite();
$("invite-info").textContent =
  invite.code && invite.groupId && invite.key
    ? "You've been invited to a private circle."
    : "This invite link is incomplete — ask your friend to resend it.";

async function preview() {
  if (!invite.code) return;
  try {
    const res = await data.previewInvite(invite.code);
    if (res) {
      $("grp").textContent = res.groupName;
      $("invite-info").textContent = `You've been invited to “${res.groupName}”.`;
    }
  } catch {
    /* preview needs a signed-in caller; it retries after sign-in */
  }
}

async function accept() {
  if (!invite.code || !invite.groupId || !invite.key) {
    return setStatus("Invite link is incomplete.", true);
  }
  try {
    // Cache the key locally first so chat is readable the moment we land.
    saveGroupKeyB64(invite.groupId, invite.key);
    const user = auth.currentUser;
    const name = (user && (user.displayName || user.email)) || "Friend";
    await data.acceptInvite(user.uid, invite.code, name, TZ);
    setStatus("Joined! Taking you to the chat…");
    location.href = "./index.html";
  } catch (err) {
    setStatus("Couldn't join: " + err.message, true);
  }
}

async function login() {
  try {
    await signInWithEmailAndPassword(auth, $("email").value.trim(), $("password").value);
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function register() {
  try {
    await createUserWithEmailAndPassword(
      auth,
      $("email").value.trim(),
      $("password").value,
    );
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function loginGoogle() {
  try {
    await signInWithPopup(auth, googleProvider);
  } catch (err) {
    setStatus(err.message, true);
  }
}

onAuthStateChanged(auth, (user) => {
  if (user) {
    $("join-signed-out").hidden = true;
    $("join-signed-in").hidden = false;
    $("who").textContent = user.displayName || user.email || user.uid;
    preview();
  } else {
    $("join-signed-out").hidden = false;
    $("join-signed-in").hidden = true;
  }
});

$("login").addEventListener("click", login);
$("register").addEventListener("click", register);
$("google").addEventListener("click", loginGoogle);
$("accept").addEventListener("click", accept);
preview();
