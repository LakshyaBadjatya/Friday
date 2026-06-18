// No-build Firebase web app (ES modules from the CDN, matching the studio/HUD
// frontends). Firebase Authentication (Email/Password + Google) + Firestore.
//
// Enable the providers once in the Firebase Console:
//   Authentication -> Sign-in method -> enable "Email/Password" and "Google".

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.5/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
  signOut,
  onAuthStateChanged,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-auth.js";
import {
  getFirestore,
  doc,
  setDoc,
  getDoc,
  serverTimestamp,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-firestore.js";
import {
  getAnalytics,
  isSupported as analyticsSupported,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-analytics.js";
import { firebaseConfig } from "./firebase-config.js";
import { initCircle } from "./circle.js";

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const googleProvider = new GoogleAuthProvider();

// Wire the circle UI (groups, invites, E2EE chat). Listeners attach once here; the
// group list is refreshed each time a user signs in (below). Client-direct Firestore.
const circle = initCircle(auth, db);

// Firebase Analytics — initialise only where the browser supports it (needs a
// measurementId and a supporting environment); never let it break the app.
analyticsSupported()
  .then((ok) => {
    if (ok) getAnalytics(app);
  })
  .catch(() => {});

const $ = (id) => document.getElementById(id);

function setStatus(message, isError = false) {
  const el = $("status");
  if (!el) return;
  el.textContent = message;
  el.className = isError ? "err" : "ok";
}

// Map the common auth error codes to friendly text.
function friendly(err) {
  const code = (err && err.code) || "";
  const table = {
    "auth/invalid-email": "That email doesn't look right.",
    "auth/missing-password": "Enter a password.",
    "auth/weak-password": "Password should be at least 6 characters.",
    "auth/email-already-in-use": "That email is already registered — try signing in.",
    "auth/invalid-credential": "Wrong email or password.",
    "auth/popup-closed-by-user": "Sign-in window closed before finishing.",
    "auth/operation-not-allowed":
      "This sign-in method isn't enabled yet (enable it in the Firebase Console).",
  };
  return table[code] || (err && err.message) || "Something went wrong.";
}

// Upsert a minimal profile document for the signed-in user.
async function writeProfile(user) {
  await setDoc(
    doc(db, "users", user.uid),
    {
      uid: user.uid,
      email: user.email,
      displayName: user.displayName || null,
      lastSeen: serverTimestamp(),
    },
    { merge: true },
  );
}

async function readProfile(uid) {
  const snap = await getDoc(doc(db, "users", uid));
  return snap.exists() ? snap.data() : null;
}

// ── Auth actions ─────────────────────────────────────────────────────────────
async function registerWithEmail() {
  try {
    const cred = await createUserWithEmailAndPassword(
      auth,
      $("email").value.trim(),
      $("password").value,
    );
    await writeProfile(cred.user);
    setStatus(`Account created for ${cred.user.email}.`);
  } catch (err) {
    setStatus(friendly(err), true);
  }
}

async function loginWithEmail() {
  try {
    const cred = await signInWithEmailAndPassword(
      auth,
      $("email").value.trim(),
      $("password").value,
    );
    setStatus(`Signed in as ${cred.user.email}.`);
  } catch (err) {
    setStatus(friendly(err), true);
  }
}

async function loginWithGoogle() {
  try {
    const cred = await signInWithPopup(auth, googleProvider);
    await writeProfile(cred.user);
    setStatus(`Signed in as ${cred.user.email}.`);
  } catch (err) {
    setStatus(friendly(err), true);
  }
}

async function logout() {
  await signOut(auth);
  setStatus("Signed out.");
}

// ── Auth state → UI ──────────────────────────────────────────────────────────
onAuthStateChanged(auth, async (user) => {
  const signedOut = $("signed-out");
  const signedIn = $("signed-in");
  if (user) {
    signedOut.hidden = true;
    signedIn.hidden = false;
    $("who").textContent = user.displayName || user.email || user.uid;
    try {
      const profile = await readProfile(user.uid);
      $("profile").textContent = profile
        ? `Firestore profile: ${JSON.stringify({ email: profile.email })}`
        : "No Firestore profile yet.";
    } catch (err) {
      $("profile").textContent = `Firestore read failed: ${friendly(err)}`;
    }
    circle.refreshGroups();
  } else {
    signedOut.hidden = false;
    signedIn.hidden = true;
  }
});

// ── Wire buttons ─────────────────────────────────────────────────────────────
$("register").addEventListener("click", registerWithEmail);
$("login").addEventListener("click", loginWithEmail);
$("google").addEventListener("click", loginWithGoogle);
$("logout").addEventListener("click", logout);
