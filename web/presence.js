// Auto presence + precise location. Every member of a circle publishes their own
// live location/presence into their member doc (which only circle members can read,
// per firestore.rules) so the app, the HUD, and Siri all see one source of truth —
// "she's in New York, 9pm her time, active 2 min ago". Writes are to your OWN member
// doc only, so the existing security rules already permit them.

import {
  collection,
  doc,
  getDocs,
  updateDoc,
  serverTimestamp,
} from "https://www.gstatic.com/firebasejs/10.12.5/firebase-firestore.js";

const REFRESH_MS = 5 * 60 * 1000; // republish every 5 minutes while open

function currentPosition() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) return resolve(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve(pos),
      () => resolve(null), // permission denied / unavailable → presence without location
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
    );
  });
}

export function startPresence(db, auth) {
  async function myGroupIds(uid) {
    const snap = await getDocs(collection(db, "users", uid, "memberships"));
    return snap.docs.map((d) => d.id); // membership doc id == group id
  }

  async function publish(presence) {
    const user = auth.currentUser;
    if (!user) return;
    let gids = [];
    try {
      gids = await myGroupIds(user.uid);
    } catch {
      return;
    }
    if (!gids.length) return;

    const update = {
      presence,
      lastSeen: serverTimestamp(),
      tz: Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
    };
    if (presence === "active") {
      const pos = await currentPosition();
      if (pos) {
        update.location = {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
        };
        update.locationUpdatedAt = serverTimestamp();
      }
    }
    await Promise.all(
      gids.map((gid) =>
        updateDoc(doc(db, "groups", gid, "members", user.uid), update).catch(
          () => {},
        ),
      ),
    );
  }

  const timer = setInterval(() => publish("active"), REFRESH_MS);
  document.addEventListener("visibilitychange", () => {
    publish(document.hidden ? "away" : "active");
  });

  publish("active"); // kick off immediately on sign-in
  return { stop: () => clearInterval(timer), publish };
}
