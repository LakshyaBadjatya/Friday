// Copy to firebase-config.js and fill from the Firebase Console.
// NOTE: the web apiKey is public-by-design (it ships to the browser). Protect the
// project with Firestore Security Rules + API-key restrictions in Google Cloud
// Console (HTTP referrers limited to your domains), NOT by hiding this file.
export const firebaseConfig = {
  apiKey: "REPLACE_WITH_API_KEY",
  authDomain: "REPLACE_PROJECT_ID.firebaseapp.com",
  projectId: "REPLACE_WITH_PROJECT_ID",
  storageBucket: "REPLACE_PROJECT_ID.firebasestorage.app",
  messagingSenderId: "REPLACE_SENDER_ID",
  appId: "REPLACE_WITH_APP_ID",
};
