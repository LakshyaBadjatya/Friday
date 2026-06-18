// End-to-end encryption for circle chat — AES-GCM 256 via the Web Crypto API.
//
// The group key is generated in the browser and NEVER sent to the server: it is
// shared with a friend only inside an invite link's URL #fragment (which browsers
// don't transmit in HTTP requests) and cached locally. The backend only ever sees
// the ciphertext + nonce produced here, so it cannot read messages.

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function bytesToB64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function b64ToBytes(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export async function generateKey() {
  return crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, [
    "encrypt",
    "decrypt",
  ]);
}

export async function exportKeyB64(key) {
  return bytesToB64(await crypto.subtle.exportKey("raw", key));
}

export async function importKeyB64(b64) {
  return crypto.subtle.importKey("raw", b64ToBytes(b64), { name: "AES-GCM" }, true, [
    "encrypt",
    "decrypt",
  ]);
}

export async function encrypt(key, plaintext) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const sealed = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    encoder.encode(plaintext),
  );
  return { ciphertext: bytesToB64(sealed), nonce: bytesToB64(iv) };
}

export async function decrypt(key, ciphertext, nonce) {
  try {
    const plain = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: b64ToBytes(nonce) },
      key,
      b64ToBytes(ciphertext),
    );
    return decoder.decode(plain);
  } catch {
    return null; // wrong key / tampered ciphertext — show as undecryptable
  }
}

// Per-group key cache (base64 raw key) so a returning member keeps reading without
// re-opening the invite link.
const KEY_PREFIX = "friday-circle-key:";

export function saveGroupKeyB64(groupId, b64) {
  localStorage.setItem(KEY_PREFIX + groupId, b64);
}

export function loadGroupKeyB64(groupId) {
  return localStorage.getItem(KEY_PREFIX + groupId);
}
