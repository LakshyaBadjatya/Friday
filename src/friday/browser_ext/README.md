# FRIDAY Quick-Ask — browser extension

A tiny Manifest V3 extension that asks your **local** FRIDAY from the browser
toolbar. No build step, no dependencies, no secrets baked in — it just POSTs to
your running FRIDAY's `/chat` endpoint over the same machine.

## Load it (unpacked)

1. Start FRIDAY locally (`friday serve`, default `http://127.0.0.1:8000`).
2. Chrome/Edge → `chrome://extensions` → enable **Developer mode** →
   **Load unpacked** → select this `browser_ext/` folder.
   (Firefox: `about:debugging` → **This Firefox** → **Load Temporary Add-on** →
   pick `manifest.json`.)
3. Click the FRIDAY toolbar icon, type a question, **Send** (or ⌘/Ctrl-Enter).

The base URL is editable in the popup and remembered via `chrome.storage`. Adjust
`host_permissions` in `manifest.json` if you serve FRIDAY on a different host/port.
