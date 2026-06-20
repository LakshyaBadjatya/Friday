# FRIDAY for Android TV (sideloaded)

A thin FRIDAY client for a Mi (Xiaomi) / Android TV. You open it (or the experimental
key), press **OK**, and talk through the remote mic — it answers aloud and controls the
TV (open any installed app, play/search YouTube/Netflix/Prime/Spotify, media keys). A
background receiver also runs commands spoken to your phone ("ask Friday to … on the TV").

It is a **thin client** over the backend `/tv` API — the AI is the existing FRIDAY brain.

## 1. Backend

Run FRIDAY with the TV surface on and auth required, reachable by the TV (same tunnel /
tailnet / LAN as Siri):

```bash
FRIDAY_ENABLE_TV=true \
FRIDAY_REQUIRE_AUTH=true \
FRIDAY_API_KEYS=<your-strong-token> \
friday serve --host 0.0.0.0 --port 8000
```

Smoke-test it:

```bash
curl -X POST "https://<your-host>/tv/ask?q=open%20youtube&format=json" \
     -H "Authorization: Bearer <your-token>"
# -> {"speak":"Opening youtube.","mode":"tv","action":{"type":"open_app","app":"youtube",...}}
```

## 2. Build the APK

Open the `android-tv/` folder in **Android Studio** (it generates the Gradle wrapper and
downloads the SDK), or with a local Gradle/SDK:

```bash
cd android-tv
./gradlew :app:assembleDebug
# APK -> app/build/outputs/apk/debug/app-debug.apk
```

## 3. Enable ADB on the Mi TV, then install

1. Settings → Device Preferences → **About** → click **Build** 7× (unlocks Developer options).
2. Developer options → enable **USB debugging** (and network ADB if present).
3. From your computer:

```bash
adb connect <tv-ip>:5555
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## 4. First run

Open **FRIDAY** from the TV home screen → enter the **Backend URL** and **Bearer token**
→ **Save, pair & start**. Grant the overlay permission when prompted. You should see
"Paired. Receiver starting."

## 5. Keep it alive on Xiaomi (important)

MIUI / PatchWall kill background apps aggressively. So the receiver survives:

- Settings → Apps → **FRIDAY** → **Autostart: ON**.
- Settings → Apps → FRIDAY → Battery → **No restrictions** (remove battery optimization).

## 6. Using it

- **On the TV:** open FRIDAY (or the experimental key, if it works) and press **OK**, then
  speak: *"open YouTube"*, *"play lofi beats"*, *"open Netflix"*, *"pause"*, *"go home"*.
- **Hands-free via phone:** say to your phone *"Hey Siri, ask Friday to play lofi beats on
  the TV"* — the TV obeys via the background receiver.

## 7. Experimental remote-key trigger

`KeyAccessibilityService` can open FRIDAY from a remote key, IF the device lets the key
reach an accessibility service (the Google Assistant/mic button usually will **not**).
Enable it under Settings → Accessibility → FRIDAY, find a free keycode with
`adb shell getevent -l`, set `triggerKey` in `KeyAccessibilityService.kt`, and rebuild.
If no key reaches it, use the launcher tile + the phone relay instead.

## Honest limits

- **No on-TV "Hey Friday" wake word.** The Mi TV's only mic is in the Bluetooth remote and
  streams to Google only while its button is held — there is no open mic for a wake engine.
- **The remote's Google mic button can't be reassigned** on a stock (non-rooted) device.
- **YouTube auto-play of the first result is best-effort** — the search-results deep link is
  the reliable floor.
- App icon/banner are placeholder vectors — swap in real artwork in `res/drawable/`.
