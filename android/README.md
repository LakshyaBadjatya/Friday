# friday-phone (Android skeleton)

This is the phone-side companion app for FRIDAY. Task 16 scope only: a Gradle
project that compiles and launches a blank Compose screen. Camera, screen
capture, the assistant role, and wake word come later.

Target device: Xiaomi / HyperOS, Android 15 era. `minSdk = 29`, `compileSdk`/`targetSdk = 35`.

## Toolchain that was installed and verified to build

- JDK: Eclipse Temurin 17.0.20+8 (`java -version` reports
  `Temurin-17.0.20+8`). No system JDK was present on the build host and no
  passwordless root was available, so this was installed user-local under
  `~/jdk/jdk-17.0.20+8` rather than via apt.
- Android command-line tools: `commandlinetools-linux-11076708_latest.zip`
  (the URL/version in the task brief was still current), installed at
  `~/android-sdk/cmdline-tools/latest`.
- SDK packages installed via `sdkmanager --sdk_root=~/android-sdk`:
  - `platform-tools` (r37.0.1)
  - `platforms;android-35`
  - `build-tools;35.0.0`
- Gradle: 8.10, via the wrapper (`gradle/wrapper/gradle-wrapper.properties`).
  No system Gradle was present, so the wrapper itself was bootstrapped from a
  one-off Gradle 8.10 distribution zip downloaded to `~/gradle-bootstrap`
  (not part of this repo).
- AGP 8.7.3, Kotlin 2.0.20, `org.jetbrains.kotlin.plugin.compose` 2.0.20,
  KSP 2.0.20-1.0.25, Compose BOM 2024.09.03.

## Build

From `android/`:

```bash
export JAVA_HOME=~/jdk/jdk-17.0.20+8   # or any JDK 17+
export PATH="$JAVA_HOME/bin:$PATH"
./gradlew assembleDebug
```

`local.properties` must point `sdk.dir` at your Android SDK root; it is not
committed (see `android/.gitignore`) because it is machine-specific.

Output APK: `android/app/build/outputs/apk/debug/app-debug.apk`.

## Install on device over adb

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

## HyperOS notes (for later tasks, camera/background work)

Xiaomi's HyperOS aggressively kills background apps beyond stock Android's
behavior. Once this app needs to run a foreground service (screen capture,
background listening, etc.) the following will very likely be required and
are worth setting up early when testing on a real device:

- **Autostart**: Security app -> Permissions -> Autostart -> enable for
  FRIDAY. Without this, HyperOS may not restart the app's services after
  reboot or after being swiped from recents.
- **Battery saver**: Settings -> Apps -> Manage apps -> FRIDAY -> Battery
  saver -> set to "No restrictions" (unrestricted), not just "Save battery".
- **Display pop-up windows while running in the background**: Settings ->
  Apps -> Permissions (or the per-app permission page) -> enable "Display
  pop-up windows while running in the background" — HyperOS gates
  `SYSTEM_ALERT_WINDOW`-style overlays and some notification behavior behind
  this separately from the standard Android overlay permission.
- **Lock the app in Recents**: swiping up on the app card in Recents and
  tapping the lock icon can further reduce the odds of HyperOS killing it.
- These are user-toggled settings, not something the app can force-enable;
  the app should detect degraded background behavior and prompt the user to
  open the relevant settings page.
