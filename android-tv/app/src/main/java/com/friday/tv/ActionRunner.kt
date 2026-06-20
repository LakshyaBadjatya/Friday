package com.friday.tv

import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.net.Uri
import android.view.KeyEvent
import org.json.JSONObject

/** Turns a backend TVAction JSON into an Android intent / media key. */
class ActionRunner(private val context: Context) {

    fun run(action: JSONObject) {
        when (action.optString("type")) {
            "open_app" -> openApp(action.optString("app"))
            "play", "search" -> play(action.optString("app", "youtube"), action.optString("query"))
            "media", "navigate" -> sendKey(action.optString("key"))
        }
    }

    /** Launch an installed app whose label best matches [spokenName]. */
    private fun openApp(spokenName: String) {
        val pm = context.packageManager
        val leanback = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LEANBACK_LAUNCHER)
        val main = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
        val apps = pm.queryIntentActivities(leanback, 0) + pm.queryIntentActivities(main, 0)
        val target = spokenName.lowercase().trim()
        val best = apps.minByOrNull { ri ->
            val label = ri.loadLabel(pm).toString().lowercase()
            when {
                label == target -> 0
                label.startsWith(target) || target.startsWith(label) -> 1
                label.contains(target) || target.contains(label) -> 2
                else -> 99
            }
        }
        val pkg = best?.activityInfo?.packageName ?: return
        val launch = pm.getLeanbackLaunchIntentForPackage(pkg) ?: pm.getLaunchIntentForPackage(pkg)
        launch?.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)?.let { context.startActivity(it) }
    }

    /** Deep-link play/search in a known app, else open the app. */
    private fun play(app: String, query: String) {
        val a = app.lowercase()
        val intent = when {
            a.contains("youtube") -> Intent(Intent.ACTION_VIEW,
                Uri.parse("https://www.youtube.com/results?search_query=" + Uri.encode(query)))
            a.contains("netflix") -> Intent(Intent.ACTION_VIEW,
                Uri.parse("nflx://www.netflix.com/search?q=" + Uri.encode(query)))
            a.contains("spotify") -> Intent(Intent.ACTION_VIEW,
                Uri.parse("spotify:search:" + Uri.encode(query)))
            a.contains("prime") || a.contains("amazon") -> Intent(Intent.ACTION_VIEW,
                Uri.parse("https://www.primevideo.com/search?phrase=" + Uri.encode(query)))
            else -> { openApp(app); return }
        }
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        runCatching { context.startActivity(intent) }.onFailure { openApp(app) }
    }

    /** Dispatch a transport/navigation key as a global media/key event. */
    private fun sendKey(key: String) {
        if (key == "home") {
            context.startActivity(Intent(Intent.ACTION_MAIN)
                .addCategory(Intent.CATEGORY_HOME).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            return
        }
        val code = when (key) {
            "play_pause" -> KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE
            "stop" -> KeyEvent.KEYCODE_MEDIA_STOP
            "next" -> KeyEvent.KEYCODE_MEDIA_NEXT
            "previous" -> KeyEvent.KEYCODE_MEDIA_PREVIOUS
            "rewind" -> KeyEvent.KEYCODE_MEDIA_REWIND
            "fast_forward" -> KeyEvent.KEYCODE_MEDIA_FAST_FORWARD
            else -> return
        }
        val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        am.dispatchMediaKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, code))
        am.dispatchMediaKeyEvent(KeyEvent(KeyEvent.ACTION_UP, code))
    }
}
