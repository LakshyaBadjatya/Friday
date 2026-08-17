package com.friday.phone.wake

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import androidx.core.content.ContextCompat
import com.friday.phone.assistant.FridayInteractionService
import com.friday.phone.voice.VoiceActivity
import kotlin.concurrent.thread

/**
 * Listens for "Hey FRIDAY" and opens the talk screen when it hears it.
 *
 * A foreground service with a visible notification, because something holding
 * the microphone open should be impossible to miss — and because Android will
 * kill it otherwise.
 *
 * Nothing leaves the phone. Frames are scored locally and thrown away; only a
 * detection causes anything to happen, and what happens is a screen opening,
 * not audio being sent. A wake word that streamed every frame to a server to
 * be scored would be a bug, not a feature.
 */
class WakeService : Service() {

    private var detector: WakeWordDetector? = null

    override fun onBind(intent: Intent?): IBinder? = null

    @SuppressLint("MissingPermission") // started only after RECORD_AUDIO is granted
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (running) return START_STICKY
        startForeground(NOTIFICATION_ID, notification())
        running = true

        thread(name = "friday-wake") {
            val det = WakeWordDetector(applicationContext).also { detector = it }
            val frame = ShortArray(FRAME)

            while (running) {
                // Hand the microphone over entirely while a turn is happening,
                // rather than holding it and hoping two recorders in one app
                // can share. It also stops her own reply, coming out of the
                // speaker a foot away, from being scored as someone saying her
                // name again.
                if (paused) {
                    Thread.sleep(POLL_MILLIS)
                    continue
                }

                val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
                val recorder = AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    SAMPLE_RATE, CHANNEL, ENCODING, maxOf(minBuffer, FRAME * 4),
                )
                det.reset()
                recorder.startRecording()
                try {
                    while (running && !paused) {
                        val read = recorder.read(frame, 0, frame.size)
                        if (read < frame.size) continue
                        val score = det.feed(frame)
                        if (score >= THRESHOLD) {
                            det.reset()   // one wake per utterance, not one per frame
                            paused = true
                            summon()
                        }
                    }
                } finally {
                    runCatching { recorder.stop() }
                    recorder.release()
                }
            }
        }
        return START_STICKY
    }

    /**
     * Bring her up, by the route the platform actually permits.
     *
     * Starting an activity from a background service is blocked outright from
     * Android 10 onwards, so the original `startActivity` here did nothing at
     * all — the word was heard and then silently dropped. When FRIDAY holds the
     * assistant slot she can show her own session instead, which is both
     * allowed and the same overlay the assist gesture opens. The activity
     * remains the fallback for when she does not hold the slot, where it will
     * work only if the overlay permission has been granted.
     */
    private fun summon() {
        if (FridayInteractionService.showSession()) return
        runCatching {
            startActivity(
                Intent(this, VoiceActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            )
        }
    }

    private fun notification(): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "FRIDAY listening", NotificationManager.IMPORTANCE_LOW),
            )
        }
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("FRIDAY is listening for her name")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .build()
    }

    override fun onDestroy() {
        running = false
        paused = false
        detector = null
        super.onDestroy()
    }

    companion object {
        private const val SAMPLE_RATE = 16_000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        private const val FRAME = 1280           // 80 ms, the frame the model scores
        private const val THRESHOLD = 0.5f       // the notebook's own reporting threshold
        private const val CHANNEL_ID = "friday-wake"
        private const val NOTIFICATION_ID = 44

        private const val PREFS = "friday"
        private const val KEY_ENABLED = "wake_enabled"
        private const val POLL_MILLIS = 200L

        @Volatile private var running = false

        /**
         * Whether the microphone has been handed to a turn in progress.
         *
         * Companion-scoped so the session can release it without holding a
         * reference to the service; there is only ever one wake loop.
         */
        @Volatile private var paused = false

        /** Give the microphone back after a turn ends. */
        fun resumeListening() {
            paused = false
        }

        fun start(ctx: Context) {
            setEnabled(ctx, true)
            ContextCompat.startForegroundService(ctx, Intent(ctx, WakeService::class.java))
        }

        fun stop(ctx: Context) {
            setEnabled(ctx, false)
            ctx.stopService(Intent(ctx, WakeService::class.java))
        }

        /**
         * Whether the user asked her to listen, remembered across restarts.
         *
         * Without this the choice lived only in the running process, so a
         * reboot — or MIUI deciding to reclaim the service — turned the wake
         * word off with nothing to say it had happened. "Hey FRIDAY" then does
         * nothing, and looks exactly like a wake word that does not work.
         */
        fun isEnabled(ctx: Context): Boolean =
            ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getBoolean(KEY_ENABLED, false)

        private fun setEnabled(ctx: Context, on: Boolean) {
            ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit().putBoolean(KEY_ENABLED, on).apply()
        }
    }
}
