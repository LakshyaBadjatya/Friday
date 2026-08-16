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

    @Volatile private var running = false
    private var detector: WakeWordDetector? = null

    override fun onBind(intent: Intent?): IBinder? = null

    @SuppressLint("MissingPermission") // started only after RECORD_AUDIO is granted
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (running) return START_STICKY
        startForeground(NOTIFICATION_ID, notification())
        running = true

        thread(name = "friday-wake") {
            val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE, CHANNEL, ENCODING, maxOf(minBuffer, FRAME * 4),
            )
            val det = WakeWordDetector(applicationContext).also { detector = it }
            val frame = ShortArray(FRAME)
            recorder.startRecording()
            try {
                while (running) {
                    val read = recorder.read(frame, 0, frame.size)
                    if (read < frame.size) continue
                    val score = det.feed(frame)
                    if (score >= THRESHOLD) {
                        det.reset()   // one wake per utterance, not one per frame
                        startActivity(
                            Intent(this, VoiceActivity::class.java)
                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                        )
                    }
                }
            } finally {
                recorder.stop()
                recorder.release()
            }
        }
        return START_STICKY
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

        fun start(ctx: Context) {
            ContextCompat.startForegroundService(ctx, Intent(ctx, WakeService::class.java))
        }

        fun stop(ctx: Context) {
            ctx.stopService(Intent(ctx, WakeService::class.java))
        }
    }
}
