package com.friday.tv

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import okhttp3.WebSocket

/** Foreground service: holds the /tv/stream WebSocket and runs pushed actions. */
class ReceiverService : Service() {
    private var ws: WebSocket? = null
    private lateinit var runner: ActionRunner

    override fun onCreate() {
        super.onCreate()
        runner = ActionRunner(this)
        startForeground(1, buildNotification())
        val config = TvConfig(this)
        if (config.isConfigured && config.deviceId.isNotEmpty()) {
            ws = Api(config).openStream(config.deviceId) { action -> runner.run(action) }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY
    override fun onBind(intent: Intent?): IBinder? = null
    override fun onDestroy() { ws?.close(1000, "bye"); super.onDestroy() }

    private fun buildNotification(): Notification {
        val channelId = "friday_tv"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(channelId, "FRIDAY", NotificationManager.IMPORTANCE_MIN))
        }
        return Notification.Builder(this, channelId)
            .setContentTitle("FRIDAY is listening for TV commands")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .build()
    }

    companion object {
        fun start(context: Context) {
            val intent = Intent(context, ReceiverService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                context.startForegroundService(intent) else context.startService(intent)
        }
    }
}
