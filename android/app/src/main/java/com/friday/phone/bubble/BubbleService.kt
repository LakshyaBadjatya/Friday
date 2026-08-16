package com.friday.phone.bubble

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Build
import android.os.IBinder
import android.view.Gravity
import android.view.MotionEvent
import android.view.WindowManager
import android.widget.ImageView
import androidx.core.content.ContextCompat
import com.friday.phone.capture.ScreenCaptureActivity
import kotlin.math.abs

/**
 * A small always-there FRIDAY button, floating over whatever is on screen.
 *
 * The tile needs the shade pulled down and the assistant needs a gesture the
 * launcher may have taken; the bubble is the one entry point that is visible
 * from inside another app without any of that. Tapping it files the screen.
 *
 * Draggable, because a fixed overlay eventually sits on top of the one control
 * the user needs, and the fix for that cannot be "uninstall the app".
 */
class BubbleService : Service() {

    private var windows: WindowManager? = null
    private var bubble: ImageView? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, notification())
        if (bubble != null) return START_STICKY

        val wm = getSystemService(WindowManager::class.java)
        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 0
            y = 300
        }

        val view = ImageView(this).apply {
            setImageResource(android.R.drawable.ic_menu_camera)
            setBackgroundResource(android.R.drawable.dialog_holo_dark_frame)
            setPadding(24, 24, 24, 24)
        }

        var downX = 0; var downY = 0
        var startX = 0f; var startY = 0f
        view.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    downX = params.x; downY = params.y
                    startX = event.rawX; startY = event.rawY
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    params.x = downX + (event.rawX - startX).toInt()
                    params.y = downY + (event.rawY - startY).toInt()
                    wm.updateViewLayout(view, params)
                    true
                }
                MotionEvent.ACTION_UP -> {
                    // A drag is not a tap. Without this the bubble fires a
                    // capture every time it is moved out of the way.
                    val moved = abs(event.rawX - startX) > 12 || abs(event.rawY - startY) > 12
                    if (!moved) {
                        startActivity(
                            Intent(this, ScreenCaptureActivity::class.java)
                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                        )
                    }
                    true
                }
                else -> false
            }
        }

        wm.addView(view, params)
        windows = wm
        bubble = view
        return START_STICKY
    }

    private fun notification(): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL, "FRIDAY bubble", NotificationManager.IMPORTANCE_MIN),
            )
        }
        return Notification.Builder(this, CHANNEL)
            .setContentTitle("FRIDAY is on screen")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .build()
    }

    override fun onDestroy() {
        bubble?.let { runCatching { windows?.removeView(it) } }
        bubble = null
        super.onDestroy()
    }

    companion object {
        private const val CHANNEL = "friday-bubble"
        private const val NOTIFICATION_ID = 43

        fun start(ctx: android.content.Context) {
            ContextCompat.startForegroundService(ctx, Intent(ctx, BubbleService::class.java))
        }

        fun stop(ctx: android.content.Context) {
            ctx.stopService(Intent(ctx, BubbleService::class.java))
        }
    }
}
