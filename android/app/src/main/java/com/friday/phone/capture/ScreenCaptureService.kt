package com.friday.phone.capture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.view.WindowManager
import com.friday.phone.sync.Filing
import java.io.File

/**
 * Grabs exactly one frame of the screen, files it, and stops.
 *
 * A foreground service because that is what MediaProjection requires, and a
 * short-lived one because FRIDAY wants a screenshot, not a recording: the
 * projection is torn down the moment the frame is on disk, so nothing is left
 * holding the right to watch the screen.
 */
class ScreenCaptureService : Service() {

    private var projection: MediaProjection? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, notification())

        val code = intent?.getIntExtra(EXTRA_CODE, 0) ?: 0
        val data: Intent? = intent?.getParcelableExtra(EXTRA_INTENT)
        if (data == null) {
            stopSelf()
            return START_NOT_STICKY
        }

        val manager = getSystemService(MediaProjectionManager::class.java)
        projection = manager.getMediaProjection(code, data)
        grabOneFrame()
        return START_NOT_STICKY
    }

    private fun grabOneFrame() {
        val metrics = DisplayMetrics().also {
            @Suppress("DEPRECATION")
            getSystemService(WindowManager::class.java).defaultDisplay.getRealMetrics(it)
        }
        val reader = ImageReader.newInstance(
            metrics.widthPixels, metrics.heightPixels, PixelFormat.RGBA_8888, 2,
        )
        val display = projection?.createVirtualDisplay(
            "friday-screen",
            metrics.widthPixels,
            metrics.heightPixels,
            metrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface,
            null,
            null,
        )

        reader.setOnImageAvailableListener({ source ->
            val image = source.acquireLatestImage() ?: return@setOnImageAvailableListener
            val plane = image.planes[0]
            val rowPadding = plane.rowStride - plane.pixelStride * metrics.widthPixels
            val bitmap = Bitmap.createBitmap(
                metrics.widthPixels + rowPadding / plane.pixelStride,
                metrics.heightPixels,
                Bitmap.Config.ARGB_8888,
            ).also { it.copyPixelsFromBuffer(plane.buffer) }
            image.close()

            val cropped = Bitmap.createBitmap(
                bitmap, 0, 0, metrics.widthPixels, metrics.heightPixels,
            )
            val target = File(cacheDir, "screen-${System.currentTimeMillis()}.jpg")
            target.outputStream().use { cropped.compress(Bitmap.CompressFormat.JPEG, 90, it) }
            Filing.file(applicationContext, target.absolutePath, "screen")

            source.setOnImageAvailableListener(null, null)
            display?.release()
            reader.close()
            projection?.stop()
            stopSelf()
        }, Handler(Looper.getMainLooper()))
    }

    private fun notification(): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL, "FRIDAY capture", NotificationManager.IMPORTANCE_LOW),
            )
        }
        return Notification.Builder(this, CHANNEL)
            .setContentTitle("FRIDAY is filing your screen")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .build()
    }

    override fun onDestroy() {
        projection?.stop()
        super.onDestroy()
    }

    companion object {
        const val EXTRA_CODE = "code"
        const val EXTRA_INTENT = "intent"
        private const val CHANNEL = "friday-capture"
        private const val NOTIFICATION_ID = 42
    }
}
