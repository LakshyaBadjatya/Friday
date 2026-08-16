package com.friday.phone.capture

import android.app.Activity
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat

/**
 * Asks for screen-capture consent, then gets out of the way.
 *
 * The grant Android hands back is a one-shot token that must be used by a
 * foreground service, not by an activity that is about to disappear — so this
 * screen exists only to collect it and pass it on.
 */
class ScreenCaptureActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val ask = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult(),
        ) { result ->
            if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                val service = Intent(this, ScreenCaptureService::class.java)
                    .putExtra(ScreenCaptureService.EXTRA_CODE, result.resultCode)
                    .putExtra(ScreenCaptureService.EXTRA_INTENT, result.data)
                ContextCompat.startForegroundService(this, service)
            } else {
                Toast.makeText(this, "Screen capture refused", Toast.LENGTH_SHORT).show()
            }
            finish()
        }

        val manager = getSystemService(MediaProjectionManager::class.java)
        ask.launch(manager.createScreenCaptureIntent())
    }
}
