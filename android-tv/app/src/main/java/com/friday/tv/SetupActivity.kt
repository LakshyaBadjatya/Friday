package com.friday.tv

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

/** First-run setup: store URL + token, pair the device, request overlay, start the service. */
class SetupActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_setup)
        val config = TvConfig(this)
        val url = findViewById<EditText>(R.id.url).apply { setText(config.baseUrl) }
        val token = findViewById<EditText>(R.id.token).apply { setText(config.token) }
        val status = findViewById<TextView>(R.id.status)

        findViewById<Button>(R.id.save).setOnClickListener {
            config.baseUrl = url.text.toString()
            config.token = token.text.toString()
            requestOverlayPermission()
            status.text = "Pairing…"
            thread {
                val id = Api(config).pair(Build.MODEL ?: "TV")
                runOnUiThread {
                    if (id != null) {
                        config.deviceId = id
                        status.text = "Paired. Receiver starting."
                        ReceiverService.start(this)
                    } else {
                        status.text = "Pairing failed — check URL/token and that the backend " +
                            "is running with FRIDAY_ENABLE_TV=true."
                    }
                }
            }
        }
    }

    private fun requestOverlayPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
            startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")))
        }
    }
}
