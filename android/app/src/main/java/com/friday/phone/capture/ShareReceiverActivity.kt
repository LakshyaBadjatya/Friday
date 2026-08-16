package com.friday.phone.capture

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import com.friday.phone.sync.Filing
import java.io.File

/**
 * "Share to FRIDAY" from anywhere on the phone.
 *
 * Translucent and no-history: sharing should feel like the thing left the other
 * app, not like it opened a new one. The only visible evidence is a toast.
 */
class ShareReceiverActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val filed = runCatching { accept() }.getOrDefault(false)
        Toast.makeText(
            this,
            if (filed) "Filed with FRIDAY" else "Nothing to file",
            Toast.LENGTH_SHORT,
        ).show()
        finish()
    }

    private fun accept(): Boolean {
        if (intent?.action != Intent.ACTION_SEND) return false

        val uri: Uri? = intent.getParcelableExtra(Intent.EXTRA_STREAM)
        if (uri != null) {
            val stream = contentResolver.openInputStream(uri) ?: return false
            Filing.fileStream(applicationContext, stream, "share")
            return true
        }

        // Shared text has no bytes to upload, but it is still something she was
        // shown; it goes in as a text file so the same queue carries it.
        val text = intent.getStringExtra(Intent.EXTRA_TEXT) ?: return false
        if (text.isBlank()) return false
        val target = File(cacheDir, "share-${System.currentTimeMillis()}.txt")
        target.writeText(text)
        Filing.file(applicationContext, target.absolutePath, "share", offlineOcr = text)
        return true
    }
}
