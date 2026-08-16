package com.friday.phone.capture

import android.app.Activity
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import com.friday.phone.sync.Filing
import com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions
import com.google.mlkit.vision.documentscanner.GmsDocumentScanning
import com.google.mlkit.vision.documentscanner.GmsDocumentScanningResult

/**
 * The proper scan: crop, deskew, multiple pages.
 *
 * Play Services already ships this and does it better than a hand-rolled corner
 * detector would; a photographed page and a scanned one are different jobs, and
 * a textbook chapter wants the second.
 */
class ScanActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val result = registerForActivityResult(
            ActivityResultContracts.StartIntentSenderForResult(),
        ) { activityResult ->
            if (activityResult.resultCode == Activity.RESULT_OK) {
                file(GmsDocumentScanningResult.fromActivityResultIntent(activityResult.data))
            }
            finish()
        }

        val options = GmsDocumentScannerOptions.Builder()
            .setGalleryImportAllowed(true)
            .setPageLimit(30)
            .setResultFormats(GmsDocumentScannerOptions.RESULT_FORMAT_JPEG)
            .setScannerMode(GmsDocumentScannerOptions.SCANNER_MODE_FULL)
            .build()

        GmsDocumentScanning.getClient(options)
            .getStartScanIntent(this)
            .addOnSuccessListener { sender ->
                result.launch(IntentSenderRequest.Builder(sender).build())
            }
            .addOnFailureListener {
                Toast.makeText(this, "Scanner unavailable", Toast.LENGTH_SHORT).show()
                finish()
            }
    }

    /** One queue row per page: she files pages, not documents. */
    private fun file(scan: GmsDocumentScanningResult?) {
        val pages = scan?.pages.orEmpty()
        if (pages.isEmpty()) return
        pages.forEach { page ->
            val stream = contentResolver.openInputStream(page.imageUri) ?: return@forEach
            Filing.fileStream(applicationContext, stream, "scan")
        }
        Toast.makeText(this, "${pages.size} pages filed", Toast.LENGTH_SHORT).show()
    }
}
