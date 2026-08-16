package com.friday.phone.capture

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.friday.phone.sync.Filing
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.io.File
import kotlin.coroutines.resume
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext

/**
 * Photograph something and hand it to the queue.
 *
 * The camera stays open after a shot. A chapter is thirty pages and a whiteboard
 * is four photographs, so the flow that costs nothing is the one where the
 * shutter can be pressed again immediately; going back to a gallery between
 * pages would make the common case the slow one.
 *
 * Nothing is uploaded from here. Each shot lands in cacheDir, gets its text read
 * on-device, and becomes a queue row — the worker does the network, so a capture
 * in a basement with no signal is indistinguishable from one at a desk.
 */
class CameraActivity : ComponentActivity() {

    private var imageCapture: ImageCapture? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) { CaptureScreen() }
            }
        }
    }

    @Composable
    private fun CaptureScreen() {
        var granted by remember {
            mutableStateOf(
                ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED,
            )
        }
        var shots by remember { mutableStateOf(0) }
        var status by remember { mutableStateOf("") }

        val ask = androidx.activity.compose.rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission(),
        ) { ok -> granted = ok }

        if (!granted) {
            Column(
                modifier = Modifier.fillMaxSize().padding(24.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text("FRIDAY needs the camera to see what you are looking at.")
                Button(
                    onClick = { ask.launch(Manifest.permission.CAMERA) },
                    modifier = Modifier.padding(top = 16.dp),
                ) { Text("Allow camera") }
            }
            return
        }

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).also { view -> bindCamera(view) }
                },
                modifier = Modifier.fillMaxSize(),
            )
            Column(
                modifier = Modifier.fillMaxWidth().align(Alignment.BottomCenter).padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                if (status.isNotEmpty()) {
                    Text(status, modifier = Modifier.padding(bottom = 8.dp))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = {
                        status = "Filing…"
                        capture { text ->
                            shots += 1
                            status = if (text.isEmpty()) {
                                "$shots filed"
                            } else {
                                "$shots filed — \"${text.take(40)}…\""
                            }
                        }
                    }) { Text("Capture") }
                    Button(onClick = { finish() }) { Text("Done") }
                }
            }
        }
    }

    /** Wire preview + still capture to this activity's lifecycle. */
    private fun bindCamera(view: PreviewView) {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(view.surfaceProvider)
            }
            val capture = ImageCapture.Builder()
                // A photographed page is read, not admired: minimise latency so
                // the shutter is ready again before the next page is turned.
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture,
            )
            imageCapture = capture
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Take one photograph, read it on-device, and queue it.
     *
     * [onFiled] runs on the main thread with whatever text ML Kit found, which
     * is what the screen echoes back — proof she saw the page, before any
     * network has been touched.
     */
    private fun capture(onFiled: (String) -> Unit) {
        val capture = imageCapture ?: return
        val file = File(cacheDir, "cap-${System.currentTimeMillis()}.jpg")
        val output = ImageCapture.OutputFileOptions.Builder(file).build()

        capture.takePicture(
            output,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    onFiled("")
                }

                override fun onImageSaved(results: ImageCapture.OutputFileResults) {
                    lifecycleScope.launch {
                        val text = readText(Uri.fromFile(file))
                        withContext(Dispatchers.IO) {
                            Filing.file(applicationContext, file.absolutePath, "camera", text)
                        }
                        onFiled(text)
                    }
                }
            },
        )
    }

    /** ML Kit's on-device read. Empty string when it finds nothing legible. */
    private suspend fun readText(uri: Uri): String = suspendCancellableCoroutine { cont ->
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        runCatching { InputImage.fromFilePath(this, uri) }
            .onFailure { cont.resume("") }
            .onSuccess { image ->
                recognizer.process(image)
                    .addOnSuccessListener { cont.resume(it.text) }
                    // A failed read is not a failed capture: the backend runs its
                    // own OCR over the real bytes anyway, and this text only
                    // exists so the queue has something to show while offline.
                    .addOnFailureListener { cont.resume("") }
            }
    }
}
