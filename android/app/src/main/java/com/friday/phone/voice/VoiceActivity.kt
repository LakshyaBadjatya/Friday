package com.friday.phone.voice

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.friday.phone.ui.FridayTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Press, speak, listen. One turn at a time, with what she heard shown back. */
class VoiceActivity : ComponentActivity() {

    private val client by lazy { VoiceClient(applicationContext) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            FridayTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) { TalkScreen() }
            }
        }
    }

    /** Stop talking when the screen goes away, rather than into an empty room. */
    override fun onStop() {
        client.stop()
        super.onStop()
    }

    @Composable
    private fun TalkScreen() {
        val ctx = LocalContext.current
        val scope = rememberCoroutineScope()
        var granted by remember {
            mutableStateOf(
                ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) ==
                    PackageManager.PERMISSION_GRANTED,
            )
        }
        var status by remember { mutableStateOf("") }
        var heard by remember { mutableStateOf("") }
        var said by remember { mutableStateOf("") }
        var busy by remember { mutableStateOf(false) }

        val ask = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission(),
        ) { ok -> granted = ok }

        Column(
            modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                "TALK",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
            )

            if (!granted) {
                Text(
                    "FRIDAY needs the microphone to hear you.",
                    color = MaterialTheme.colorScheme.onBackground,
                )
                Button(
                    onClick = { ask.launch(Manifest.permission.RECORD_AUDIO) },
                    modifier = Modifier.fillMaxWidth(),
                ) { Text("Allow microphone") }
                return@Column
            }

            Button(
                enabled = !busy,
                onClick = {
                    busy = true
                    status = "Listening…"
                    heard = ""; said = ""
                    scope.launch {
                        // Record and post off the main thread; the recorder
                        // blocks until it hears the sentence end.
                        val turn = withContext(Dispatchers.IO) {
                            val wav = VoiceRecorder().record()
                            status = "Thinking…"
                            client.ask(wav)
                        }
                        if (turn == null) {
                            status = "No answer — check the backend URL and token."
                        } else {
                            status = ""
                            heard = turn.transcript
                            said = turn.text
                        }
                        busy = false
                    }
                },
                modifier = Modifier.fillMaxWidth(),
            ) { Text(if (busy) "…" else "Speak") }

            if (status.isNotEmpty()) {
                Text(status, color = MaterialTheme.colorScheme.tertiary)
            }
            if (heard.isNotEmpty()) {
                Text(
                    "You: $heard",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (said.isNotEmpty()) {
                Text("FRIDAY: $said", color = MaterialTheme.colorScheme.onBackground)
            }
        }
    }
}
