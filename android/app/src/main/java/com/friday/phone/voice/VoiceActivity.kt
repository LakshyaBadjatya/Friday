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
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import com.friday.phone.ui.EdgeGlowView
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.friday.phone.ui.FridayTheme
import com.friday.phone.wake.WakeService
import kotlin.math.abs
import kotlin.math.sin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
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
        // If the wake word opened this screen it stood down to free the
        // microphone; leaving it stood down would mean her name works once.
        WakeService.resumeListening()
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
        var listening by remember { mutableStateOf(false) }
        var speaking by remember { mutableStateOf(false) }
        // Held rather than recreated so the recorder has something to push
        // levels into while the turn is running.
        val edge = remember { EdgeGlowView(ctx) }

        val ask = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission(),
        ) { ok -> granted = ok }

        // While she talks the microphone is closed, so there is no real level
        // to follow. A cadence near the speed of speech says the same thing —
        // that the voice you are hearing is hers — and costs nothing.
        LaunchedEffect(speaking) {
            var t = 0f
            while (speaking) {
                t += 0.16f
                edge.amplitude = 0.35f + 0.45f * abs(sin(t * 3f))
                delay(60)
            }
            edge.amplitude = 0f
        }

        // The same light the assist overlay uses, so a spoken turn looks the
        // same wherever it starts from.
        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { edge },
                update = { view ->
                    // Speaking is checked before busy: playback starts inside
                    // the request, so the turn is still "busy" while she talks.
                    view.state = when {
                        listening -> EdgeGlowView.State.LISTENING
                        speaking -> EdgeGlowView.State.SPEAKING
                        busy -> EdgeGlowView.State.THINKING
                        else -> EdgeGlowView.State.IDLE
                    }
                },
                modifier = Modifier.fillMaxSize(),
            )

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
                    listening = true
                    speaking = false
                    status = "Listening…"
                    heard = ""; said = ""
                    client.onSpeechStart = { speaking = true }
                    client.onSpeechEnd = { speaking = false; edge.amplitude = 0f }
                    scope.launch {
                        // Record and post off the main thread; the recorder
                        // blocks until it hears the sentence end.
                        val result = withContext(Dispatchers.IO) {
                            val recorder = VoiceRecorder()
                            // The edge answers the room, not a timer.
                            recorder.onLevel = { level -> edge.amplitude = level }
                            val wav = recorder.record()
                            listening = false
                            edge.amplitude = 0f
                            if (wav == null) {
                                VoiceClient.Result.Unheard
                            } else {
                                status = "Thinking…"
                                client.ask(wav)
                            }
                        }
                        when (result) {
                            is VoiceClient.Result.Spoke -> {
                                status = ""
                                heard = result.turn.transcript
                                said = result.turn.text
                            }
                            VoiceClient.Result.Unheard ->
                                status = "I didn't catch that."
                            is VoiceClient.Result.Failed ->
                                status = result.message
                        }
                        busy = false
                        listening = false
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
}
