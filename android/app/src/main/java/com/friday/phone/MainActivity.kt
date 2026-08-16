package com.friday.phone

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.friday.phone.capture.CameraActivity
import com.friday.phone.capture.ScanActivity
import com.friday.phone.sync.QueueDb
import com.friday.phone.sync.UploadWorker
import com.friday.phone.ui.FridayTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * The app shell: point the phone at FRIDAY, then start photographing.
 *
 * Setup lives on the front screen rather than behind a menu because it is the
 * one thing that must happen before anything else works, and hiding it would
 * make a fresh install look broken instead of unconfigured. Once she is
 * configured the setup collapses out of the way and the capture buttons take
 * the screen, because that is what gets used forty times a day.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            FridayTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) { Home() }
            }
        }
    }
}

@Composable
private fun Home() {
    val ctx = LocalContext.current
    var base by remember { mutableStateOf(Config.baseUrl(ctx)) }
    var token by remember { mutableStateOf(Config.token(ctx)) }
    var configured by remember { mutableStateOf(Config.isConfigured(ctx)) }
    var status by remember { mutableStateOf("") }
    var pending by remember { mutableStateOf(0) }

    // The queue depth is the one number that answers "did that work?" without a
    // network round trip, so it is read on every visit rather than on demand.
    LaunchedEffect(status) {
        pending = withContext(Dispatchers.IO) {
            runCatching { QueueDb.get(ctx).pending().count() }.getOrDefault(0)
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            "FRIDAY",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary,
        )
        Text(
            if (configured) "Linked to ${Config.baseUrl(ctx)}" else "Not linked yet",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface,
                contentColor = MaterialTheme.colorScheme.onSurface,
            ),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                OutlinedTextField(
                    value = base,
                    onValueChange = { base = it },
                    label = { Text("Backend URL") },
                    placeholder = { Text("http://192.168.1.32:8000") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = token,
                    onValueChange = { token = it },
                    label = { Text("Token") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Button(
                    onClick = {
                        Config.setBaseUrl(ctx, base)
                        Config.setToken(ctx, token)
                        configured = Config.isConfigured(ctx)
                        // Saving is also the moment to retry: anything that
                        // queued while the app was pointed nowhere can go now.
                        UploadWorker.enqueue(ctx)
                        status = if (configured) "Linked" else "Enter a backend URL"
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) { Text("Save") }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                onClick = { ctx.startActivity(Intent(ctx, CameraActivity::class.java)) },
                enabled = configured,
                modifier = Modifier.weight(1f),
            ) { Text("Capture") }
            OutlinedButton(
                onClick = { ctx.startActivity(Intent(ctx, ScanActivity::class.java)) },
                enabled = configured,
                modifier = Modifier.weight(1f),
            ) { Text("Scan") }
        }

        Text(
            when {
                !configured -> "Set a backend URL before capturing."
                pending == 0 -> "Nothing waiting. Everything she has been shown is filed."
                pending == 1 -> "1 capture waiting for a network."
                else -> "$pending captures waiting for a network."
            },
            style = MaterialTheme.typography.bodyMedium,
            color = if (pending > 0) {
                MaterialTheme.colorScheme.tertiary
            } else {
                MaterialTheme.colorScheme.onSurfaceVariant
            },
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Start,
        )

        if (status.isNotEmpty()) {
            Text(
                status,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.secondary,
            )
        }
    }
}
