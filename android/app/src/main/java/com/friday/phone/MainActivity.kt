package com.friday.phone

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.friday.phone.capture.CameraActivity
import com.friday.phone.sync.UploadWorker

/**
 * The app shell: point the phone at FRIDAY, then start photographing.
 *
 * Setup lives on the front screen rather than behind a menu because it is the
 * one thing that must happen before anything else works, and hiding it would
 * make a fresh install look broken instead of unconfigured.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { Surface(modifier = Modifier.fillMaxSize()) { Home() } } }
    }
}

@Composable
private fun Home() {
    val ctx = LocalContext.current
    var base by remember { mutableStateOf(Config.baseUrl(ctx)) }
    var token by remember { mutableStateOf(Config.token(ctx)) }
    var saved by remember { mutableStateOf("") }

    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("FRIDAY", style = MaterialTheme.typography.headlineMedium)

        OutlinedTextField(
            value = base,
            onValueChange = { base = it },
            label = { Text("Backend URL") },
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
                // Saving is also the moment to retry: anything that queued while
                // the app was pointed nowhere can go now.
                UploadWorker.enqueue(ctx)
                saved = "Pointed at ${Config.baseUrl(ctx)}"
            },
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Save") }

        Button(
            onClick = { ctx.startActivity(Intent(ctx, CameraActivity::class.java)) },
            enabled = Config.isConfigured(ctx),
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Capture") }

        if (saved.isNotEmpty()) Text(saved)
        if (!Config.isConfigured(ctx)) Text("Set a backend URL before capturing.")
    }
}
