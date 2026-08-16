package com.friday.phone.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.friday.phone.bubble.BubbleService

/**
 * The permissions HyperOS will not grant on its own.
 *
 * Every item is a thing an assistant needs and a stock launcher hands over
 * quietly, and that MIUI puts behind a switch three menus deep. Each row says
 * whether it is satisfied and opens the exact screen that fixes it, because
 * "enable autostart" is useless advice on a phone with four settings apps.
 *
 * Nothing here is requested silently. The app cannot grant any of it, and a
 * checklist the user drives is the honest shape for that.
 */
class SetupActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            FridayTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) { SetupScreen() }
            }
        }
    }

    @Composable
    private fun SetupScreen() {
        val ctx = LocalContext.current
        // Bumped on resume so every row re-checks itself after a trip to Settings.
        var round by remember { mutableIntStateOf(0) }
        var bubbleOn by remember { mutableStateOf(false) }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .safeDrawingPadding()
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                "SETUP",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                "HyperOS keeps these off by default. She works without them; " +
                    "she is only always-there with them.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Step(
                title = "Draw over other apps",
                why = "Lets the bubble sit on top of whatever you are reading.",
                satisfied = canOverlay(ctx),
                round = round,
                onFix = {
                    ctx.startActivity(
                        Intent(
                            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                            Uri.parse("package:${ctx.packageName}"),
                        ),
                    )
                },
            )

            Step(
                title = "Unrestricted battery",
                why = "Stops the queue being killed before a capture has uploaded.",
                satisfied = batteryUnrestricted(ctx),
                round = round,
                onFix = {
                    @Suppress("BatteryLife") // the whole point of the row
                    ctx.startActivity(
                        Intent(
                            Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                            Uri.parse("package:${ctx.packageName}"),
                        ),
                    )
                },
            )

            Step(
                title = "Default digital assistant",
                why = "Routes the assist gesture to FRIDAY, and hands her the screen.",
                satisfied = isDefaultAssistant(ctx),
                round = round,
                onFix = { ctx.startActivity(Intent(Settings.ACTION_VOICE_INPUT_SETTINGS)) },
            )

            Step(
                title = "Autostart",
                why = "Lets her come back after a reboot instead of waiting to be opened.",
                satisfied = null, // MIUI exposes no way to read this
                round = round,
                onFix = { openMiuiAutostart(ctx) },
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
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text("Floating bubble", color = MaterialTheme.colorScheme.onSurface)
                    Text(
                        "A FRIDAY button over every app. Tap it to file the screen.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        OutlinedButton(
                            enabled = canOverlay(ctx) && !bubbleOn,
                            onClick = { BubbleService.start(ctx); bubbleOn = true },
                        ) { Text("Show") }
                        OutlinedButton(
                            enabled = bubbleOn,
                            onClick = { BubbleService.stop(ctx); bubbleOn = false },
                        ) { Text("Hide") }
                    }
                }
            }

            OutlinedButton(
                onClick = { round += 1 },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Re-check") }
        }
    }

    @Composable
    private fun Step(
        title: String,
        why: String,
        satisfied: Boolean?,
        round: Int,
        onFix: () -> Unit,
    ) {
        val mark = when (satisfied) {
            true -> "✓"
            false -> "✗"
            null -> "?"   // unknowable, not unsatisfied — say so rather than guess
        }
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface,
                contentColor = MaterialTheme.colorScheme.onSurface,
            ),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    "$mark  $title",
                    color = when (satisfied) {
                        true -> MaterialTheme.colorScheme.primary
                        false -> MaterialTheme.colorScheme.tertiary
                        null -> MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
                Text(why, color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (satisfied != true) {
                    OutlinedButton(onClick = onFix) { Text("Open settings") }
                }
            }
        }
    }

    private fun canOverlay(ctx: Context): Boolean = Settings.canDrawOverlays(ctx)

    private fun batteryUnrestricted(ctx: Context): Boolean =
        ctx.getSystemService(PowerManager::class.java)
            .isIgnoringBatteryOptimizations(ctx.packageName)

    /** Whether the system's chosen assistant is us. */
    private fun isDefaultAssistant(ctx: Context): Boolean {
        val current = Settings.Secure.getString(
            ctx.contentResolver, "voice_interaction_service",
        ).orEmpty()
        return current.contains(ctx.packageName)
    }

    /**
     * MIUI's autostart list, which is not a platform Settings screen.
     *
     * The component name is Xiaomi's and can be renamed by a HyperOS update, so
     * a failure falls back to the app's own settings page rather than crashing
     * on an ActivityNotFoundException.
     */
    private fun openMiuiAutostart(ctx: Context) {
        val miui = Intent().setComponent(
            ComponentName(
                "com.miui.securitycenter",
                "com.miui.permcenter.autostart.AutoStartManagementActivity",
            ),
        )
        val opened = runCatching { ctx.startActivity(miui) }.isSuccess
        if (!opened) {
            ctx.startActivity(
                Intent(
                    Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                    Uri.parse("package:${ctx.packageName}"),
                ),
            )
        }
    }
}
