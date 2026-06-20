package com.friday.tv

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Autostart the receiver service after the TV boots. */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED && TvConfig(context).isConfigured) {
            ReceiverService.start(context)
        }
    }
}
