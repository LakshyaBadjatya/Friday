package com.friday.phone.wake

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/**
 * Puts her back to listening after a restart.
 *
 * A wake word is the one feature nobody re-enables by hand, because the way you
 * discover it is off is by saying her name to a phone that does not answer.
 * Only starts the service if listening was actually switched on.
 */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent?) {
        if (intent?.action != Intent.ACTION_BOOT_COMPLETED) return
        if (!WakeService.isEnabled(context)) return
        WakeService.start(context)
    }
}
