package com.friday.phone.assistant

import android.os.Bundle
import android.service.voice.VoiceInteractionService

/**
 * Registers FRIDAY as the system assistant.
 *
 * Selecting her in Settings -> Assist & voice input -> Assist app routes the
 * assist gesture here. The class holds almost nothing: the platform needs
 * something to bind to, and the work happens in the session.
 *
 * The one exception is the live instance, kept so the wake word can bring her
 * up. Showing an assistant session is a right this service holds and a
 * background service does not — the wake word cannot legally start an activity
 * from Android 10 onwards, so without this the word was heard and then dropped.
 */
class FridayInteractionService : VoiceInteractionService() {

    override fun onReady() {
        super.onReady()
        instance = this
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        super.onDestroy()
    }

    companion object {
        /**
         * The running service, or null when FRIDAY is not the chosen assistant.
         *
         * A plain static reference to a Service looks like a leak and is not
         * one here: it is cleared in onDestroy, and the platform keeps exactly
         * one instance alive for exactly as long as she holds the slot.
         */
        @Volatile
        private var instance: FridayInteractionService? = null

        /**
         * Bring up the assistant overlay. Returns whether there was anyone to
         * ask — false means she does not hold the assistant slot, and the
         * caller needs its own way to show something.
         */
        fun showSession(): Boolean {
            val service = instance ?: return false
            return runCatching { service.showSession(Bundle(), 0) }.isSuccess
        }
    }
}
