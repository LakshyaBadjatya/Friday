package com.friday.phone.assistant

import android.service.voice.VoiceInteractionService

/**
 * Registers FRIDAY as the system assistant.
 *
 * Selecting her in Settings -> Apps -> Default apps -> Digital assistant app
 * routes the assist gesture here. The class is empty on purpose: the platform
 * only needs something to bind to, and the work happens in the session.
 */
class FridayInteractionService : VoiceInteractionService()
