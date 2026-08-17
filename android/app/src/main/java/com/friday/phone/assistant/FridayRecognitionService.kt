package com.friday.phone.assistant

import android.content.Intent
import android.speech.RecognitionService
import android.speech.SpeechRecognizer

/**
 * Exists so the platform will accept FRIDAY as an assistant at all.
 *
 * `VoiceInteractionServiceInfo` refuses to parse a service whose metadata has no
 * `android:recognitionService`, and a service that fails to parse is dropped from
 * the assistant list silently — no log, no error, the app simply never appears in
 * Settings -> Voice assistant. Declaring a recogniser is the price of admission
 * even for an assistant that never uses the system one.
 *
 * FRIDAY does her own listening: the session records with VoiceRecorder and posts
 * to the backend, so nothing here is on her path. Other apps may still bind this
 * if the user picks FRIDAY as the speech recogniser, and for them the honest
 * answer is that she does not offer that service. Reporting ERROR_CLIENT tells
 * the caller to fall back rather than leaving it waiting on a result that will
 * never arrive.
 */
class FridayRecognitionService : RecognitionService() {
    override fun onStartListening(recognizerIntent: Intent?, listener: Callback?) {
        listener?.error(SpeechRecognizer.ERROR_CLIENT)
    }

    override fun onStopListening(listener: Callback?) {}

    override fun onCancel(listener: Callback?) {}
}
