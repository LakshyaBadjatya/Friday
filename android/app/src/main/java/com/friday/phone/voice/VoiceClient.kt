package com.friday.phone.voice

import android.content.Context
import android.media.MediaPlayer
import android.util.Base64
import com.friday.phone.Api
import java.io.File

/**
 * One spoken turn: record, send, speak the answer.
 *
 * This goes to `POST /voice`, not to the `/ws/voice` socket, because the socket
 * does not carry audio — it announces readiness and echoes control frames, and
 * says so in its own docstring. `POST /voice` is the endpoint that actually
 * transcribes, drives the orchestrator and returns synthesized speech, so it is
 * the one a phone can hold a conversation with today.
 */
class VoiceClient(private val ctx: Context) {

    private val api = Api(ctx)
    private var player: MediaPlayer? = null

    /** Result of a turn: what she heard, what she said, and how she answered. */
    data class Turn(val transcript: String, val text: String, val mode: String)

    /**
     * Either a turn or the reason there isn't one.
     *
     * A failed turn used to come back as null, which the screen could only
     * report as "no answer" — the same words for a sleeping backend, a wrong
     * token and a fault upstream. Only one of those is worth doing anything
     * about, so they are worth telling apart.
     */
    sealed interface Result {
        data class Spoke(val turn: Turn) : Result

        /** She heard nothing intelligible — not a failure, just silence. */
        data object Unheard : Result

        data class Failed(val message: String) : Result
    }

    /** Told when she starts and stops speaking, so a caller can light something. */
    var onSpeechStart: (() -> Unit)? = null
    var onSpeechEnd: (() -> Unit)? = null

    fun ask(wav: ByteArray, sessionId: String = "phone"): Result {
        val audio = Base64.encodeToString(wav, Base64.NO_WRAP)
        val reply = api.voice(audio, sessionId)

        if (!reply.ok) return Result.Failed(explain(reply))
        val body = reply.body ?: return Result.Failed("She answered with nothing at all.")

        val transcript = body.optString("transcript")
        val text = body.optString("text")
        // The backend returns an empty turn when the audio held no speech.
        if (transcript.isEmpty() && text.isEmpty()) return Result.Unheard

        body.optString("audio_b64").takeIf { it.isNotEmpty() }?.let { speak(it) }
        return Result.Spoke(
            Turn(transcript = transcript, text = text, mode = body.optString("mode")),
        )
    }

    /** Say what actually went wrong, in words worth reading on a phone screen. */
    private fun explain(reply: Api.Reply): String {
        val detail = reply.body?.optString("detail").orEmpty()
        return when (reply.code) {
            Api.NO_RESPONSE -> "Couldn't reach FRIDAY. She may be asleep — try once more."
            401, 403 -> "The token was refused. Check it in Setup."
            404 ->
                if (detail == "voice disabled") "Voice is switched off on the backend."
                else "That backend has no /voice endpoint — check the URL in Setup."
            502, 503, 504 -> "FRIDAY is awake but her voice backend isn't answering."
            else -> "She answered ${reply.code}${if (detail.isEmpty()) "" else ": $detail"}"
        }
    }

    /**
     * Play her reply, stopping whatever she was already saying.
     *
     * Barge-in in the only form that matters on a phone: a second question
     * interrupts the first answer instead of queueing behind it.
     */
    private fun speak(audioB64: String) {
        stop()
        val bytes = runCatching { Base64.decode(audioB64, Base64.DEFAULT) }.getOrNull() ?: return
        val file = File(ctx.cacheDir, "reply-${System.currentTimeMillis()}.audio")
        file.writeBytes(bytes)
        player = MediaPlayer().apply {
            setDataSource(file.absolutePath)
            setOnCompletionListener {
                file.delete()
                onSpeechEnd?.invoke()
            }
            prepare()
            start()
        }
        onSpeechStart?.invoke()
    }

    /** Silence her: used on barge-in and when the screen goes away. */
    fun stop() {
        runCatching { player?.stop() }
        player?.release()
        player = null
    }
}
