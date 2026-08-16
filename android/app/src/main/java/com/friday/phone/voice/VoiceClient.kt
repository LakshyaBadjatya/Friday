package com.friday.phone.voice

import android.content.Context
import android.media.MediaPlayer
import android.util.Base64
import com.friday.phone.Api
import java.io.File
import org.json.JSONObject

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

    fun ask(wav: ByteArray, sessionId: String = "phone"): Turn? {
        val audio = Base64.encodeToString(wav, Base64.NO_WRAP)
        val reply: JSONObject = api.voice(audio, sessionId) ?: return null

        reply.optString("audio_b64").takeIf { it.isNotEmpty() }?.let { speak(it) }
        return Turn(
            transcript = reply.optString("transcript"),
            text = reply.optString("text"),
            mode = reply.optString("mode"),
        )
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
            setOnCompletionListener { file.delete() }
            prepare()
            start()
        }
    }

    /** Silence her: used on barge-in and when the screen goes away. */
    fun stop() {
        runCatching { player?.stop() }
        player?.release()
        player = null
    }
}
