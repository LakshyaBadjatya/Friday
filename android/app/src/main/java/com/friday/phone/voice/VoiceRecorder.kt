package com.friday.phone.voice

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Records one spoken turn as 16 kHz mono PCM and wraps it as a WAV.
 *
 * 16 kHz mono because that is what every STT backend in FRIDAY expects and
 * what the wake-word features are computed at; sending 44.1 kHz stereo would
 * mean resampling on the server for no gain.
 *
 * It stops on silence rather than on a second button press. A spoken question
 * has an end, the speaker knows when they have reached it, and making them
 * reach for the phone again to say so is the kind of small friction that stops
 * a feature being used.
 */
class VoiceRecorder(
    private val maxMillis: Int = 15_000,
    // 800ms rather than 1200: this is dead time on every single turn, paid
    // after the speaker already knows they have finished. Short enough to feel
    // prompt, still longer than the pause in the middle of a sentence.
    private val silenceMillis: Int = 800,
    private val silenceLevel: Int = 900,
    // How long to wait for the first word before giving up. Only matters when
    // she reopens the microphone for a follow-up nobody intends to give: without
    // it, an unanswered "anything else?" holds the mic for the full maximum.
    private val leadInMillis: Int = 3_500,
) {

    /**
     * How loud the room is right now, 0..1, reported per 80ms frame.
     *
     * The glow reads this so the edge answers the room rather than a timer.
     * Called on the recording thread — a listener that touches a view must hop
     * to the main one itself.
     */
    var onLevel: ((Float) -> Unit)? = null

    /**
     * Record one utterance, or null if nobody said anything.
     *
     * Null rather than an empty WAV so a caller cannot accidentally send
     * silence off to be transcribed, which costs a round trip to be told what
     * the microphone already knew.
     */
    @SuppressLint("MissingPermission") // caller holds RECORD_AUDIO; enforced by the UI
    fun record(): ByteArray? {
        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        val bufferSize = maxOf(minBuffer, FRAME * 2)
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE, CHANNEL, ENCODING, bufferSize,
        )
        val pcm = ByteArrayOutputStream()
        val frame = ShortArray(FRAME)

        recorder.startRecording()
        try {
            val started = System.currentTimeMillis()
            var lastVoice = started
            var heardAnything = false

            while (System.currentTimeMillis() - started < maxMillis) {
                val read = recorder.read(frame, 0, frame.size)
                if (read <= 0) continue

                var peak = 0
                for (i in 0 until read) {
                    val v = abs(frame[i].toInt())
                    if (v > peak) peak = v
                    pcm.write(frame[i].toInt() and 0xFF)
                    pcm.write((frame[i].toInt() shr 8) and 0xFF)
                }

                // Square-rooted so ordinary speech fills most of the range: a
                // linear peak against full scale leaves a voice sitting near
                // the bottom, and a light that barely moves reads as broken.
                onLevel?.invoke(sqrt((peak / MAX_PEAK).coerceIn(0f, 1f)))

                val now = System.currentTimeMillis()
                if (peak > silenceLevel) {
                    lastVoice = now
                    heardAnything = true
                }
                // Only start counting silence once something has been said, or
                // a slow start to the sentence would end the recording before
                // the first word.
                if (heardAnything && now - lastVoice > silenceMillis) break
                if (!heardAnything && now - started > leadInMillis) break
            }
            if (!heardAnything) return null
        } finally {
            recorder.stop()
            recorder.release()
        }
        return wav(pcm.toByteArray())
    }

    /** A 44-byte canonical WAV header in front of the raw samples. */
    private fun wav(pcm: ByteArray): ByteArray {
        val out = ByteArrayOutputStream(44 + pcm.size)
        val byteRate = SAMPLE_RATE * 2
        fun int(v: Int) {
            out.write(v and 0xFF); out.write((v shr 8) and 0xFF)
            out.write((v shr 16) and 0xFF); out.write((v shr 24) and 0xFF)
        }
        fun short(v: Int) { out.write(v and 0xFF); out.write((v shr 8) and 0xFF) }

        out.write("RIFF".toByteArray()); int(36 + pcm.size); out.write("WAVE".toByteArray())
        out.write("fmt ".toByteArray()); int(16); short(1); short(1)
        int(SAMPLE_RATE); int(byteRate); short(2); short(16)
        out.write("data".toByteArray()); int(pcm.size); out.write(pcm)
        return out.toByteArray()
    }

    private companion object {
        const val SAMPLE_RATE = 16_000
        const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        const val FRAME = 1280   // 80 ms, the same frame the wake word scores
        const val MAX_PEAK = 12_000f  // a loud voice, not full scale
    }
}
