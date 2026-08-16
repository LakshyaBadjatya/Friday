package com.friday.phone.voice

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import kotlin.math.abs

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
    private val silenceMillis: Int = 1_200,
    private val silenceLevel: Int = 900,
) {

    @SuppressLint("MissingPermission") // caller holds RECORD_AUDIO; enforced by the UI
    fun record(): ByteArray {
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

                val now = System.currentTimeMillis()
                if (peak > silenceLevel) {
                    lastVoice = now
                    heardAnything = true
                }
                // Only start counting silence once something has been said, or
                // a slow start to the sentence would end the recording before
                // the first word.
                if (heardAnything && now - lastVoice > silenceMillis) break
            }
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
    }
}
