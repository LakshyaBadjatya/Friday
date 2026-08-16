package com.friday.phone.wake

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

/**
 * "Hey FRIDAY", scored on the device.
 *
 * This is openWakeWord's pipeline, reimplemented over onnxruntime-android
 * because the Python package is not an option here: raw audio to a
 * melspectrogram, 76-frame windows of that to a shared speech embedding, and
 * sixteen embeddings to the classifier trained in the Colab notebook.
 *
 * The constants are not guesses — they are openWakeWord's own, and the model
 * was trained against features computed exactly this way. Get any of them
 * wrong and the model still runs, still returns a number, and never fires;
 * that is why they are named here rather than inlined.
 */
class WakeWordDetector(ctx: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val melspec: OrtSession = session(ctx, "melspectrogram.onnx")
    private val embed: OrtSession = session(ctx, "embedding_model.onnx")
    private val classifier: OrtSession = session(ctx, "hey_friday.onnx")

    // Only the newest frame plus the convolution's lead-in is ever needed, so
    // this is a fixed window that shifts rather than a growing buffer: at 80 ms
    // a frame on a low-end phone, per-sample boxing is not free.
    private val raw = FloatArray(FRAME + MEL_CONTEXT)
    private var rawFilled = 0
    private val mel = ArrayDeque<FloatArray>()     // 32-bin melspectrogram frames
    private val embeddings = ArrayDeque<FloatArray>()  // 96-d speech embeddings

    init {
        // openWakeWord primes its feature buffer with silence at construction,
        // and the classifier was trained on windows that include the quiet
        // around the phrase. Without this the detector cannot fire until
        // sixteen embeddings have accumulated -- about two seconds of audio --
        // and "hey friday" said immediately after starting it does nothing.
        prime()
    }

    private fun session(ctx: Context, asset: String): OrtSession =
        ctx.assets.open(asset).use { env.createSession(it.readBytes()) }

    /** Push silence through the whole chain so the buffers start full. */
    private fun prime() {
        val quiet = ShortArray(FRAME)
        repeat(PRIME_FRAMES) { feed(quiet) }
    }

    /**
     * Feed one 80 ms frame (1280 samples at 16 kHz) and get the current score.
     *
     * Returns 0 until enough audio has accumulated to fill the pipeline — about
     * a second — rather than scoring a half-filled buffer, which reads as a
     * confident "no" when the honest answer is "not yet".
     */
    fun feed(frame: ShortArray): Float {
        // Shift the window left by one frame and append the new samples.
        System.arraycopy(raw, frame.size, raw, 0, raw.size - frame.size)
        for (i in frame.indices) raw[raw.size - frame.size + i] = frame[i].toFloat()
        if (rawFilled < raw.size) rawFilled += frame.size
        if (rawFilled < raw.size) return 0f

        // Melspectrogram over the new samples plus a little context, which is
        // what the convolution needs to produce frames aligned with the stream.
        val newFrames = runMelspec(raw)
        newFrames.forEach { mel.addLast(it) }
        while (mel.size > MEL_KEEP) mel.removeFirst()
        if (mel.size < EMBED_WINDOW) return 0f

        val emb = runEmbedding(mel.toList().takeLast(EMBED_WINDOW))
        embeddings.addLast(emb)
        while (embeddings.size > CLASSIFIER_WINDOW) embeddings.removeFirst()
        if (embeddings.size < CLASSIFIER_WINDOW) return 0f

        return runClassifier(embeddings.toList())
    }

    /** Drop everything heard so far — used after a detection, so it fires once. */
    fun reset() {
        java.util.Arrays.fill(raw, 0f)
        rawFilled = 0
        mel.clear()
        embeddings.clear()
        prime()
    }

    private fun runMelspec(samples: FloatArray): List<FloatArray> {
        val input = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(samples), longArrayOf(1, samples.size.toLong()),
        )
        input.use {
            melspec.run(mapOf(melspec.inputNames.first() to it)).use { result ->
                // (time, 1, frames, 32)
                val raw4 = result[0].value as Array<*>
                val out = mutableListOf<FloatArray>()
                raw4.forEach { t ->
                    (t as Array<*>).forEach { ch ->
                        (ch as Array<*>).forEach { row ->
                            val f = row as FloatArray
                            // openWakeWord's own transform: x / 10 + 2. The
                            // model was trained on features scaled this way.
                            out.add(FloatArray(f.size) { i -> f[i] / 10f + 2f })
                        }
                    }
                }
                return out
            }
        }
    }

    private fun runEmbedding(frames: List<FloatArray>): FloatArray {
        val flat = FloatArray(EMBED_WINDOW * MEL_BINS)
        frames.forEachIndexed { i, row ->
            System.arraycopy(row, 0, flat, i * MEL_BINS, MEL_BINS)
        }
        val input = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(flat),
            longArrayOf(1, EMBED_WINDOW.toLong(), MEL_BINS.toLong(), 1),
        )
        input.use {
            embed.run(mapOf(embed.inputNames.first() to it)).use { result ->
                // (1, 1, 1, 96)
                @Suppress("UNCHECKED_CAST")
                val out = result[0].value as Array<Array<Array<FloatArray>>>
                return out[0][0][0]
            }
        }
    }

    private fun runClassifier(window: List<FloatArray>): Float {
        val flat = FloatArray(CLASSIFIER_WINDOW * EMBED_DIM)
        window.forEachIndexed { i, e -> System.arraycopy(e, 0, flat, i * EMBED_DIM, EMBED_DIM) }
        val input = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(flat),
            longArrayOf(1, CLASSIFIER_WINDOW.toLong(), EMBED_DIM.toLong()),
        )
        input.use {
            classifier.run(mapOf(classifier.inputNames.first() to it)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val out = result[0].value as Array<FloatArray>
                return out[0][0]
            }
        }
    }

    private companion object {
        const val MEL_BINS = 32
        const val EMBED_WINDOW = 76      // melspectrogram frames per embedding
        const val EMBED_DIM = 96
        const val CLASSIFIER_WINDOW = 16 // embeddings the classifier scores
        const val FRAME = 1280           // 80 ms at 16 kHz
        const val MEL_CONTEXT = 480      // 3 hops of lead-in, as openWakeWord uses
        const val MEL_KEEP = 200
        const val PRIME_FRAMES = 30      // ~2.4 s of silence: enough to fill both buffers
    }
}
