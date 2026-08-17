package com.friday.phone.ui

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SweepGradient
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.min
import kotlin.math.sin

/**
 * The glow that says she is listening.
 *
 * A light that runs around the edges of the screen, the way an assistant
 * overlay should: the content underneath stays readable, because the thing you
 * are asking about is usually the thing on screen. A panel that covered it
 * would make "what's this?" impossible to answer while looking at it.
 *
 * Drawn with Canvas rather than Compose on purpose. This has to live inside a
 * [android.service.voice.VoiceInteractionSession] window, which is not a
 * lifecycle or saved-state owner, and a ComposeView without those crashes on
 * attach. A View costs nothing here and works in both places.
 *
 * **The bloom is faked with stacked strokes, not a blur.** A BlurMaskFilter
 * forces the whole view onto the software renderer, and this view is the size
 * of the screen: every frame was then a full-screen blurred round-rect drawn on
 * the CPU, twice, with two filter objects allocated to do it. On a low-end
 * phone that is the difference between a light and a slideshow. Four concentric
 * strokes, widest and faintest first, read as the same soft edge and stay on
 * the GPU where a 60fps animation belongs.
 */
class EdgeGlowView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    /** What she is doing, which is the only thing the glow is trying to say. */
    enum class State { IDLE, LISTENING, THINKING, SPEAKING }

    var state: State = State.LISTENING
        set(value) {
            if (field == value) return
            field = value
            // Each state gets its own tempo. Reading the mood off the speed of
            // a light is faster than reading a label, and does not need words.
            spin.duration = when (value) {
                State.IDLE -> 9000L
                State.LISTENING -> 5200L
                State.THINKING -> 2200L   // urgent: she is working
                State.SPEAKING -> 3400L
            }
            if (spin.isRunning) spin.currentPlayTime = 0
        }

    /**
     * How loud the room is, 0..1, when listening — or how animated her reply is
     * when speaking. Drives the thickness of the light so the edge breathes with
     * the voice instead of pulsing on a timer that has nothing to do with you.
     *
     * Setting this does not invalidate: the animator is already redrawing every
     * frame, and a second invalidate per 80ms audio frame only queues work the
     * next vsync would have done anyway. It is written from the recording
     * thread and read while drawing, hence volatile.
     */
    @Volatile
    var amplitude: Float = 0f
        set(value) {
            field = value.coerceIn(0f, 1f)
        }

    private var smoothed = 0f

    private val bounds = RectF()
    private var phase = 0f
    private var sweep: SweepGradient? = null
    private val spinMatrix = Matrix()

    /** One paint per bloom layer, allocated once. */
    private val layers = Array(LAYERS) {
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
        }
    }

    private val spin = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 5200L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            phase = it.animatedValue as Float
            invalidate()
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        spin.start()
    }

    override fun onDetachedFromWindow() {
        // A light that keeps animating behind a dismissed window is a battery
        // drain nobody can see to complain about.
        spin.cancel()
        super.onDetachedFromWindow()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        val inset = STROKE_MAX / 2f
        bounds.set(inset, inset, w - inset, h - inset)
        sweep = SweepGradient(w / 2f, h / 2f, COLORS, STOPS)
        layers.forEach { it.shader = sweep }
    }

    override fun onDraw(canvas: Canvas) {
        if (bounds.isEmpty) return

        // Chase the microphone rather than snapping to it: rising fast enough
        // to feel immediate, falling slowly enough that the gaps between words
        // do not read as her stopping.
        val target = amplitude
        smoothed += (target - smoothed) * (if (target > smoothed) RISE else FALL)

        // Breathing: a slow sine so an idle screen still looks alive, plus
        // whatever the microphone is actually hearing.
        val breath = (sin(phase * TWO_PI) + 1f) / 2f
        val energy = when (state) {
            State.IDLE -> 0.22f + breath * 0.10f
            State.LISTENING -> 0.40f + smoothed * 0.60f
            State.THINKING -> 0.55f + breath * 0.45f
            State.SPEAKING -> 0.45f + smoothed * 0.55f
        }

        val radius = min(bounds.width(), bounds.height()) * 0.08f

        // Rotate the SHADER, not the canvas: the colours have to travel around
        // the edge while the edge itself stays welded to the screen. Rotating
        // the canvas would swing the whole border off into the corner.
        spinMatrix.setRotate(phase * 360f, bounds.centerX(), bounds.centerY())
        sweep?.setLocalMatrix(spinMatrix)

        val core = STROKE_MIN * (0.5f + energy * 0.5f)
        for (i in layers.indices) {
            // Widest and faintest on the outside, tightening to a bright core.
            val spread = 1f - i.toFloat() / LAYERS
            val paint = layers[i]
            paint.strokeWidth = core + (STROKE_MAX - core) * spread * energy
            paint.alpha = (BASE_ALPHA[i] * (0.45f + energy * 0.55f)).toInt().coerceIn(8, 255)
            canvas.drawRoundRect(bounds, radius, radius, paint)
        }
    }

    private companion object {
        const val TWO_PI = (Math.PI * 2).toFloat()
        const val STROKE_MIN = 12f
        const val STROKE_MAX = 54f
        const val LAYERS = 4
        const val RISE = 0.45f
        const val FALL = 0.12f

        /** Outermost layer faintest; the innermost carries the colour. */
        val BASE_ALPHA = floatArrayOf(26f, 48f, 96f, 210f)

        /**
         * FRIDAY's palette, not Google's four.
         *
         * Teal and cyan are the HUD's own; amber is what it already uses for
         * "something wants you"; the violet is the only addition, because three
         * hues around a full sweep leaves a visible seam where the ends meet.
         * The first colour is repeated last so the loop closes cleanly.
         */
        val COLORS = intArrayOf(
            Color.parseColor("#2DD4BF"),
            Color.parseColor("#4FE3FF"),
            Color.parseColor("#A78BFA"),
            Color.parseColor("#FFCE73"),
            Color.parseColor("#2DD4BF"),
        )
        val STOPS = floatArrayOf(0f, 0.28f, 0.55f, 0.8f, 1f)
    }
}
