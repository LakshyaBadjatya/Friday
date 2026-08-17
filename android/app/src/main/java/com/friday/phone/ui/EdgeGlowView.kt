package com.friday.phone.ui

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.BlurMaskFilter
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
 */
class EdgeGlowView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    /** What she is doing, which is the only thing the glow is trying to say. */
    enum class State { IDLE, LISTENING, THINKING, SPEAKING }

    var state: State = State.LISTENING
        set(value) {
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
            invalidate()
        }

    /**
     * How loud the room is, 0..1, when listening — or how animated her reply is
     * when speaking. Drives the thickness of the light so the edge breathes with
     * the voice instead of pulsing on a timer that has nothing to do with you.
     */
    var amplitude: Float = 0f
        set(value) {
            field = value.coerceIn(0f, 1f)
            invalidate()
        }

    private val bounds = RectF()
    private var phase = 0f
    private var sweep: SweepGradient? = null
    private val spinMatrix = Matrix()

    private val glow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val bloom = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
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

    init {
        // Blur needs software rendering; hardware layers ignore mask filters.
        setLayerType(LAYER_TYPE_SOFTWARE, null)
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
        glow.shader = sweep
        bloom.shader = sweep
    }

    override fun onDraw(canvas: Canvas) {
        if (bounds.isEmpty) return

        // Breathing: a slow sine so an idle screen still looks alive, plus
        // whatever the microphone is actually hearing.
        val breath = (sin(phase * TWO_PI) + 1f) / 2f
        val energy = when (state) {
            State.IDLE -> 0.25f
            State.LISTENING -> 0.45f + amplitude * 0.55f
            State.THINKING -> 0.55f + breath * 0.45f
            State.SPEAKING -> 0.5f + amplitude * 0.5f
        }

        val radius = min(bounds.width(), bounds.height()) * 0.08f

        // Rotate the SHADER, not the canvas: the colours have to travel around
        // the edge while the edge itself stays welded to the screen. Rotating
        // the canvas would swing the whole border off into the corner.
        spinMatrix.setRotate(phase * 360f, bounds.centerX(), bounds.centerY())
        sweep?.setLocalMatrix(spinMatrix)

        // Two passes: a wide soft bloom, then a tighter bright core. One pass
        // alone reads as a coloured border; two read as light.
        bloom.strokeWidth = STROKE_MIN + (STROKE_MAX - STROKE_MIN) * energy
        bloom.maskFilter = BlurMaskFilter(bloom.strokeWidth * 1.6f, BlurMaskFilter.Blur.NORMAL)
        bloom.alpha = (110 * energy).toInt().coerceIn(24, 140)
        canvas.drawRoundRect(bounds, radius, radius, bloom)

        glow.strokeWidth = (STROKE_MIN * 0.55f) + (STROKE_MIN * 0.75f) * energy
        glow.maskFilter = BlurMaskFilter(glow.strokeWidth, BlurMaskFilter.Blur.NORMAL)
        glow.alpha = (200 * (0.55f + energy * 0.45f)).toInt().coerceIn(60, 255)
        canvas.drawRoundRect(bounds, radius, radius, glow)
    }

    private companion object {
        const val TWO_PI = (Math.PI * 2).toFloat()
        const val STROKE_MIN = 14f
        const val STROKE_MAX = 46f

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
