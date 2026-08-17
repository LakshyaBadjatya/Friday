package com.friday.phone.assistant

import android.app.assist.AssistContent
import android.app.assist.AssistStructure
import android.content.Context
import android.os.Bundle
import android.service.voice.VoiceInteractionSession
import android.graphics.Color
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import com.friday.phone.ui.EdgeGlowView

/**
 * What the assist gesture opens.
 *
 * onHandleAssist receives the current screen's AssistStructure — the text the
 * foreground app has on screen, handed over by the platform. That is what makes
 * "what's this?" answerable without a screenshot, without the accessibility
 * service, and without asking for a single extra permission: the assistant slot
 * already carries the right.
 *
 * The structure is walked into plain text here and nothing else yet; the voice
 * socket is what will carry it to FRIDAY, and that lands with Step 3.
 */
class FridaySession(context: Context) : VoiceInteractionSession(context) {

    private var readout: TextView? = null
    private var glow: EdgeGlowView? = null

    /**
     * The overlay: a glow around the edges and a line of text at the bottom.
     *
     * Deliberately not a panel. The assist gesture is usually pressed *about*
     * something on screen, so covering it would hide the subject of the
     * question. The light says she is listening; the screen underneath stays
     * exactly where it was.
     */
    override fun onCreateContentView(): View {
        val frame = FrameLayout(context)

        val edge = EdgeGlowView(context).apply { state = EdgeGlowView.State.LISTENING }
        frame.addView(
            edge,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )

        val text = TextView(context).apply {
            textSize = 16f
            setTextColor(Color.parseColor("#EAFCFF"))
            setShadowLayer(24f, 0f, 0f, Color.parseColor("#06121F"))
            setPadding(64, 48, 64, 96)
            text = "FRIDAY is listening."
        }
        frame.addView(
            text,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM,
            ),
        )

        readout = text
        glow = edge
        return frame
    }

    override fun onHandleAssist(
        data: Bundle?,
        structure: AssistStructure?,
        content: AssistContent?,
    ) {
        val text = structure?.let { flatten(it) }.orEmpty()
        // She has the screen now, so the glow moves from listening to working.
        glow?.state = EdgeGlowView.State.THINKING
        readout?.text = if (text.isBlank()) {
            "FRIDAY is listening."
        } else {
            "On screen: ${text.take(300)}"
        }
    }

    /** Depth-first walk of the view tree, keeping only what a person could read. */
    private fun flatten(structure: AssistStructure): String {
        val out = StringBuilder()
        for (i in 0 until structure.windowNodeCount) {
            walk(structure.getWindowNodeAt(i).rootViewNode, out)
        }
        return out.toString().trim()
    }

    private fun walk(node: AssistStructure.ViewNode?, out: StringBuilder) {
        if (node == null) return
        node.text?.toString()?.takeIf { it.isNotBlank() }?.let { out.append(it).append(' ') }
        for (i in 0 until node.childCount) walk(node.getChildAt(i), out)
    }
}
