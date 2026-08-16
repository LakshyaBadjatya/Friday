package com.friday.phone.assistant

import android.app.assist.AssistContent
import android.app.assist.AssistStructure
import android.content.Context
import android.os.Bundle
import android.service.voice.VoiceInteractionSession
import android.view.View
import android.widget.TextView

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

    override fun onCreateContentView(): View {
        val view = TextView(context).apply {
            textSize = 16f
            setPadding(48, 48, 48, 48)
            text = "FRIDAY is listening."
        }
        readout = view
        return view
    }

    override fun onHandleAssist(
        data: Bundle?,
        structure: AssistStructure?,
        content: AssistContent?,
    ) {
        val text = structure?.let { flatten(it) }.orEmpty()
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
