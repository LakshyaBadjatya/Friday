package com.friday.phone.assistant

import android.Manifest
import android.app.assist.AssistContent
import android.app.assist.AssistStructure
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.service.voice.VoiceInteractionSession
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.friday.phone.ui.EdgeGlowView
import com.friday.phone.voice.VoiceClient
import com.friday.phone.voice.VoiceRecorder
import com.friday.phone.wake.WakeService
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.sin

/**
 * What the assist gesture opens.
 *
 * onHandleAssist receives the current screen's AssistStructure — the text the
 * foreground app has on screen, handed over by the platform. That is what makes
 * "what's this?" answerable without a screenshot, without the accessibility
 * service, and without asking for a single extra permission: the assistant slot
 * already carries the right.
 *
 * The session listens as soon as it is shown. An assistant that opens and waits
 * to be tapped is a worse button; the whole point of the gesture is that your
 * hands are already busy holding the phone up at the thing you are asking about.
 */
class FridaySession(context: Context) : VoiceInteractionSession(context) {

    private var readout: TextView? = null
    private var glow: EdgeGlowView? = null
    private val client by lazy { VoiceClient(context.applicationContext) }
    private val main = Handler(Looper.getMainLooper())

    /** What was on screen when she was summoned, if the platform offered it. */
    private var screenText: String = ""

    @Volatile private var listening = false
    @Volatile private var speaking = false

    /** Turns taken since she was summoned; 0 means nothing has been asked yet. */
    @Volatile private var turns = 0

    /**
     * One conversation, not one question.
     *
     * Kept for the life of the session so every turn reaches the backend under
     * the same id — that is what lets "and what about tomorrow?" mean anything,
     * because the orchestrator loads the short-term history for the session
     * before it answers.
     */
    private val conversation = "assist-${System.currentTimeMillis()}"

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

        val edge = EdgeGlowView(context).apply { state = EdgeGlowView.State.IDLE }
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

    /** Shown: start listening straight away. */
    override fun onShow(args: Bundle?, showFlags: Int) {
        super.onShow(args, showFlags)
        listen()
    }

    override fun onHide() {
        listening = false
        speaking = false
        client.stop()
        // The wake word gave up the microphone to summon her; give it back.
        WakeService.resumeListening()
        super.onHide()
    }

    override fun onHandleAssist(
        data: Bundle?,
        structure: AssistStructure?,
        content: AssistContent?,
    ) {
        screenText = structure?.let { flatten(it) }.orEmpty()
    }

    /**
     * One turn, entirely off the main thread.
     *
     * The recorder blocks until it hears the sentence end and the request that
     * follows can take seconds; either on the main thread would freeze the very
     * animation that is telling you she is awake.
     */
    private fun listen() {
        if (listening) return
        val granted = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO,
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            // The session window cannot request a permission; the app has to.
            say("Open FRIDAY once to allow the microphone.", EdgeGlowView.State.IDLE)
            return
        }

        listening = true
        say("Listening…", EdgeGlowView.State.LISTENING)

        thread(name = "friday-assist-turn") {
            val recorder = VoiceRecorder()
            // The edge answers the room while she is hearing it.
            recorder.onLevel = { level -> glow?.amplitude = level }
            val wav = runCatching { recorder.record() }.getOrNull()

            if (wav == null || !listening) {
                // Nobody spoke. After an answer that means the conversation is
                // over, so close rather than sit lit on a silent room.
                listening = false
                if (turns > 0) main.post { hide() } else say("I didn't catch that.", EdgeGlowView.State.IDLE)
                return@thread
            }
            turns++

            glow?.amplitude = 0f
            say("Thinking…", EdgeGlowView.State.THINKING)

            // The state has to flip here, not after ask() returns: playback
            // begins inside that call, and a pulse that checked for SPEAKING
            // before anything had set it would stop on its first tick.
            client.onSpeechStart = {
                speaking = true
                main.post { glow?.state = EdgeGlowView.State.SPEAKING }
                pulseWhileSpeaking()
            }
            client.onSpeechEnd = {
                speaking = false
                glow?.amplitude = 0f
                // Straight back to listening, so a follow-up is just the next
                // thing you say rather than another gesture. She stays on one
                // session id throughout, which is what gives the backend the
                // thread of the conversation to answer against.
                listening = false
                main.post { listen() }
            }

            when (val result = client.ask(wav, sessionId = conversation)) {
                is VoiceClient.Result.Spoke -> {
                    say(result.turn.text.ifBlank { "…" }, EdgeGlowView.State.SPEAKING)
                    // Listening resumes from onSpeechEnd, once she has actually
                    // finished the sentence — reopening the microphone now would
                    // have her transcribe her own reply.
                    if (!speaking) {
                        listening = false
                        main.post { listen() }
                    }
                    return@thread
                }
                VoiceClient.Result.Unheard -> {
                    listening = false
                    main.post { listen() }
                    return@thread
                }
                is VoiceClient.Result.Failed ->
                    say(result.message, EdgeGlowView.State.IDLE)
            }
            listening = false
        }
    }

    /**
     * Make the edge move while she talks.
     *
     * Reading the real amplitude of playback would mean a Visualizer, which
     * wants its own audio-session plumbing and a permission for what is
     * decoration. A cadence roughly the speed of speech carries the same
     * meaning — that she is the one talking now — at none of the cost.
     */
    private fun pulseWhileSpeaking() {
        val step = object : Runnable {
            var t = 0f
            override fun run() {
                if (!speaking) return
                t += 0.16f
                glow?.amplitude = 0.35f + 0.45f * abs(sin(t * 3f))
                main.postDelayed(this, 60L)
            }
        }
        main.post(step)
    }

    private fun say(message: String, state: EdgeGlowView.State) {
        main.post {
            readout?.text = message
            glow?.state = state
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
