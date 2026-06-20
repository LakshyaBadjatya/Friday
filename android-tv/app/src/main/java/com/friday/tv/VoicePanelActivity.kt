package com.friday.tv

import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.util.Locale
import kotlin.concurrent.thread

/** The assistant panel: capture one utterance, ask the backend, speak + act. */
class VoicePanelActivity : AppCompatActivity() {
    private lateinit var config: TvConfig
    private lateinit var api: Api
    private lateinit var runner: ActionRunner
    private var recognizer: SpeechRecognizer? = null
    private var tts: TextToSpeech? = null
    private lateinit var heard: TextView
    private lateinit var reply: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_panel)
        heard = findViewById(R.id.heard)
        reply = findViewById(R.id.reply)
        config = TvConfig(this)
        api = Api(config)
        runner = ActionRunner(this)
        tts = TextToSpeech(this) { }
        startListening()
    }

    private fun startListening() {
        recognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle?) {
                    val text = results
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull().orEmpty()
                    handle(text)
                }
                override fun onError(error: Int) { speak("Sorry, I didn't catch that."); finishSoon() }
                override fun onReadyForSpeech(p0: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(p0: Float) {}
                override fun onBufferReceived(p0: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onPartialResults(p0: Bundle?) {}
                override fun onEvent(p0: Int, p1: Bundle?) {}
            })
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        recognizer?.startListening(intent)
    }

    private fun handle(text: String) {
        if (text.isBlank()) { finishSoon(); return }
        heard.text = text
        thread {
            val resp: JSONObject? = api.ask(text)
            runOnUiThread {
                val speak = resp?.optString("speak").orEmpty()
                reply.text = speak
                if (speak.isNotEmpty()) speak(speak)
                val action = resp?.optJSONObject("action")
                if (action != null) runner.run(action)
                finishSoon()
            }
        }
    }

    private fun speak(s: String) = tts?.speak(s, TextToSpeech.QUEUE_FLUSH, null, "friday")

    private fun finishSoon() { window.decorView.postDelayed({ finish() }, 3500) }

    override fun onDestroy() { recognizer?.destroy(); tts?.shutdown(); super.onDestroy() }
}
