package com.friday.phone

import android.content.Context
import java.io.File
import java.util.concurrent.TimeUnit
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

/**
 * The three-step upload: ask FRIDAY to sign, put the bytes straight into
 * Cloudinary, then tell her to verify and file it.
 *
 * The API secret is never here. The signature the backend returns is scoped to
 * exactly one upload, so a stolen phone gets one upload slot, not an account —
 * and `commit` is what makes that safe, because the backend re-asks Cloudinary
 * what actually landed instead of believing this class about it.
 *
 * Every call returns null rather than throwing on a dead network or a refusal.
 * The one caller is a WorkManager worker whose answer to "no" is to keep the
 * capture queued and try later, and an exception would be a slower way of
 * saying the same thing.
 */
class Api(private val ctx: Context) {

    /**
     * The read timeout is the one that matters, and OkHttp's default of ten
     * seconds is far too short here. A free-tier host sleeps when idle and
     * takes the better part of a minute to wake, during which it sends nothing
     * at all — indistinguishable, to a ten-second reader, from a backend that
     * is never going to answer. That is what made the first question after a
     * quiet hour fail while the second one worked.
     */
    private val http = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .callTimeout(180, TimeUnit.SECONDS)
        .build()

    /** An HTTP answer: the status we got, and the body if there was one. */
    data class Reply(val code: Int, val body: JSONObject?) {
        val ok: Boolean get() = code in 200..299
    }

    private fun postReply(path: String, body: JSONObject): Reply {
        val base = Config.baseUrl(ctx).ifEmpty { return Reply(NO_RESPONSE, null) }
        val request = Request.Builder()
            .url("$base$path")
            .addHeader("Authorization", "Bearer ${Config.token(ctx)}")
            .post(body.toString().toRequestBody(JSON))
            .build()
        return runCatching {
            http.newCall(request).execute().use { response ->
                val text = response.body?.string().orEmpty()
                Reply(response.code, runCatching { JSONObject(text) }.getOrNull())
            }
        }.getOrDefault(Reply(NO_RESPONSE, null))
    }

    private fun post(path: String, body: JSONObject): JSONObject? {
        val base = Config.baseUrl(ctx).ifEmpty { return null }
        val request = Request.Builder()
            .url("$base$path")
            .addHeader("Authorization", "Bearer ${Config.token(ctx)}")
            .post(body.toString().toRequestBody(JSON))
            .build()
        return runCatching {
            http.newCall(request).execute().use { response ->
                val text = response.body?.string() ?: return null
                if (!response.isSuccessful) null else JSONObject(text)
            }
        }.getOrNull()
    }

    /** Step 1: a pending item and a one-shot Cloudinary signature. */
    fun sign(source: String, privacy: String = "private"): JSONObject? =
        post("/vault/sign", JSONObject().put("source", source).put("privacy", privacy))

    /**
     * Step 2: the bytes, straight to Cloudinary.
     *
     * Direct-to-Cloudinary rather than through FRIDAY: a photographed page is a
     * few megabytes, and routing that through the backend would pay for the
     * same bytes twice and put a phone's flaky uplink on her request thread.
     */
    fun uploadToCloudinary(upload: JSONObject, file: File): Boolean {
        val params = upload.getJSONObject("params")
        val builder = MultipartBody.Builder().setType(MultipartBody.FORM)
        params.keys().forEach { key ->
            builder.addFormDataPart(key, params.get(key).toString())
        }
        builder.addFormDataPart("file", file.name, file.asRequestBody(JPEG))
        val request = Request.Builder()
            .url(upload.getString("url"))
            .post(builder.build())
            .build()
        return runCatching {
            http.newCall(request).execute().use { it.isSuccessful }
        }.getOrDefault(false)
    }

    /**
     * Step 3: FRIDAY verifies with Cloudinary herself before trusting it.
     *
     * Returns the HTTP status as well as the body, because the caller has to
     * tell two failures apart: 409 means Cloudinary genuinely does not have the
     * asset (so the bytes must be sent again), while a timeout or a 5xx means
     * we could not ask (so sending them again would just duplicate them).
     */
    fun commit(itemId: String): Reply = postReply("/vault/items/$itemId/commit", JSONObject())

    /**
     * One spoken turn: base64 WAV in, transcript + reply + reply audio out.
     *
     * The socket at /ws/voice cannot do this — it carries control frames only —
     * so the phone's voice goes over plain HTTP like the Siri shortcut's does.
     */
    /**
     * Returns the status alongside the body, unlike the vault calls, because a
     * person is waiting on this one. "No answer" covers a dead network, an
     * expired token and a backend fault equally badly — and only one of those
     * is worth reaching for the phone over.
     */
    fun voice(audioB64: String, sessionId: String = "phone"): Reply =
        postReply(
            "/voice",
            JSONObject().put("audio_b64", audioB64).put("session_id", sessionId),
        )

    /** Ask her for notes over a selection of captures. */
    fun makeNotes(itemIds: List<String>, prompt: String = ""): JSONObject? =
        post(
            "/vault/notes",
            JSONObject().put("item_ids", JSONArray(itemIds)).put("prompt", prompt),
        )

    companion object {
        /** No status at all: the request never got an answer. */
        const val NO_RESPONSE = -1

        private val JSON = "application/json".toMediaType()
        private val JPEG = "image/jpeg".toMediaType()
    }
}
