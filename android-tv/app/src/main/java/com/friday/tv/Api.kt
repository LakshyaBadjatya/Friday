package com.friday.tv

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/** Thin backend client. All calls carry the bearer token; JSON in, JSON out. */
class Api(private val config: TvConfig) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS) // 0 = no read timeout (for the WebSocket)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()
    private val jsonType = "application/json".toMediaType()

    /** POST /tv/ask — returns the parsed JSON envelope, or null on failure. */
    fun ask(query: String): JSONObject? {
        val body = JSONObject().put("q", query).toString().toRequestBody(jsonType)
        val req = Request.Builder()
            .url("${config.baseUrl}/tv/ask")
            .addHeader("Authorization", "Bearer ${config.token}")
            .post(body)
            .build()
        return runCatching {
            client.newCall(req).execute().use { resp ->
                resp.body?.string()?.let { JSONObject(it) }
            }
        }.getOrNull()
    }

    /** POST /tv/pair — returns the new device id, or null. */
    fun pair(name: String): String? {
        val body = JSONObject().put("name", name).toString().toRequestBody(jsonType)
        val req = Request.Builder()
            .url("${config.baseUrl}/tv/pair")
            .addHeader("Authorization", "Bearer ${config.token}")
            .post(body)
            .build()
        return runCatching {
            client.newCall(req).execute().use { resp ->
                resp.body?.string()?.let { JSONObject(it).optString("device_id") }
            }
        }.getOrNull()?.ifEmpty { null }
    }

    /** Open the /tv/stream WebSocket; each pushed action arrives as a JSONObject. */
    fun openStream(deviceId: String, onAction: (JSONObject) -> Unit): WebSocket {
        val wsUrl = config.baseUrl.replaceFirst("http", "ws") + "/tv/stream?device_id=$deviceId"
        val req = Request.Builder()
            .url(wsUrl)
            .addHeader("Authorization", "Bearer ${config.token}")
            .build()
        return client.newWebSocket(req, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) {
                runCatching { onAction(JSONObject(text)) }
            }
        })
    }
}
