package com.friday.tv

import android.content.Context

/** Persisted backend URL, bearer token, and paired device id (SharedPreferences). */
class TvConfig(context: Context) {
    private val prefs = context.getSharedPreferences("friday_tv", Context.MODE_PRIVATE)

    var baseUrl: String
        get() = prefs.getString("base_url", "")!!.trimEnd('/')
        set(v) = prefs.edit().putString("base_url", v.trimEnd('/')).apply()

    var token: String
        get() = prefs.getString("token", "")!!
        set(v) = prefs.edit().putString("token", v).apply()

    var deviceId: String
        get() = prefs.getString("device_id", "")!!
        set(v) = prefs.edit().putString("device_id", v).apply()

    val isConfigured: Boolean get() = baseUrl.isNotEmpty() && token.isNotEmpty()
}
