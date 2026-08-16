package com.friday.phone

import android.content.Context

/**
 * Where FRIDAY lives, and what this device signs in with.
 *
 * Kept in SharedPreferences rather than compiled in, because the backend moves
 * between a laptop on the LAN and the deployed host and the app should not need
 * rebuilding for that. Nothing Cloudinary-shaped is ever stored here: the phone
 * holds a FRIDAY token and nothing else, and every upload signature it uses is
 * fetched per capture and dies with it.
 */
object Config {
    private const val PREFS = "friday"

    fun baseUrl(ctx: Context): String = prefs(ctx).getString("base_url", "") ?: ""

    fun setBaseUrl(ctx: Context, url: String) {
        // Trailing slashes are folded in here rather than at every call site, so
        // "$base/vault/sign" can never come out as "...//vault/sign".
        prefs(ctx).edit().putString("base_url", url.trim().trimEnd('/')).apply()
    }

    fun token(ctx: Context): String = prefs(ctx).getString("token", "") ?: ""

    fun setToken(ctx: Context, token: String) {
        prefs(ctx).edit().putString("token", token.trim()).apply()
    }

    /** Whether the app has been pointed at a backend yet. */
    fun isConfigured(ctx: Context): Boolean = baseUrl(ctx).isNotEmpty()

    private fun prefs(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
}
