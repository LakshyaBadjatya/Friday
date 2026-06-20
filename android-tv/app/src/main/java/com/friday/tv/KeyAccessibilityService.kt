package com.friday.tv

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.view.KeyEvent
import android.view.accessibility.AccessibilityEvent

/** Experimental: a chosen remote key opens the voice panel. Device-dependent —
 *  many TV keys (the Assistant/mic button especially) are consumed by the system
 *  before they reach an accessibility service, so this may never fire on the Mi remote. */
class KeyAccessibilityService : AccessibilityService() {
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}

    override fun onKeyEvent(event: KeyEvent): Boolean {
        // EDIT THIS to a free keycode (find it with `adb shell getevent -l`).
        val triggerKey = KeyEvent.KEYCODE_PROG_RED
        if (event.action == KeyEvent.ACTION_UP && event.keyCode == triggerKey) {
            startActivity(Intent(this, VoicePanelActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            return true // consume it
        }
        return super.onKeyEvent(event)
    }
}
