package com.friday.phone.tile

import android.content.Intent
import android.service.quicksettings.TileService
import com.friday.phone.capture.ScreenCaptureActivity

/** Pull down the shade, tap FRIDAY, and whatever is on screen is filed. */
class FridayTileService : TileService() {
    override fun onClick() {
        super.onClick()
        val intent = Intent(this, ScreenCaptureActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        // Deprecated on 34+ in favour of the PendingIntent overload, but the
        // replacement does not exist on 29, which is this app's floor.
        @Suppress("DEPRECATION")
        startActivityAndCollapse(intent)
    }
}
