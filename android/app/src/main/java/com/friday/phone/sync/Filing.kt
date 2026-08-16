package com.friday.phone.sync

import android.content.Context
import com.friday.phone.widget.FridayWidget
import java.io.File
import java.io.InputStream

/**
 * The one way anything enters the vault from this phone.
 *
 * Every capture surface — camera, share sheet, tile, scanner — ends here, so
 * "what happens to a capture" is answered in one place rather than four times
 * with three of them slightly wrong. Copy the bytes into cacheDir, write the
 * row, wake the worker.
 */
object Filing {

    /** File an existing on-disk capture and schedule its upload. */
    fun file(ctx: Context, path: String, source: String, offlineOcr: String = "") {
        QueueDb.get(ctx).pending().add(
            PendingCapture(
                path = path,
                source = source,
                privacy = "private",
                offlineOcr = offlineOcr,
                createdAt = System.currentTimeMillis(),
            ),
        )
        UploadWorker.enqueue(ctx)
        // A queue count on the home screen is worse than none if it is stale.
        FridayWidget.refresh(ctx)
    }

    /**
     * Copy a stream (a shared image, a scanned page) into the cache and file it.
     *
     * The bytes are copied rather than referenced because a content:// URI from
     * another app is borrowed, not owned — the permission behind it can be gone
     * by the time the worker runs, which on a queue that survives offline spells
     * hours, not milliseconds.
     */
    fun fileStream(ctx: Context, stream: InputStream, source: String): File {
        val target = File(ctx.cacheDir, "cap-${System.currentTimeMillis()}.jpg")
        stream.use { input -> target.outputStream().use { input.copyTo(it) } }
        file(ctx, target.absolutePath, source)
        return target
    }
}
