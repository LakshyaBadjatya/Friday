package com.friday.phone.sync

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.Constraints
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.friday.phone.Api
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Drains the pending queue: sign, upload, commit, then delete the local copy.
 *
 * One failed capture never blocks the ones behind it — a refusal marks the whole
 * run for retry but the loop carries on, so a single unreadable file cannot wedge
 * the queue forever. Deleting the JPEG only after a successful commit is what
 * makes a retry safe: until FRIDAY has confirmed with Cloudinary that the bytes
 * landed, the only copy that exists is still on the phone.
 */
class UploadWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val api = Api(applicationContext)
        val dao = QueueDb.get(applicationContext).pending()
        var retry = false

        for (capture in dao.all()) {
            val file = File(capture.path)
            if (!file.exists()) {
                // The photograph is gone (cache eviction, or a user clearing
                // storage). Nothing to upload and nothing to wait for.
                dao.remove(capture.id)
                continue
            }

            // Already signed on an earlier run? Ask her to commit it before
            // sending anything. An upload can succeed on Cloudinary and still
            // look like a failure here — a timeout after the last byte reads
            // exactly like a timeout before the first — and starting over from
            // sign in that case mints a second item and stores the same
            // photograph twice. Commit is the cheap question that settles it,
            // because the server verifies with Cloudinary rather than with us.
            if (capture.itemId.isNotEmpty()) {
                val reply = api.commit(capture.itemId)
                when {
                    reply.ok -> {
                        file.delete()
                        dao.remove(capture.id)
                        continue
                    }
                    // 409 says Cloudinary does not have it — but it says that
                    // about an asset it accepted seconds ago too, because the
                    // Admin API can lag behind the upload that fed it. Trusting
                    // the first one re-uploads bytes that are already there:
                    // one duplicate out of two captures, measured. So the first
                    // 409 only buys a later look; a second one is believed.
                    reply.code == HTTP_CONFLICT && runAttemptCount < BELIEVE_409_AFTER -> {
                        retry = true
                        continue
                    }
                    reply.code == HTTP_CONFLICT -> dao.setItemId(capture.id, "")
                    // Could not ask (offline, 5xx, cold start). Try later
                    // rather than duplicate an upload that may have worked.
                    else -> {
                        retry = true
                        continue
                    }
                }
            }

            val signed = api.sign(capture.source, capture.privacy)
            if (signed == null) {
                retry = true
                continue
            }
            val itemId = signed.getString("item_id")
            // Record the item BEFORE the upload, so a crash or a timeout during
            // it still leaves the retry something to commit against.
            dao.setItemId(capture.id, itemId)

            if (!api.uploadToCloudinary(signed.getJSONObject("upload"), file)) {
                retry = true
                continue
            }
            if (!api.commit(itemId).ok) {
                retry = true
                continue
            }

            file.delete()
            dao.remove(capture.id)
        }

        if (retry) Result.retry() else Result.success()
    }

    companion object {
        /** Cloudinary does not have the asset — or has not indexed it yet. */
        private const val HTTP_CONFLICT = 409

        /**
         * How many runs a 409 must survive before the bytes are sent again.
         *
         * One is enough: the indexing lag is seconds, and the retry after it is
         * at least thirty. Higher would strand a genuinely missing upload for
         * no gain.
         */
        private const val BELIEVE_409_AFTER = 1

        /**
         * Queue a drain for when there is a network.
         *
         * Every capture surface calls this and nothing else — the constraint,
         * not the caller, decides when it is worth trying.
         */
        fun enqueue(ctx: Context) {
            val request = OneTimeWorkRequestBuilder<UploadWorker>()
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.CONNECTED)
                        .build(),
                )
                .build()
            WorkManager.getInstance(ctx).enqueue(request)
        }
    }
}
