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

            val signed = api.sign(capture.source, capture.privacy)
            if (signed == null) {
                retry = true
                continue
            }
            if (!api.uploadToCloudinary(signed.getJSONObject("upload"), file)) {
                retry = true
                continue
            }
            if (api.commit(signed.getString("item_id")) == null) {
                retry = true
                continue
            }

            file.delete()
            dao.remove(capture.id)
        }

        if (retry) Result.retry() else Result.success()
    }

    companion object {
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
