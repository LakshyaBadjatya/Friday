package com.friday.phone.sync

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase

/**
 * A capture that has not reached FRIDAY yet.
 *
 * Captures never fail because the network did. They queue here with their
 * on-device OCR text already extracted, so the screen has something true to show
 * immediately, and WorkManager drains the queue when connectivity returns. The
 * photograph itself stays in cacheDir until the commit succeeds — the row and
 * the file are deleted together, so a half-finished upload leaves neither a
 * ghost row nor an orphaned megabyte.
 */
@Entity(tableName = "pending")
data class PendingCapture(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val path: String,
    val source: String,
    val privacy: String,
    val offlineOcr: String,
    val createdAt: Long,
)

@Dao
interface PendingDao {
    @Insert fun add(capture: PendingCapture): Long

    /** Oldest first: the order things were photographed in is the order she gets them. */
    @Query("SELECT * FROM pending ORDER BY createdAt ASC") fun all(): List<PendingCapture>

    @Query("DELETE FROM pending WHERE id = :id") fun remove(id: Long)

    @Query("SELECT COUNT(*) FROM pending") fun count(): Int
}

@Database(entities = [PendingCapture::class], version = 1, exportSchema = false)
abstract class QueueDb : RoomDatabase() {
    abstract fun pending(): PendingDao

    companion object {
        @Volatile private var instance: QueueDb? = null

        fun get(ctx: Context): QueueDb = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(
                ctx.applicationContext, QueueDb::class.java, "friday-queue",
            ).build().also { instance = it }
        }
    }
}
