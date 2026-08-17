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
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

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
    /**
     * The vault item this capture was signed into, empty until it has been.
     *
     * Without this a retry started again from `sign`, which mints a NEW item
     * and uploads the same bytes again. Two photographs became thirty-nine
     * pending items and seventy megabytes of duplicate assets before anyone
     * noticed, because an upload that succeeds server-side can still look like
     * a failure to the client that sent it.
     */
    val itemId: String = "",
)

@Dao
interface PendingDao {
    @Insert fun add(capture: PendingCapture): Long

    /** Oldest first: the order things were photographed in is the order she gets them. */
    @Query("SELECT * FROM pending ORDER BY createdAt ASC") fun all(): List<PendingCapture>

    @Query("DELETE FROM pending WHERE id = :id") fun remove(id: Long)

    @Query("UPDATE pending SET itemId = :itemId WHERE id = :id")
    fun setItemId(id: Long, itemId: String)

    @Query("SELECT COUNT(*) FROM pending") fun count(): Int
}

@Database(entities = [PendingCapture::class], version = 2, exportSchema = false)
abstract class QueueDb : RoomDatabase() {
    abstract fun pending(): PendingDao

    companion object {
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE pending ADD COLUMN itemId TEXT NOT NULL DEFAULT ''")
            }
        }

        @Volatile private var instance: QueueDb? = null

        fun get(ctx: Context): QueueDb = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(
                ctx.applicationContext, QueueDb::class.java, "friday-queue",
            )
                // Migrate rather than fall back to destructive: the rows here
                // are captures that have not reached her yet, and dropping the
                // table to add a column would throw away the photographs.
                .addMigrations(MIGRATION_1_2)
                .build().also { instance = it }
        }
    }
}
