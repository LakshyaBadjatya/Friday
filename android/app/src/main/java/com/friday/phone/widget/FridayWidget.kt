package com.friday.phone.widget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.widget.RemoteViews
import com.friday.phone.R
import com.friday.phone.capture.CameraActivity
import com.friday.phone.sync.QueueDb
import java.util.concurrent.Executors

/**
 * Home-screen widget: how many captures are still waiting, and a way to add one.
 *
 * Plain RemoteViews rather than Glance. Glance would mean pulling its own
 * Compose runtime into an app whose widget is one line of text and one button,
 * and the plan's reason for naming it was convenience, not capability.
 *
 * The queue count is the useful number here: everything else the widget could
 * show requires a network round trip, and this is the one fact that says
 * whether the phone still owes her anything.
 */
class FridayWidget : AppWidgetProvider() {

    override fun onUpdate(
        context: Context,
        manager: AppWidgetManager,
        widgetIds: IntArray,
    ) {
        // Room touches disk, and onUpdate runs on the main thread.
        io.execute {
            val pending = runCatching { QueueDb.get(context).pending().count() }.getOrDefault(0)
            widgetIds.forEach { id -> render(context, manager, id, pending) }
        }
    }

    private fun render(ctx: Context, manager: AppWidgetManager, id: Int, pending: Int) {
        val views = RemoteViews(ctx.packageName, R.layout.widget_friday).apply {
            setTextViewText(
                R.id.widget_status,
                when (pending) {
                    0 -> "FRIDAY — all filed"
                    1 -> "FRIDAY — 1 waiting"
                    else -> "FRIDAY — $pending waiting"
                },
            )
            setOnClickPendingIntent(
                R.id.widget_capture,
                PendingIntent.getActivity(
                    ctx,
                    0,
                    Intent(ctx, CameraActivity::class.java)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                    PendingIntent.FLAG_IMMUTABLE,
                ),
            )
        }
        manager.updateAppWidget(id, views)
    }

    companion object {
        private val io = Executors.newSingleThreadExecutor()

        /** Nudge every placed widget after the queue changes. */
        fun refresh(ctx: Context) {
            val manager = AppWidgetManager.getInstance(ctx)
            val ids = manager.getAppWidgetIds(ComponentName(ctx, FridayWidget::class.java))
            if (ids.isNotEmpty()) {
                ctx.sendBroadcast(
                    Intent(AppWidgetManager.ACTION_APPWIDGET_UPDATE)
                        .setComponent(ComponentName(ctx, FridayWidget::class.java))
                        .putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, ids),
                )
            }
        }
    }
}
