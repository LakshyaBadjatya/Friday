package com.friday.phone.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * FRIDAY's colours, taken from the HUD rather than invented here.
 *
 * The web surfaces are deep navy with teal and cyan; the phone is the same
 * product and should not arrive looking like a different one. It is also the
 * honest choice for a capture app — a screen that is mostly dark is a screen
 * that does not blind you when you open it to photograph something at a desk
 * at night.
 */
private val Navy = Color(0xFF06121F)
private val Panel = Color(0xFF0D1E30)
private val Teal = Color(0xFF2DD4BF)
private val Cyan = Color(0xFF4FE3FF)
private val Amber = Color(0xFFFFCE73)
private val Pale = Color(0xFFEAFCFF)
private val Muted = Color(0xFF7FA3B8)
private val Danger = Color(0xFFFF6B78)

private val FridayColors = darkColorScheme(
    primary = Teal,
    onPrimary = Navy,
    secondary = Cyan,
    onSecondary = Navy,
    tertiary = Amber,
    onTertiary = Navy,
    background = Navy,
    onBackground = Pale,
    surface = Panel,
    onSurface = Pale,
    surfaceVariant = Panel,
    onSurfaceVariant = Muted,
    outline = Muted,
    error = Danger,
    onError = Navy,
)

/**
 * Headings are monospaced on purpose: this is an instrument, and the HUD makes
 * the same choice for the same reason.
 */
private val FridayType = Typography(
    headlineMedium = TextStyle(
        fontFamily = FontFamily.Monospace,
        fontWeight = FontWeight.Bold,
        fontSize = 30.sp,
        letterSpacing = 6.sp,
    ),
    labelLarge = TextStyle(
        fontFamily = FontFamily.Monospace,
        fontWeight = FontWeight.Medium,
        fontSize = 14.sp,
        letterSpacing = 1.sp,
    ),
)

@Composable
fun FridayTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = FridayColors, typography = FridayType, content = content)
}
