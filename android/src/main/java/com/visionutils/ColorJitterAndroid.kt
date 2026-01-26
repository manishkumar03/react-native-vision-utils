package com.visionutils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import java.io.ByteArrayOutputStream
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.random.Random

/**
 * ColorJitter provides granular color augmentation for data augmentation pipelines.
 *
 * Supports random sampling within specified ranges for:
 * - Brightness: additive adjustment
 * - Contrast: multiplicative adjustment around mean
 * - Saturation: multiplicative adjustment
 * - Hue: cyclic shift
 *
 * Example Usage (from JS):
 * ```typescript
 * const result = await colorJitter(
 *   { uri: 'file://image.jpg' },
 *   {
 *     brightness: 0.2,          // [-0.2, +0.2]
 *     contrast: [0.8, 1.2],     // [0.8, 1.2]
 *     saturation: 0.3,          // [0.7, 1.3]
 *     hue: 0.1,                 // [-0.1, +0.1]
 *     seed: 42                  // for reproducibility
 *   }
 * );
 * ```
 */
object ColorJitterAndroid {

    /**
     * Apply color jitter augmentation to an image.
     *
     * @param bitmap Source bitmap
     * @param options Color jitter options (brightness, contrast, saturation, hue, seed)
     * @return WritableMap containing augmented image and applied values
     */
    fun apply(bitmap: Bitmap, options: ReadableMap): WritableMap {
        val startTimeNs = System.nanoTime()

        // Parse options and get ranges
        val brightnessRange = parseRange(options, "brightness", 0.0, 0.0, isMultiplicative = false)
        val contrastRange = parseRange(options, "contrast", 1.0, 1.0, isMultiplicative = true)
        val saturationRange = parseRange(options, "saturation", 1.0, 1.0, isMultiplicative = true)
        val hueRange = parseRange(options, "hue", 0.0, 0.0, isMultiplicative = false)

        // Setup random number generator
        val seed = if (options.hasKey("seed") && !options.isNull("seed")) {
            options.getInt("seed")
        } else {
            Random.nextInt(Int.MAX_VALUE)
        }
        val random = Random(seed.toLong())

        // Sample random values within ranges
        val appliedBrightness = randomInRange(brightnessRange, random)
        val appliedContrast = randomInRange(contrastRange, random)
        val appliedSaturation = randomInRange(saturationRange, random)
        val appliedHue = randomInRange(hueRange, random)

        // Apply transformations
        var result = bitmap.copy(Bitmap.Config.ARGB_8888, true)

        // Apply brightness
        if (appliedBrightness != 0.0) {
            result = adjustBrightness(result, appliedBrightness.toFloat())
        }

        // Apply contrast
        if (appliedContrast != 1.0) {
            result = adjustContrast(result, appliedContrast.toFloat())
        }

        // Apply saturation
        if (appliedSaturation != 1.0) {
            result = adjustSaturation(result, appliedSaturation.toFloat())
        }

        // Apply hue shift
        if (appliedHue != 0.0) {
            result = adjustHue(result, appliedHue.toFloat())
        }

        // Encode to base64
        val outputStream = ByteArrayOutputStream()
        result.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
        val base64String = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)

        val width = result.width
        val height = result.height
        val processingTimeMs = (System.nanoTime() - startTimeNs) / 1_000_000.0

        result.recycle()

        return Arguments.createMap().apply {
            putString("base64", base64String)
            putInt("width", width)
            putInt("height", height)
            putDouble("appliedBrightness", appliedBrightness)
            putDouble("appliedContrast", appliedContrast)
            putDouble("appliedSaturation", appliedSaturation)
            putDouble("appliedHue", appliedHue)
            putInt("seed", seed)
            putDouble("processingTimeMs", processingTimeMs)
        }
    }

    // MARK: - Range Parsing

    /**
     * Parse a range value from options.
     * - Single number: symmetric range around default
     * - Tuple [min, max]: explicit range
     */
    private fun parseRange(
        options: ReadableMap,
        key: String,
        defaultMin: Double,
        defaultMax: Double,
        isMultiplicative: Boolean
    ): Pair<Double, Double> {
        if (!options.hasKey(key) || options.isNull(key)) {
            return Pair(defaultMin, defaultMax)
        }

        // Check if it's an array (tuple)
        try {
            val array = options.getArray(key)
            if (array != null && array.size() == 2) {
                return Pair(array.getDouble(0), array.getDouble(1))
            }
        } catch (_: Exception) {
            // Not an array, try as single number
        }

        // Single number
        try {
            val value = options.getDouble(key)
            return if (isMultiplicative) {
                // For contrast/saturation: range is [max(0, 1-v), 1+v]
                Pair(max(0.0, 1.0 - value), 1.0 + value)
            } else {
                // For brightness/hue: range is [-v, +v]
                Pair(-value, value)
            }
        } catch (_: Exception) {
            return Pair(defaultMin, defaultMax)
        }
    }

    /**
     * Generate a random value within a range.
     */
    private fun randomInRange(range: Pair<Double, Double>, random: Random): Double {
        if (range.first == range.second) {
            return range.first
        }
        return range.first + random.nextDouble() * (range.second - range.first)
    }

    // MARK: - Color Adjustments

    private fun adjustBrightness(bitmap: Bitmap, value: Float): Bitmap {
        // Value is additive (-1 to +1), scale to pixel range
        val offset = value * 255f

        val colorMatrix = ColorMatrix(floatArrayOf(
            1f, 0f, 0f, 0f, offset,
            0f, 1f, 0f, 0f, offset,
            0f, 0f, 1f, 0f, offset,
            0f, 0f, 0f, 1f, 0f
        ))

        return applyColorMatrix(bitmap, colorMatrix)
    }

    private fun adjustContrast(bitmap: Bitmap, factor: Float): Bitmap {
        val translate = (1f - factor) * 127.5f

        val colorMatrix = ColorMatrix(floatArrayOf(
            factor, 0f, 0f, 0f, translate,
            0f, factor, 0f, 0f, translate,
            0f, 0f, factor, 0f, translate,
            0f, 0f, 0f, 1f, 0f
        ))

        return applyColorMatrix(bitmap, colorMatrix)
    }

    private fun adjustSaturation(bitmap: Bitmap, factor: Float): Bitmap {
        val colorMatrix = ColorMatrix().apply {
            setSaturation(factor)
        }

        return applyColorMatrix(bitmap, colorMatrix)
    }

    private fun adjustHue(bitmap: Bitmap, value: Float): Bitmap {
        // Value is fraction of color wheel (0-1 = 0-360°)
        val angleRadians = value * 2 * Math.PI.toFloat()

        // Hue rotation matrix
        // Based on rotating the RGB color cube around the gray axis (1,1,1)
        val cosA = cos(angleRadians.toDouble()).toFloat()
        val sinA = sin(angleRadians.toDouble()).toFloat()

        // Luminance weights
        val lumR = 0.213f
        val lumG = 0.715f
        val lumB = 0.072f

        val colorMatrix = ColorMatrix(floatArrayOf(
            lumR + cosA * (1 - lumR) + sinA * (-lumR),
            lumG + cosA * (-lumG) + sinA * (-lumG),
            lumB + cosA * (-lumB) + sinA * (1 - lumB),
            0f, 0f,

            lumR + cosA * (-lumR) + sinA * (0.143f),
            lumG + cosA * (1 - lumG) + sinA * (0.140f),
            lumB + cosA * (-lumB) + sinA * (-0.283f),
            0f, 0f,

            lumR + cosA * (-lumR) + sinA * (-(1 - lumR)),
            lumG + cosA * (-lumG) + sinA * (lumG),
            lumB + cosA * (1 - lumB) + sinA * (lumB),
            0f, 0f,

            0f, 0f, 0f, 1f, 0f
        ))

        return applyColorMatrix(bitmap, colorMatrix)
    }

    private fun applyColorMatrix(bitmap: Bitmap, colorMatrix: ColorMatrix): Bitmap {
        val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val paint = Paint().apply {
            colorFilter = ColorMatrixColorFilter(colorMatrix)
        }
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        bitmap.recycle()
        return result
    }
}
