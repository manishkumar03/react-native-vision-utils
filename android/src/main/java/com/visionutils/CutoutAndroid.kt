package com.visionutils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import java.io.ByteArrayOutputStream
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * CutoutAndroid provides random erasing augmentation for data augmentation pipelines.
 *
 * Randomly masks rectangular regions of the image to improve model robustness
 * by forcing the model to rely on diverse features rather than specific regions.
 *
 * Supports:
 * - Single or multiple cutouts
 * - Configurable size and aspect ratio ranges
 * - Constant color, noise, or random color fill modes
 * - Probability-based application
 * - Reproducible results via seed
 *
 * Example Usage (from JS):
 * ```typescript
 * const result = await cutout(
 *   { uri: 'file://image.jpg' },
 *   {
 *     numCutouts: 2,
 *     minSize: 0.02,
 *     maxSize: 0.33,
 *     fillMode: 'noise',
 *     seed: 42
 *   }
 * );
 * ```
 */
object CutoutAndroid {

    /**
     * Apply cutout augmentation to an image.
     *
     * @param bitmap Source bitmap
     * @param options Cutout options
     * @return WritableMap containing augmented image and cutout details
     */
    fun apply(bitmap: Bitmap, options: ReadableMap): WritableMap {
        val startTimeNs = System.nanoTime()

        // Parse options
        val numCutouts = if (options.hasKey("numCutouts") && !options.isNull("numCutouts")) {
            options.getInt("numCutouts")
        } else 1

        val minSize = if (options.hasKey("minSize") && !options.isNull("minSize")) {
            options.getDouble("minSize")
        } else 0.02

        val maxSize = if (options.hasKey("maxSize") && !options.isNull("maxSize")) {
            options.getDouble("maxSize")
        } else 0.33

        val minAspect = if (options.hasKey("minAspect") && !options.isNull("minAspect")) {
            options.getDouble("minAspect")
        } else 0.3

        val maxAspect = if (options.hasKey("maxAspect") && !options.isNull("maxAspect")) {
            options.getDouble("maxAspect")
        } else 3.3

        val fillMode = if (options.hasKey("fillMode") && !options.isNull("fillMode")) {
            options.getString("fillMode") ?: "constant"
        } else "constant"

        val fillValue = parseFillValue(options)

        val probability = if (options.hasKey("probability") && !options.isNull("probability")) {
            options.getDouble("probability")
        } else 1.0

        // Setup random number generator
        val seed = if (options.hasKey("seed") && !options.isNull("seed")) {
            options.getInt("seed")
        } else {
            Random.nextInt(Int.MAX_VALUE)
        }
        val random = Random(seed.toLong())

        val width = bitmap.width
        val height = bitmap.height
        val imageArea = (width * height).toDouble()

        val regions = Arguments.createArray()

        // Check probability
        val shouldApply = random.nextDouble() < probability

        if (!shouldApply || numCutouts == 0) {
            // Return original image without cutouts
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
            val base64String = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
            val processingTimeMs = (System.nanoTime() - startTimeNs) / 1_000_000.0

            return Arguments.createMap().apply {
                putString("base64", base64String)
                putInt("width", width)
                putInt("height", height)
                putBoolean("applied", false)
                putInt("numCutouts", 0)
                putArray("regions", regions)
                putInt("seed", seed)
                putDouble("processingTimeMs", processingTimeMs)
            }
        }

        // Create mutable bitmap
        val resultBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBitmap)
        val paint = Paint()

        // Apply cutouts
        for (i in 0 until numCutouts) {
            // Random area within range
            val targetArea = (random.nextDouble() * (maxSize - minSize) + minSize) * imageArea

            // Random aspect ratio
            val aspectRatio = random.nextDouble() * (maxAspect - minAspect) + minAspect

            // Calculate dimensions
            var cutoutWidth = sqrt(targetArea * aspectRatio).toInt()
            var cutoutHeight = sqrt(targetArea / aspectRatio).toInt()

            // Clamp to image bounds
            cutoutWidth = min(cutoutWidth, width)
            cutoutHeight = min(cutoutHeight, height)

            if (cutoutWidth <= 0 || cutoutHeight <= 0) continue

            // Random position
            val x = if (width > cutoutWidth) random.nextInt(width - cutoutWidth) else 0
            val y = if (height > cutoutHeight) random.nextInt(height - cutoutHeight) else 0

            // Apply fill based on mode
            val regionInfo = Arguments.createMap()
            regionInfo.putInt("x", x)
            regionInfo.putInt("y", y)
            regionInfo.putInt("width", cutoutWidth)
            regionInfo.putInt("height", cutoutHeight)

            when (fillMode) {
                "noise" -> {
                    // Fill with noise
                    fillWithNoise(resultBitmap, x, y, cutoutWidth, cutoutHeight, random)
                    regionInfo.putString("fill", "noise")
                }
                "random" -> {
                    // Random constant color
                    val r = random.nextInt(256)
                    val g = random.nextInt(256)
                    val b = random.nextInt(256)
                    paint.color = Color.rgb(r, g, b)
                    canvas.drawRect(
                        x.toFloat(),
                        y.toFloat(),
                        (x + cutoutWidth).toFloat(),
                        (y + cutoutHeight).toFloat(),
                        paint
                    )
                    val fillArray = Arguments.createArray()
                    fillArray.pushInt(r)
                    fillArray.pushInt(g)
                    fillArray.pushInt(b)
                    regionInfo.putArray("fill", fillArray)
                }
                else -> {
                    // Constant fill
                    paint.color = Color.rgb(fillValue[0], fillValue[1], fillValue[2])
                    canvas.drawRect(
                        x.toFloat(),
                        y.toFloat(),
                        (x + cutoutWidth).toFloat(),
                        (y + cutoutHeight).toFloat(),
                        paint
                    )
                    val fillArray = Arguments.createArray()
                    fillArray.pushInt(fillValue[0])
                    fillArray.pushInt(fillValue[1])
                    fillArray.pushInt(fillValue[2])
                    regionInfo.putArray("fill", fillArray)
                }
            }

            regions.pushMap(regionInfo)
        }

        // Encode to base64
        val outputStream = ByteArrayOutputStream()
        resultBitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
        val base64String = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)

        val processingTimeMs = (System.nanoTime() - startTimeNs) / 1_000_000.0

        resultBitmap.recycle()

        return Arguments.createMap().apply {
            putString("base64", base64String)
            putInt("width", width)
            putInt("height", height)
            putBoolean("applied", true)
            putInt("numCutouts", regions.size())
            putArray("regions", regions)
            putInt("seed", seed)
            putDouble("processingTimeMs", processingTimeMs)
        }
    }

    private fun parseFillValue(options: ReadableMap): IntArray {
        if (options.hasKey("fillValue") && !options.isNull("fillValue")) {
            try {
                val array = options.getArray("fillValue")
                if (array != null && array.size() >= 3) {
                    return intArrayOf(
                        array.getInt(0),
                        array.getInt(1),
                        array.getInt(2)
                    )
                }
            } catch (_: Exception) {
                // Fall through to default
            }
        }
        return intArrayOf(0, 0, 0) // Default black
    }

    private fun fillWithNoise(
        bitmap: Bitmap,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        random: Random
    ) {
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val r = random.nextInt(256)
            val g = random.nextInt(256)
            val b = random.nextInt(256)
            pixels[i] = Color.rgb(r, g, b)
        }
        bitmap.setPixels(pixels, 0, width, x, y, width, height)
    }
}
