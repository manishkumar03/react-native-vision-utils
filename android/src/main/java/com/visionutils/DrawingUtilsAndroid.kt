package com.visionutils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.util.Base64
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Utility class for drawing visualizations on images
 */
object DrawingUtilsAndroid {

    // Default color palette for up to 80 classes (COCO-style)
    private val defaultColors = listOf(
        intArrayOf(255, 56, 56), intArrayOf(255, 157, 151), intArrayOf(255, 112, 31), intArrayOf(255, 178, 29),
        intArrayOf(207, 210, 49), intArrayOf(72, 249, 10), intArrayOf(146, 204, 23), intArrayOf(61, 219, 134),
        intArrayOf(26, 147, 52), intArrayOf(0, 212, 187), intArrayOf(44, 153, 168), intArrayOf(0, 194, 255),
        intArrayOf(52, 69, 147), intArrayOf(100, 115, 255), intArrayOf(0, 24, 236), intArrayOf(132, 56, 255),
        intArrayOf(82, 0, 133), intArrayOf(203, 56, 255), intArrayOf(255, 149, 200), intArrayOf(255, 55, 199),
        intArrayOf(255, 99, 99), intArrayOf(255, 173, 173), intArrayOf(255, 155, 85), intArrayOf(255, 198, 94),
        intArrayOf(224, 226, 117), intArrayOf(135, 252, 82), intArrayOf(175, 221, 98), intArrayOf(121, 232, 168),
        intArrayOf(91, 174, 115), intArrayOf(64, 225, 208), intArrayOf(102, 179, 192), intArrayOf(64, 210, 255),
        intArrayOf(103, 121, 176), intArrayOf(144, 156, 255), intArrayOf(64, 84, 244), intArrayOf(166, 114, 255),
        intArrayOf(132, 64, 170), intArrayOf(219, 114, 255), intArrayOf(255, 177, 217), intArrayOf(255, 113, 214)
    )

    private fun getColor(classIndex: Int): IntArray {
        return defaultColors[classIndex % defaultColors.size]
    }

    /**
     * Draw bounding boxes on an image
     */
    fun drawBoxes(
        bitmap: Bitmap,
        boxes: List<Map<String, Any>>,
        lineWidth: Float,
        fontSize: Float,
        drawLabels: Boolean,
        labelBackgroundAlpha: Float,
        labelColor: List<Int>,
        defaultColor: List<Int>?,
        quality: Int
    ): Map<String, Any> {
        val startTime = System.nanoTime()

        // Create mutable copy
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)

        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = lineWidth
            isAntiAlias = true
        }

        val fillPaint = Paint().apply {
            style = Paint.Style.FILL
        }

        val textPaint = Paint().apply {
            textSize = fontSize
            typeface = Typeface.DEFAULT_BOLD
            isAntiAlias = true
            color = Color.rgb(
                if (labelColor.isNotEmpty()) labelColor[0] else 255,
                if (labelColor.size > 1) labelColor[1] else 255,
                if (labelColor.size > 2) labelColor[2] else 255
            )
        }

        var boxesDrawn = 0

        for (boxInfo in boxes) {
            @Suppress("UNCHECKED_CAST")
            val boxArray = boxInfo["box"] as? List<Number> ?: continue
            if (boxArray.size < 4) continue

            val x1 = boxArray[0].toFloat()
            val y1 = boxArray[1].toFloat()
            val x2 = boxArray[2].toFloat()
            val y2 = boxArray[3].toFloat()

            // Get color
            val color: IntArray = when {
                boxInfo["color"] is List<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    val c = boxInfo["color"] as List<Number>
                    if (c.size >= 3) intArrayOf(c[0].toInt(), c[1].toInt(), c[2].toInt())
                    else intArrayOf(255, 0, 0)
                }
                boxInfo["classIndex"] is Number -> {
                    getColor((boxInfo["classIndex"] as Number).toInt())
                }
                defaultColor != null && defaultColor.size >= 3 -> {
                    intArrayOf(defaultColor[0], defaultColor[1], defaultColor[2])
                }
                else -> intArrayOf(255, 0, 0)
            }

            boxPaint.color = Color.rgb(color[0], color[1], color[2])

            // Draw box
            canvas.drawRect(x1, y1, x2, y2, boxPaint)

            // Draw label if provided
            if (drawLabels) {
                var labelText = ""
                if (boxInfo["label"] is String) {
                    labelText = boxInfo["label"] as String
                }
                if (boxInfo["score"] is Number) {
                    val scoreStr = String.format("%.2f", (boxInfo["score"] as Number).toDouble())
                    labelText = if (labelText.isEmpty()) scoreStr else "$labelText $scoreStr"
                }

                if (labelText.isNotEmpty()) {
                    val textBounds = Rect()
                    textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)

                    val labelY = max(0f, y1 - textBounds.height() - 4)
                    val labelRect = RectF(
                        x1,
                        labelY,
                        x1 + textBounds.width() + 8,
                        labelY + textBounds.height() + 4
                    )

                    // Background
                    fillPaint.color = Color.argb(
                        (labelBackgroundAlpha * 255).toInt(),
                        color[0], color[1], color[2]
                    )
                    canvas.drawRect(labelRect, fillPaint)

                    // Text
                    canvas.drawText(labelText, x1 + 4, labelY + textBounds.height(), textPaint)
                }
            }

            boxesDrawn++
        }

        // Convert to base64
        val outputStream = ByteArrayOutputStream()
        outputBitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        val base64String = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)

        val width = outputBitmap.width
        val height = outputBitmap.height
        outputBitmap.recycle()

        val endTime = System.nanoTime()
        val processingTimeMs = (endTime - startTime) / 1_000_000.0

        return mapOf(
            "imageBase64" to base64String,
            "width" to width,
            "height" to height,
            "boxesDrawn" to boxesDrawn,
            "processingTimeMs" to processingTimeMs
        )
    }

    /**
     * Draw keypoints and skeleton on an image
     */
    fun drawKeypoints(
        bitmap: Bitmap,
        keypoints: List<Map<String, Any>>,
        pointRadius: Float,
        pointColors: List<List<Int>>?,
        skeleton: List<Map<String, Any>>?,
        lineWidth: Float,
        minConfidence: Float,
        quality: Int
    ): Map<String, Any> {
        val startTime = System.nanoTime()

        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)

        val linePaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = lineWidth
            isAntiAlias = true
        }

        val circlePaint = Paint().apply {
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        var connectionsDrawn = 0

        // Draw skeleton connections first
        skeleton?.forEach { connection ->
            val fromIdx = (connection["from"] as? Number)?.toInt() ?: return@forEach
            val toIdx = (connection["to"] as? Number)?.toInt() ?: return@forEach

            if (fromIdx >= keypoints.size || toIdx >= keypoints.size) return@forEach

            val fromPoint = keypoints[fromIdx]
            val toPoint = keypoints[toIdx]

            val fromX = (fromPoint["x"] as? Number)?.toFloat() ?: return@forEach
            val fromY = (fromPoint["y"] as? Number)?.toFloat() ?: return@forEach
            val toX = (toPoint["x"] as? Number)?.toFloat() ?: return@forEach
            val toY = (toPoint["y"] as? Number)?.toFloat() ?: return@forEach

            val fromConf = (fromPoint["confidence"] as? Number)?.toFloat() ?: 1f
            val toConf = (toPoint["confidence"] as? Number)?.toFloat() ?: 1f

            if (fromConf < minConfidence || toConf < minConfidence) return@forEach

            // Get connection color
            @Suppress("UNCHECKED_CAST")
            val connColor = connection["color"] as? List<Number>
            linePaint.color = if (connColor != null && connColor.size >= 3) {
                Color.rgb(connColor[0].toInt(), connColor[1].toInt(), connColor[2].toInt())
            } else {
                Color.rgb(0, 255, 0)
            }

            canvas.drawLine(fromX, fromY, toX, toY, linePaint)
            connectionsDrawn++
        }

        // Draw keypoints
        var pointsDrawn = 0
        keypoints.forEachIndexed { idx, point ->
            val x = (point["x"] as? Number)?.toFloat() ?: return@forEachIndexed
            val y = (point["y"] as? Number)?.toFloat() ?: return@forEachIndexed
            val confidence = (point["confidence"] as? Number)?.toFloat() ?: 1f

            if (confidence < minConfidence) return@forEachIndexed

            // Get point color
            val color = if (pointColors != null && idx < pointColors.size && pointColors[idx].size >= 3) {
                val c = pointColors[idx]
                Color.rgb(c[0], c[1], c[2])
            } else {
                val c = getColor(idx)
                Color.rgb(c[0], c[1], c[2])
            }

            circlePaint.color = color
            canvas.drawCircle(x, y, pointRadius, circlePaint)
            pointsDrawn++
        }

        // Convert to base64
        val outputStream = ByteArrayOutputStream()
        outputBitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        val base64String = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)

        val width = outputBitmap.width
        val height = outputBitmap.height
        outputBitmap.recycle()

        val endTime = System.nanoTime()

        return mapOf(
            "imageBase64" to base64String,
            "width" to width,
            "height" to height,
            "pointsDrawn" to pointsDrawn,
            "connectionsDrawn" to connectionsDrawn,
            "processingTimeMs" to (endTime - startTime) / 1_000_000.0
        )
    }

    /**
     * Overlay a segmentation mask on an image
     */
    fun overlayMask(
        bitmap: Bitmap,
        mask: List<Int>,
        maskWidth: Int,
        maskHeight: Int,
        alpha: Float,
        colorMap: List<List<Int>>?,
        singleColor: List<Int>?,
        isClassMask: Boolean,
        quality: Int
    ): Map<String, Any> {
        val startTime = System.nanoTime()

        require(mask.size == maskWidth * maskHeight) { "Mask size doesn't match dimensions" }

        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)

        // Create mask bitmap
        val maskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888)
        val maskPixels = IntArray(maskWidth * maskHeight)

        for (y in 0 until maskHeight) {
            for (x in 0 until maskWidth) {
                val idx = y * maskWidth + x
                val value = mask[idx]

                if (isClassMask) {
                    if (value > 0) {
                        val color = when {
                            colorMap != null && value < colorMap.size -> colorMap[value]
                            singleColor != null && singleColor.size >= 3 -> singleColor
                            else -> getColor(value).toList()
                        }
                        maskPixels[idx] = Color.argb(
                            (alpha * 255).toInt(),
                            color[0], color[1], color[2]
                        )
                    }
                } else {
                    if (value > 0) {
                        val color = singleColor ?: listOf(0, 255, 0)
                        maskPixels[idx] = Color.argb(
                            (alpha * 255 * min(value, 255) / 255f).toInt(),
                            color[0], color[1], color[2]
                        )
                    }
                }
            }
        }

        maskBitmap.setPixels(maskPixels, 0, maskWidth, 0, 0, maskWidth, maskHeight)

        // Scale and draw mask
        val scaledMask = Bitmap.createScaledBitmap(maskBitmap, bitmap.width, bitmap.height, true)
        canvas.drawBitmap(scaledMask, 0f, 0f, null)

        maskBitmap.recycle()
        scaledMask.recycle()

        // Convert to base64
        val outputStream = ByteArrayOutputStream()
        outputBitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        val base64String = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)

        val width = outputBitmap.width
        val height = outputBitmap.height
        outputBitmap.recycle()

        val endTime = System.nanoTime()

        return mapOf(
            "imageBase64" to base64String,
            "width" to width,
            "height" to height,
            "processingTimeMs" to (endTime - startTime) / 1_000_000.0
        )
    }

    /**
     * Overlay a heatmap on an image
     */
    fun overlayHeatmap(
        bitmap: Bitmap,
        heatmap: List<Double>,
        heatmapWidth: Int,
        heatmapHeight: Int,
        alpha: Float,
        colorScheme: String,
        minValue: Double?,
        maxValue: Double?,
        quality: Int
    ): Map<String, Any> {
        val startTime = System.nanoTime()

        require(heatmap.size == heatmapWidth * heatmapHeight) { "Heatmap size doesn't match dimensions" }

        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)

        // Normalize heatmap
        val minVal = minValue ?: heatmap.minOrNull() ?: 0.0
        val maxVal = maxValue ?: heatmap.maxOrNull() ?: 1.0
        val range = maxVal - minVal

        // Create heatmap bitmap
        val heatmapBitmap = Bitmap.createBitmap(heatmapWidth, heatmapHeight, Bitmap.Config.ARGB_8888)
        val heatmapPixels = IntArray(heatmapWidth * heatmapHeight)

        for (y in 0 until heatmapHeight) {
            for (x in 0 until heatmapWidth) {
                val idx = y * heatmapWidth + x
                val normalizedValue = if (range > 0) (heatmap[idx] - minVal) / range else 0.0
                val clampedValue = max(0.0, min(1.0, normalizedValue))

                val (r, g, b) = colorForValue(clampedValue, colorScheme)
                heatmapPixels[idx] = Color.argb((alpha * 255 * clampedValue).toInt(), r, g, b)
            }
        }

        heatmapBitmap.setPixels(heatmapPixels, 0, heatmapWidth, 0, 0, heatmapWidth, heatmapHeight)

        // Scale and draw heatmap
        val scaledHeatmap = Bitmap.createScaledBitmap(heatmapBitmap, bitmap.width, bitmap.height, true)
        canvas.drawBitmap(scaledHeatmap, 0f, 0f, null)

        heatmapBitmap.recycle()
        scaledHeatmap.recycle()

        // Convert to base64
        val outputStream = ByteArrayOutputStream()
        outputBitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        val base64String = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)

        val width = outputBitmap.width
        val height = outputBitmap.height
        outputBitmap.recycle()

        val endTime = System.nanoTime()

        return mapOf(
            "imageBase64" to base64String,
            "width" to width,
            "height" to height,
            "processingTimeMs" to (endTime - startTime) / 1_000_000.0
        )
    }

    /**
     * Get color for a normalized value using a color scheme
     */
    private fun colorForValue(value: Double, scheme: String): Triple<Int, Int, Int> {
        val v = max(0.0, min(1.0, value))

        return when (scheme) {
            "hot" -> {
                when {
                    v < 0.33 -> {
                        val t = v / 0.33
                        Triple((t * 255).toInt(), 0, 0)
                    }
                    v < 0.66 -> {
                        val t = (v - 0.33) / 0.33
                        Triple(255, (t * 255).toInt(), 0)
                    }
                    else -> {
                        val t = (v - 0.66) / 0.34
                        Triple(255, 255, (t * 255).toInt())
                    }
                }
            }
            "viridis" -> {
                when {
                    v < 0.25 -> {
                        val t = v / 0.25
                        Triple(
                            (68 + t * (59 - 68)).toInt(),
                            (1 + t * (82 - 1)).toInt(),
                            (84 + t * (139 - 84)).toInt()
                        )
                    }
                    v < 0.5 -> {
                        val t = (v - 0.25) / 0.25
                        Triple(
                            (59 + t * (33 - 59)).toInt(),
                            (82 + t * (145 - 82)).toInt(),
                            (139 + t * (140 - 139)).toInt()
                        )
                    }
                    v < 0.75 -> {
                        val t = (v - 0.5) / 0.25
                        Triple(
                            (33 + t * (94 - 33)).toInt(),
                            (145 + t * (201 - 145)).toInt(),
                            (140 + t * (98 - 140)).toInt()
                        )
                    }
                    else -> {
                        val t = (v - 0.75) / 0.25
                        Triple(
                            (94 + t * (253 - 94)).toInt(),
                            (201 + t * (231 - 201)).toInt(),
                            (98 + t * (37 - 98)).toInt()
                        )
                    }
                }
            }
            else -> { // "jet"
                when {
                    v < 0.125 -> {
                        val t = v / 0.125
                        Triple(0, 0, (128 + t * 127).toInt())
                    }
                    v < 0.375 -> {
                        val t = (v - 0.125) / 0.25
                        Triple(0, (t * 255).toInt(), 255)
                    }
                    v < 0.625 -> {
                        val t = (v - 0.375) / 0.25
                        Triple((t * 255).toInt(), 255, (255 - t * 255).toInt())
                    }
                    v < 0.875 -> {
                        val t = (v - 0.625) / 0.25
                        Triple(255, (255 - t * 255).toInt(), 0)
                    }
                    else -> {
                        val t = (v - 0.875) / 0.125
                        Triple((255 - t * 127).toInt(), 0, 0)
                    }
                }
            }
        }
    }
}
