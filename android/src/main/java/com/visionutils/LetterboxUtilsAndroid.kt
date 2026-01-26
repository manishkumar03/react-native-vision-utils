package com.visionutils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Base64
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Utility class for letterbox padding operations
 */
object LetterboxUtilsAndroid {

    /**
     * Apply letterbox padding to an image
     */
    fun letterbox(
        bitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        padColor: List<Int>,
        scaleUp: Boolean,
        autoStride: Boolean,
        stride: Int,
        center: Boolean
    ): Map<String, Any> {
        val startTime = System.nanoTime()

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        // Calculate scale factor
        val scaleW = targetWidth.toDouble() / originalWidth
        val scaleH = targetHeight.toDouble() / originalHeight
        var scale = min(scaleW, scaleH)

        // Don't scale up if not allowed
        if (!scaleUp) {
            scale = min(scale, 1.0)
        }

        // Calculate new dimensions
        var newWidth = (originalWidth * scale).roundToInt()
        var newHeight = (originalHeight * scale).roundToInt()

        // Apply stride alignment if requested
        if (autoStride) {
            newWidth = ((newWidth + stride - 1) / stride) * stride
            newHeight = ((newHeight + stride - 1) / stride) * stride
        }

        // Calculate padding
        val padW = targetWidth - newWidth
        val padH = targetHeight - newHeight

        val padLeft: Int
        val padTop: Int
        val padRight: Int
        val padBottom: Int

        if (center) {
            padLeft = padW / 2
            padTop = padH / 2
            padRight = padW - padLeft
            padBottom = padH - padTop
        } else {
            padLeft = 0
            padTop = 0
            padRight = padW
            padBottom = padH
        }

        // Create output bitmap
        val outputBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)

        // Fill with pad color
        val r = if (padColor.isNotEmpty()) padColor[0] else 114
        val g = if (padColor.size > 1) padColor[1] else 114
        val b = if (padColor.size > 2) padColor[2] else 114
        canvas.drawColor(Color.rgb(r, g, b))

        // Scale and draw the image
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        canvas.drawBitmap(scaledBitmap, padLeft.toFloat(), padTop.toFloat(), null)
        scaledBitmap.recycle()

        // Convert to base64
        val outputStream = ByteArrayOutputStream()
        outputBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
        val base64String = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
        outputBitmap.recycle()

        // Create letterbox info for reverse transformation
        val letterboxInfo = mapOf(
            "scale" to scale,
            "padding" to listOf(padLeft, padTop, padRight, padBottom),
            "offset" to listOf(padLeft, padTop),
            "originalSize" to listOf(originalWidth, originalHeight),
            "letterboxedSize" to listOf(targetWidth, targetHeight)
        )

        val endTime = System.nanoTime()
        val processingTimeMs = (endTime - startTime) / 1_000_000.0

        return mapOf(
            "imageBase64" to base64String,
            "width" to targetWidth,
            "height" to targetHeight,
            "scale" to scale,
            "padding" to listOf(padLeft, padTop, padRight, padBottom),
            "offset" to listOf(padLeft, padTop),
            "originalSize" to listOf(originalWidth, originalHeight),
            "letterboxInfo" to letterboxInfo,
            "processingTimeMs" to processingTimeMs
        )
    }

    /**
     * Reverse letterbox transformation on bounding boxes
     */
    fun reverseLetterbox(
        boxes: List<List<Double>>,
        scale: Double,
        padding: List<Int>,
        originalWidth: Int,
        originalHeight: Int,
        format: String,
        clip: Boolean
    ): List<List<Double>> {
        val padLeft = if (padding.isNotEmpty()) padding[0].toDouble() else 0.0
        val padTop = if (padding.size > 1) padding[1].toDouble() else 0.0

        // Convert to xyxy for transformation
        var xyxyBoxes = if (format != "xyxy") {
            BoundingBoxUtilsAndroid.convertBoxFormat(boxes, format, "xyxy")
        } else {
            boxes
        }

        // Apply reverse transformation
        var transformedBoxes = xyxyBoxes.map { box ->
            if (box.size != 4) return@map box
            listOf(
                (box[0] - padLeft) / scale,
                (box[1] - padTop) / scale,
                (box[2] - padLeft) / scale,
                (box[3] - padTop) / scale
            )
        }

        // Clip if requested
        if (clip) {
            transformedBoxes = transformedBoxes.map { box ->
                if (box.size != 4) return@map box
                listOf(
                    max(0.0, min(originalWidth.toDouble(), box[0])),
                    max(0.0, min(originalHeight.toDouble(), box[1])),
                    max(0.0, min(originalWidth.toDouble(), box[2])),
                    max(0.0, min(originalHeight.toDouble(), box[3]))
                )
            }
        }

        // Convert back to original format if needed
        return if (format != "xyxy") {
            BoundingBoxUtilsAndroid.convertBoxFormat(transformedBoxes, "xyxy", format)
        } else {
            transformedBoxes
        }
    }
}
