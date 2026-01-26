package com.visionutils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.WritableMap
import kotlin.math.max
import kotlin.math.min

/**
 * Blur detection using Laplacian variance method
 */
object BlurDetectorAndroid {

    /**
     * Detect if an image is blurry using Laplacian variance
     * @param bitmap The input bitmap to analyze
     * @param threshold Variance threshold below which image is considered blurry (default: 100)
     * @param downsampleSize Optional max size to downsample to for faster processing
     * @return WritableMap containing isBlurry, score, threshold, and processingTimeMs
     */
    fun detectBlur(
        bitmap: Bitmap,
        threshold: Double = 100.0,
        downsampleSize: Int? = null
    ): WritableMap {
        val startTimeNs = System.nanoTime()

        // Optionally downsample for faster processing
        val processedBitmap = if (downsampleSize != null &&
            (bitmap.width > downsampleSize || bitmap.height > downsampleSize)) {
            downsample(bitmap, downsampleSize)
        } else {
            bitmap
        }

        val width = processedBitmap.width
        val height = processedBitmap.height

        // Convert to grayscale
        val grayscale = convertToGrayscale(processedBitmap)

        // Apply Laplacian and calculate variance
        val laplacianVariance = calculateLaplacianVariance(grayscale, width, height)

        // Clean up if we created a downsampled bitmap
        if (processedBitmap != bitmap) {
            processedBitmap.recycle()
        }

        val processingTimeMs = (System.nanoTime() - startTimeNs) / 1_000_000.0

        return Arguments.createMap().apply {
            putBoolean("isBlurry", laplacianVariance < threshold)
            putDouble("score", laplacianVariance)
            putDouble("threshold", threshold)
            putDouble("processingTimeMs", processingTimeMs)
        }
    }

    /**
     * Downsample bitmap for faster processing
     */
    private fun downsample(bitmap: Bitmap, maxSize: Int): Bitmap {
        val scale = maxSize.toDouble() / max(bitmap.width, bitmap.height)
        val newWidth = (bitmap.width * scale).toInt()
        val newHeight = (bitmap.height * scale).toInt()
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }

    /**
     * Convert bitmap to grayscale float array
     * Using luminance formula: 0.299*R + 0.587*G + 0.114*B
     */
    private fun convertToGrayscale(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixelCount = width * height

        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val grayscale = FloatArray(pixelCount)

        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            grayscale[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }

        return grayscale
    }

    /**
     * Apply Laplacian kernel and calculate variance
     * Laplacian kernel:
     * [0,  1, 0]
     * [1, -4, 1]
     * [0,  1, 0]
     */
    private fun calculateLaplacianVariance(grayscale: FloatArray, width: Int, height: Int): Double {
        val pixelCount = width * height
        val laplacian = FloatArray(pixelCount)

        // Apply Laplacian convolution (skip edges)
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val idx = y * width + x

                val top = grayscale[(y - 1) * width + x]
                val bottom = grayscale[(y + 1) * width + x]
                val left = grayscale[y * width + (x - 1)]
                val right = grayscale[y * width + (x + 1)]
                val center = grayscale[idx]

                // Laplacian = top + bottom + left + right - 4*center
                laplacian[idx] = top + bottom + left + right - 4.0f * center
            }
        }

        // Calculate variance of Laplacian response
        // Variance = E[X^2] - E[X]^2

        // Calculate mean
        var sum = 0.0
        for (value in laplacian) {
            sum += value
        }
        val mean = sum / pixelCount

        // Calculate variance
        var varianceSum = 0.0
        for (value in laplacian) {
            val diff = value - mean
            varianceSum += diff * diff
        }
        val variance = varianceSum / pixelCount

        return variance
    }
}
