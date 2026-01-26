package com.visionutils

import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import kotlin.math.max
import kotlin.math.min

/**
 * High-performance camera frame processor for ML preprocessing on Android
 */
object CameraFrameAndroid {

    /**
     * Process camera frame buffer into ML-ready tensor
     */
    fun processCameraFrame(source: ReadableMap, options: ReadableMap): WritableMap {
        val startTime = System.nanoTime()

        val width = if (source.hasKey("width")) source.getInt("width") else 0
        val height = if (source.hasKey("height")) source.getInt("height") else 0
        val pixelFormat = source.getString("pixelFormat") ?: "yuv420"
        val bytesPerRow = if (source.hasKey("bytesPerRow")) source.getInt("bytesPerRow") else width * 4
        val orientation = if (source.hasKey("orientation")) source.getInt("orientation") else 0

        // Extract options
        val outputWidth = if (options.hasKey("outputWidth")) options.getInt("outputWidth") else width
        val outputHeight = if (options.hasKey("outputHeight")) options.getInt("outputHeight") else height
        val normalize = if (options.hasKey("normalize")) options.getBoolean("normalize") else true
        val outputFormat = options.getString("outputFormat") ?: "rgb"

        val meanValues = if (options.hasKey("mean")) {
            val meanArray = options.getArray("mean")!!
            DoubleArray(meanArray.size()) { meanArray.getDouble(it) }
        } else {
            doubleArrayOf(0.0, 0.0, 0.0)
        }

        val stdValues = if (options.hasKey("std")) {
            val stdArray = options.getArray("std")!!
            DoubleArray(stdArray.size()) { stdArray.getDouble(it) }
        } else {
            doubleArrayOf(1.0, 1.0, 1.0)
        }

        // Get input data
        val inputData = if (source.hasKey("dataBase64")) {
            Base64.decode(source.getString("dataBase64"), Base64.DEFAULT)
        } else {
            throw VisionUtilsException("INVALID_SOURCE", "Base64 data required for processCameraFrame")
        }

        // Convert based on pixel format
        var rgbData = when (pixelFormat.lowercase()) {
            "yuv420", "yuv420f", "420f" -> convertYUV420ToRGB(inputData, width, height, bytesPerRow)
            "nv12" -> convertNV12ToRGB(inputData, width, height, bytesPerRow)
            "nv21" -> convertNV21ToRGB(inputData, width, height, bytesPerRow)
            "bgra" -> convertBGRAToRGB(inputData, width, height, bytesPerRow)
            "rgba" -> convertRGBAToRGB(inputData, width, height, bytesPerRow)
            "rgb" -> extractRGB(inputData, width, height, bytesPerRow)
            else -> throw VisionUtilsException("INVALID_FORMAT", "Unsupported pixel format: $pixelFormat")
        }

        // Apply rotation if needed
        var currentWidth = width
        var currentHeight = height
        if (orientation != 0) {
            val rotatedResult = rotateRGB(rgbData, currentWidth, currentHeight, orientation)
            rgbData = rotatedResult.first
            if (orientation == 2 || orientation == 3) {
                currentWidth = height
                currentHeight = width
            }
        }

        // Resize if needed
        if (currentWidth != outputWidth || currentHeight != outputHeight) {
            rgbData = resizeRGB(rgbData, currentWidth, currentHeight, outputWidth, outputHeight)
        }

        // Normalize to float tensor
        val channelCount = if (outputFormat == "grayscale") 1 else 3

        var tensorData: DoubleArray = if (outputFormat == "grayscale") {
            rgbToGrayscale(rgbData, outputWidth, outputHeight)
        } else {
            DoubleArray(rgbData.size) { rgbData[it].toDouble() and 0xFF.toDouble() }
        }

        // Apply normalization
        if (normalize) {
            tensorData = applyNormalization(tensorData, meanValues, stdValues, channelCount)
        }

        val processingTime = (System.nanoTime() - startTime) / 1_000_000.0

        return Arguments.createMap().apply {
            val tensorArray = Arguments.createArray()
            tensorData.forEach { tensorArray.pushDouble(it) }
            putArray("tensor", tensorArray)

            val shapeArray = Arguments.createArray()
            shapeArray.pushInt(outputHeight)
            shapeArray.pushInt(outputWidth)
            shapeArray.pushInt(channelCount)
            putArray("shape", shapeArray)

            putInt("width", outputWidth)
            putInt("height", outputHeight)
            putDouble("processingTimeMs", processingTime)
        }
    }

    /**
     * Direct YUV to RGB conversion
     */
    fun convertYUVToRGB(options: ReadableMap): WritableMap {
        val startTime = System.nanoTime()

        val width = if (options.hasKey("width")) options.getInt("width") else 0
        val height = if (options.hasKey("height")) options.getInt("height") else 0
        val pixelFormat = options.getString("pixelFormat") ?: "yuv420"
        val outputFormat = options.getString("outputFormat") ?: "rgb"

        // Check for base64 input (separate planes)
        if (options.hasKey("yPlaneBase64") && options.hasKey("uPlaneBase64") && options.hasKey("vPlaneBase64")) {
            val yData = Base64.decode(options.getString("yPlaneBase64"), Base64.DEFAULT)
            val uData = Base64.decode(options.getString("uPlaneBase64"), Base64.DEFAULT)
            val vData = Base64.decode(options.getString("vPlaneBase64"), Base64.DEFAULT)

            val rgbData = convertYUVPlanesToRGB(yData, uData, vData, width, height)

            val processingTime = (System.nanoTime() - startTime) / 1_000_000.0

            return Arguments.createMap().apply {
                if (outputFormat == "base64") {
                    putString("dataBase64", Base64.encodeToString(rgbData, Base64.DEFAULT))
                } else {
                    val dataArray = Arguments.createArray()
                    rgbData.forEach { dataArray.pushDouble((it.toInt() and 0xFF).toDouble()) }
                    putArray("data", dataArray)
                }
                putInt("width", width)
                putInt("height", height)
                putInt("channels", 3)
                putDouble("processingTimeMs", processingTime)
            }
        }

        throw VisionUtilsException("INVALID_DATA", "YUV plane data (yPlaneBase64, uPlaneBase64, vPlaneBase64) required")
    }

    // MARK: - Private Conversion Methods

    private fun convertYUV420ToRGB(data: ByteArray, width: Int, height: Int, bytesPerRow: Int): ByteArray {
        val yPlaneSize = bytesPerRow * height
        val uvPlaneSize = (bytesPerRow / 2) * (height / 2)

        val rgbData = ByteArray(width * height * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val yIndex = y * bytesPerRow + x
                val uvIndex = yPlaneSize + (y / 2) * (bytesPerRow / 2) + (x / 2)
                val vIndex = yPlaneSize + uvPlaneSize + (y / 2) * (bytesPerRow / 2) + (x / 2)

                val yValue = (data[yIndex].toInt() and 0xFF)
                val uValue = (data[uvIndex].toInt() and 0xFF) - 128
                val vValue = (data[vIndex].toInt() and 0xFF) - 128

                // YUV to RGB conversion (BT.601)
                var r = yValue + (351 * vValue) / 256
                var g = yValue - (86 * uValue + 179 * vValue) / 256
                var b = yValue + (444 * uValue) / 256

                // Clamp values
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                val rgbIndex = (y * width + x) * 3
                rgbData[rgbIndex] = r.toByte()
                rgbData[rgbIndex + 1] = g.toByte()
                rgbData[rgbIndex + 2] = b.toByte()
            }
        }

        return rgbData
    }

    private fun convertNV12ToRGB(data: ByteArray, width: Int, height: Int, bytesPerRow: Int): ByteArray {
        val yPlaneSize = bytesPerRow * height
        val rgbData = ByteArray(width * height * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val yIndex = y * bytesPerRow + x
                val uvIndex = yPlaneSize + (y / 2) * bytesPerRow + (x / 2) * 2

                val yValue = (data[yIndex].toInt() and 0xFF)
                val uValue = (data[uvIndex].toInt() and 0xFF) - 128
                val vValue = (data[uvIndex + 1].toInt() and 0xFF) - 128

                var r = yValue + (351 * vValue) / 256
                var g = yValue - (86 * uValue + 179 * vValue) / 256
                var b = yValue + (444 * uValue) / 256

                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                val rgbIndex = (y * width + x) * 3
                rgbData[rgbIndex] = r.toByte()
                rgbData[rgbIndex + 1] = g.toByte()
                rgbData[rgbIndex + 2] = b.toByte()
            }
        }

        return rgbData
    }

    private fun convertNV21ToRGB(data: ByteArray, width: Int, height: Int, bytesPerRow: Int): ByteArray {
        val yPlaneSize = bytesPerRow * height
        val rgbData = ByteArray(width * height * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val yIndex = y * bytesPerRow + x
                val vuIndex = yPlaneSize + (y / 2) * bytesPerRow + (x / 2) * 2

                val yValue = (data[yIndex].toInt() and 0xFF)
                val vValue = (data[vuIndex].toInt() and 0xFF) - 128
                val uValue = (data[vuIndex + 1].toInt() and 0xFF) - 128

                var r = yValue + (351 * vValue) / 256
                var g = yValue - (86 * uValue + 179 * vValue) / 256
                var b = yValue + (444 * uValue) / 256

                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                val rgbIndex = (y * width + x) * 3
                rgbData[rgbIndex] = r.toByte()
                rgbData[rgbIndex + 1] = g.toByte()
                rgbData[rgbIndex + 2] = b.toByte()
            }
        }

        return rgbData
    }

    private fun convertBGRAToRGB(data: ByteArray, width: Int, height: Int, bytesPerRow: Int): ByteArray {
        val rgbData = ByteArray(width * height * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val bgraIndex = y * bytesPerRow + x * 4
                val rgbIndex = (y * width + x) * 3

                rgbData[rgbIndex] = data[bgraIndex + 2]     // R
                rgbData[rgbIndex + 1] = data[bgraIndex + 1] // G
                rgbData[rgbIndex + 2] = data[bgraIndex]     // B
            }
        }

        return rgbData
    }

    private fun convertRGBAToRGB(data: ByteArray, width: Int, height: Int, bytesPerRow: Int): ByteArray {
        val rgbData = ByteArray(width * height * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val rgbaIndex = y * bytesPerRow + x * 4
                val rgbIndex = (y * width + x) * 3

                rgbData[rgbIndex] = data[rgbaIndex]
                rgbData[rgbIndex + 1] = data[rgbaIndex + 1]
                rgbData[rgbIndex + 2] = data[rgbaIndex + 2]
            }
        }

        return rgbData
    }

    private fun extractRGB(data: ByteArray, width: Int, height: Int, bytesPerRow: Int): ByteArray {
        val rgbData = ByteArray(width * height * 3)

        for (y in 0 until height) {
            val srcRow = y * bytesPerRow
            val dstRow = y * width * 3
            for (x in 0 until (width * 3)) {
                rgbData[dstRow + x] = data[srcRow + x]
            }
        }

        return rgbData
    }

    private fun convertYUVPlanesToRGB(yPlane: ByteArray, uPlane: ByteArray, vPlane: ByteArray, width: Int, height: Int): ByteArray {
        val rgbData = ByteArray(width * height * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val yIndex = y * width + x
                val uvIndex = (y / 2) * (width / 2) + (x / 2)

                val yValue = (yPlane[yIndex].toInt() and 0xFF)
                val uValue = (uPlane[uvIndex].toInt() and 0xFF) - 128
                val vValue = (vPlane[uvIndex].toInt() and 0xFF) - 128

                var r = yValue + (351 * vValue) / 256
                var g = yValue - (86 * uValue + 179 * vValue) / 256
                var b = yValue + (444 * uValue) / 256

                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                val rgbIndex = (y * width + x) * 3
                rgbData[rgbIndex] = r.toByte()
                rgbData[rgbIndex + 1] = g.toByte()
                rgbData[rgbIndex + 2] = b.toByte()
            }
        }

        return rgbData
    }

    // MARK: - Image Processing Helpers

    private fun rotateRGB(data: ByteArray, width: Int, height: Int, orientation: Int): Pair<ByteArray, Pair<Int, Int>> {
        val newWidth = if (orientation == 2 || orientation == 3) height else width
        val newHeight = if (orientation == 2 || orientation == 3) width else height
        val rotated = ByteArray(newWidth * newHeight * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val srcIndex = (y * width + x) * 3

                val (dstX, dstY) = when (orientation) {
                    1 -> Pair(width - 1 - x, height - 1 - y)  // 180 degrees
                    2 -> Pair(height - 1 - y, x)               // 90 degrees clockwise
                    3 -> Pair(y, width - 1 - x)                // 90 degrees counter-clockwise
                    else -> Pair(x, y)
                }

                val dstIndex = (dstY * newWidth + dstX) * 3
                rotated[dstIndex] = data[srcIndex]
                rotated[dstIndex + 1] = data[srcIndex + 1]
                rotated[dstIndex + 2] = data[srcIndex + 2]
            }
        }

        return Pair(rotated, Pair(newWidth, newHeight))
    }

    private fun resizeRGB(data: ByteArray, fromWidth: Int, fromHeight: Int, toWidth: Int, toHeight: Int): ByteArray {
        val resized = ByteArray(toWidth * toHeight * 3)

        val xScale = fromWidth.toDouble() / toWidth
        val yScale = fromHeight.toDouble() / toHeight

        for (y in 0 until toHeight) {
            for (x in 0 until toWidth) {
                val srcX = min((x * xScale).toInt(), fromWidth - 1)
                val srcY = min((y * yScale).toInt(), fromHeight - 1)

                val srcIndex = (srcY * fromWidth + srcX) * 3
                val dstIndex = (y * toWidth + x) * 3

                resized[dstIndex] = data[srcIndex]
                resized[dstIndex + 1] = data[srcIndex + 1]
                resized[dstIndex + 2] = data[srcIndex + 2]
            }
        }

        return resized
    }

    private fun rgbToGrayscale(rgbData: ByteArray, width: Int, height: Int): DoubleArray {
        val grayscale = DoubleArray(width * height)

        for (i in 0 until (width * height)) {
            val r = (rgbData[i * 3].toInt() and 0xFF).toDouble()
            val g = (rgbData[i * 3 + 1].toInt() and 0xFF).toDouble()
            val b = (rgbData[i * 3 + 2].toInt() and 0xFF).toDouble()

            // Standard grayscale conversion
            grayscale[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        return grayscale
    }

    private fun applyNormalization(data: DoubleArray, mean: DoubleArray, std: DoubleArray, channels: Int): DoubleArray {
        val normalized = DoubleArray(data.size)
        val pixelCount = data.size / channels

        for (i in 0 until pixelCount) {
            for (c in 0 until channels) {
                val index = i * channels + c
                val meanVal = if (c < mean.size) mean[c] else 0.0
                val stdVal = if (c < std.size) std[c] else 1.0

                // Normalize: (value / 255.0 - mean) / std
                normalized[index] = (data[index] / 255.0 - meanVal) / stdVal
            }
        }

        return normalized
    }
}
