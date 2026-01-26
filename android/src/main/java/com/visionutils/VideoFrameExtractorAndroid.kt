package com.visionutils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import com.facebook.react.bridge.WritableNativeArray
import com.facebook.react.bridge.WritableNativeMap
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.math.abs
import kotlin.math.min

/**
 * Extracts frames from video files at specific timestamps or intervals.
 *
 * Supports three extraction modes:
 * - `timestamps`: Array of specific timestamps in seconds
 * - `interval`: Extract every N seconds (with optional startTime/endTime/maxFrames)
 * - `count`: Extract N evenly-spaced frames
 *
 * If no mode is specified, extracts a single frame at t=0.
 *
 * **Supported source types:** `file`, `url` only (asset not supported on Android)
 *
 * **Output formats:**
 * - `base64` (default): JPEG encoded, quality 0-100 (default 90)
 * - `pixelData`: Raw pixel arrays with optional colorFormat/normalization
 *
 * **Note:** `colorFormat` and `normalization` only apply when `outputFormat === "pixelData"`.
 *
 * Per-frame extraction errors are captured in the frame's `error` field; extraction continues for remaining frames.
 */
object VideoFrameExtractorAndroid {

    /**
     * Extract frames from a video file
     *
     * @param source Video source with `type` (file/url) and `value` (path or URL string).
     *               Note: `asset` type is NOT supported on Android.
     * @param options Extraction options:
     *   - `timestamps`: DoubleArray - specific timestamps in seconds
     *   - `interval`: Double - extract every N seconds
     *   - `count`: Int - number of evenly-spaced frames
     *   - `startTime`/`endTime`: Double - range for interval/count modes
     *   - `maxFrames`: Int - limit for interval mode (default 100)
     *   - `resize`: {width: Int, height: Int} - resize frames
     *   - `outputFormat`: "base64" (default) or "pixelData"
     *   - `quality`: Int 0-100 (default 90) - JPEG quality for base64
     *   - `colorFormat`: String - for pixelData (rgb/rgba/bgr/grayscale)
     *   - `normalization`: {preset: String} - for pixelData (scale/imagenet/tensorflow)
     * @return WritableMap with `frames`, `frameCount`, `videoDuration`, `videoWidth`, `videoHeight`, `frameRate`, `processingTimeMs`
     */
    fun extractFrames(source: ReadableMap, options: ReadableMap): WritableMap {
        val startTime = System.currentTimeMillis()
        val result = WritableNativeMap()

        val retriever = MediaMetadataRetriever()

        try {
            // Set data source
            setDataSource(retriever, source)

            // Get video metadata
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val durationSeconds = durationMs / 1000.0

            val videoWidth = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val videoHeight = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val frameRateStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
            val frameRate = frameRateStr?.toFloatOrNull() ?: 30f

            // Determine timestamps to extract
            val timestamps = getTimestamps(options, durationSeconds)

            if (timestamps.isEmpty()) {
                throw Exception("No timestamps to extract")
            }

            // Get output options
            val outputFormat = if (options.hasKey("outputFormat")) options.getString("outputFormat") else "base64"
            val quality = if (options.hasKey("quality")) options.getInt("quality") else 90

            // Get resize options
            var targetWidth: Int? = null
            var targetHeight: Int? = null
            if (options.hasKey("resize")) {
                val resize = options.getMap("resize")
                if (resize != null) {
                    if (resize.hasKey("width")) targetWidth = resize.getInt("width")
                    if (resize.hasKey("height")) targetHeight = resize.getInt("height")
                }
            }

            // Extract frames
            val frames = WritableNativeArray()
            var frameCount = 0

            for (timestamp in timestamps) {
                val frameData = WritableNativeMap()
                frameData.putDouble("requestedTimestamp", timestamp)

                try {
                    // Convert to microseconds
                    val timeUs = (timestamp * 1_000_000).toLong()

                    // Extract frame
                    var bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)

                    if (bitmap != null) {
                        // Resize if needed
                        if (targetWidth != null && targetHeight != null) {
                            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
                            if (scaledBitmap != bitmap) {
                                bitmap.recycle()
                                bitmap = scaledBitmap
                            }
                        }

                        frameData.putDouble("timestamp", timestamp)
                        frameData.putInt("width", bitmap.width)
                        frameData.putInt("height", bitmap.height)

                        when (outputFormat) {
                            "base64" -> {
                                val outputStream = ByteArrayOutputStream()
                                bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
                                val base64 = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
                                frameData.putString("base64", base64)
                            }
                            "pixelData" -> {
                                val pixelData = extractPixelData(bitmap, options)
                                frameData.putArray("data", pixelData.first)
                                frameData.putInt("channels", pixelData.second)
                            }
                        }

                        bitmap.recycle()
                    } else {
                        frameData.putString("error", "Failed to extract frame at timestamp $timestamp")
                    }
                } catch (e: Exception) {
                    frameData.putString("error", e.message ?: "Unknown error")
                }

                frames.pushMap(frameData)
                frameCount++
            }

            val processingTime = System.currentTimeMillis() - startTime

            result.putArray("frames", frames)
            result.putInt("frameCount", frameCount)
            result.putDouble("videoDuration", durationSeconds)
            result.putInt("videoWidth", videoWidth)
            result.putInt("videoHeight", videoHeight)
            result.putDouble("frameRate", frameRate.toDouble())
            result.putDouble("processingTimeMs", processingTime.toDouble())

        } finally {
            retriever.release()
        }

        return result
    }

    /**
     * Set data source based on source type
     */
    private fun setDataSource(retriever: MediaMetadataRetriever, source: ReadableMap) {
        val type = source.getString("type") ?: throw Exception("Source type is required")
        val value = source.getString("value") ?: throw Exception("Source value is required")

        when (type) {
            "file" -> {
                val file = File(value)
                if (!file.exists()) {
                    throw Exception("Video file not found: $value")
                }
                retriever.setDataSource(value)
            }
            "url" -> {
                retriever.setDataSource(value, HashMap())
            }
            else -> {
                throw Exception("Unsupported video source type: $type")
            }
        }
    }

    /**
     * Calculate timestamps to extract based on options
     */
    private fun getTimestamps(options: ReadableMap, duration: Double): List<Double> {
        val timestamps = mutableListOf<Double>()

        // Option 1: Explicit timestamps
        if (options.hasKey("timestamps")) {
            val timestampArray = options.getArray("timestamps")
            if (timestampArray != null) {
                for (i in 0 until timestampArray.size()) {
                    val ts = timestampArray.getDouble(i)
                    if (ts >= 0 && ts <= duration) {
                        timestamps.add(ts)
                    }
                }
            }
        }
        // Option 2: Interval-based extraction
        else if (options.hasKey("interval")) {
            val interval = options.getDouble("interval")
            if (interval > 0) {
                val startTime = if (options.hasKey("startTime")) options.getDouble("startTime") else 0.0
                val endTime = if (options.hasKey("endTime")) options.getDouble("endTime") else duration
                val maxFrames = if (options.hasKey("maxFrames")) options.getInt("maxFrames") else 100

                var currentTime = maxOf(0.0, startTime)
                val effectiveEndTime = minOf(endTime, duration)

                while (currentTime <= effectiveEndTime && timestamps.size < maxFrames) {
                    timestamps.add(currentTime)
                    currentTime += interval
                }
            }
        }
        // Option 3: Count-based extraction (evenly spaced)
        else if (options.hasKey("count")) {
            val count = options.getInt("count")
            if (count > 0) {
                val startTime = if (options.hasKey("startTime")) options.getDouble("startTime") else 0.0
                val endTime = if (options.hasKey("endTime")) options.getDouble("endTime") else duration
                val effectiveEndTime = minOf(endTime, duration)
                val effectiveStartTime = maxOf(0.0, startTime)

                if (count == 1) {
                    timestamps.add((effectiveStartTime + effectiveEndTime) / 2)
                } else {
                    val interval = (effectiveEndTime - effectiveStartTime) / (count - 1)
                    for (i in 0 until count) {
                        timestamps.add(effectiveStartTime + i * interval)
                    }
                }
            }
        }
        // Default: Single frame at 0
        else {
            timestamps.add(0.0)
        }

        return timestamps
    }

    /**
     * Extract pixel data from bitmap
     */
    private fun extractPixelData(bitmap: Bitmap, options: ReadableMap): Pair<WritableArray, Int> {
        val width = bitmap.width
        val height = bitmap.height
        val colorFormat = if (options.hasKey("colorFormat")) options.getString("colorFormat") else "rgb"
        val channels = when {
            colorFormat == "grayscale" -> 1
            colorFormat?.contains("a") == true -> 4
            else -> 3
        }

        // Get pixels
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Get normalization settings
        var mean = floatArrayOf(0f, 0f, 0f)
        var std = floatArrayOf(1f, 1f, 1f)
        var scale = 1f / 255f

        if (options.hasKey("normalization")) {
            val normalization = options.getMap("normalization")
            val preset = normalization?.getString("preset") ?: "scale"

            when (preset) {
                "imagenet" -> {
                    mean = floatArrayOf(0.485f, 0.456f, 0.406f)
                    std = floatArrayOf(0.229f, 0.224f, 0.225f)
                }
                "tensorflow" -> {
                    scale = 2f / 255f
                    mean = floatArrayOf(1f, 1f, 1f)
                }
            }
        }

        val result = WritableNativeArray()

        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            val a = (pixel shr 24) and 0xFF

            if (colorFormat == "grayscale") {
                val gray = 0.299f * r / 255f + 0.587f * g / 255f + 0.114f * b / 255f
                result.pushDouble(gray.toDouble())
            } else {
                val rNorm = (r * scale - mean[0]) / std[0]
                val gNorm = (g * scale - mean[1]) / std[1]
                val bNorm = (b * scale - mean[2]) / std[2]

                if (colorFormat?.startsWith("bgr") == true) {
                    result.pushDouble(bNorm.toDouble())
                    result.pushDouble(gNorm.toDouble())
                    result.pushDouble(rNorm.toDouble())
                } else {
                    result.pushDouble(rNorm.toDouble())
                    result.pushDouble(gNorm.toDouble())
                    result.pushDouble(bNorm.toDouble())
                }

                if (channels == 4) {
                    result.pushDouble(a / 255.0)
                }
            }
        }

        return Pair(result, channels)
    }
}
