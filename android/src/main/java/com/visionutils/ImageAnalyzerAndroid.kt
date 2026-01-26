package com.visionutils

import android.graphics.Bitmap
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Image analysis utilities for Android
 */
object ImageAnalyzerAndroid {

  /**
   * Get image statistics (mean, std, min, max, histogram)
   */
  fun getStatistics(bitmap: Bitmap): WritableMap {
    val width = bitmap.width
    val height = bitmap.height
    val pixelCount = width * height

    val pixels = IntArray(pixelCount)
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

    // Initialize accumulators
    var rSum = 0.0
    var gSum = 0.0
    var bSum = 0.0
    var rMin = 255
    var gMin = 255
    var bMin = 255
    var rMax = 0
    var gMax = 0
    var bMax = 0

    val rHistogram = IntArray(256)
    val gHistogram = IntArray(256)
    val bHistogram = IntArray(256)

    // First pass: calculate sums, min, max, and histograms
    for (pixel in pixels) {
      val r = (pixel shr 16) and 0xFF
      val g = (pixel shr 8) and 0xFF
      val b = pixel and 0xFF

      rSum += r
      gSum += g
      bSum += b

      rMin = minOf(rMin, r)
      gMin = minOf(gMin, g)
      bMin = minOf(bMin, b)

      rMax = maxOf(rMax, r)
      gMax = maxOf(gMax, g)
      bMax = maxOf(bMax, b)

      rHistogram[r]++
      gHistogram[g]++
      bHistogram[b]++
    }

    val rMean = rSum / pixelCount
    val gMean = gSum / pixelCount
    val bMean = bSum / pixelCount

    // Second pass: calculate standard deviation
    var rVariance = 0.0
    var gVariance = 0.0
    var bVariance = 0.0

    for (pixel in pixels) {
      val r = (pixel shr 16) and 0xFF
      val g = (pixel shr 8) and 0xFF
      val b = pixel and 0xFF

      rVariance += (r - rMean).pow(2)
      gVariance += (g - gMean).pow(2)
      bVariance += (b - bMean).pow(2)
    }

    val rStd = sqrt(rVariance / pixelCount)
    val gStd = sqrt(gVariance / pixelCount)
    val bStd = sqrt(bVariance / pixelCount)

    // Create result
    val result = Arguments.createMap()

    // Mean
    val mean = Arguments.createArray().apply {
      pushDouble(rMean / 255.0)
      pushDouble(gMean / 255.0)
      pushDouble(bMean / 255.0)
    }
    result.putArray("mean", mean)

    // Std
    val std = Arguments.createArray().apply {
      pushDouble(rStd / 255.0)
      pushDouble(gStd / 255.0)
      pushDouble(bStd / 255.0)
    }
    result.putArray("std", std)

    // Min
    val min = Arguments.createArray().apply {
      pushDouble(rMin / 255.0)
      pushDouble(gMin / 255.0)
      pushDouble(bMin / 255.0)
    }
    result.putArray("min", min)

    // Max
    val max = Arguments.createArray().apply {
      pushDouble(rMax / 255.0)
      pushDouble(gMax / 255.0)
      pushDouble(bMax / 255.0)
    }
    result.putArray("max", max)

    // Histograms
    val histograms = Arguments.createMap()

    val rHistArr = Arguments.createArray()
    val gHistArr = Arguments.createArray()
    val bHistArr = Arguments.createArray()

    for (i in 0 until 256) {
      rHistArr.pushInt(rHistogram[i])
      gHistArr.pushInt(gHistogram[i])
      bHistArr.pushInt(bHistogram[i])
    }

    histograms.putArray("r", rHistArr)
    histograms.putArray("g", gHistArr)
    histograms.putArray("b", bHistArr)

    result.putMap("histogram", histograms)

    return result
  }

  /**
   * Get image metadata
   */
  fun getMetadata(bitmap: Bitmap): WritableMap {
    return Arguments.createMap().apply {
      putInt("width", bitmap.width)
      putInt("height", bitmap.height)
      putInt("channels", if (bitmap.hasAlpha()) 4 else 3)
      putString("colorSpace", bitmap.config?.name ?: "unknown")
      putDouble("aspectRatio", bitmap.width.toDouble() / bitmap.height.toDouble())
      putBoolean("hasAlpha", bitmap.hasAlpha())
      putInt("byteCount", bitmap.byteCount)
    }
  }

  /**
   * Validate image against criteria
   */
  fun validate(bitmap: Bitmap, options: ReadableMap): WritableMap {
    val issues = Arguments.createArray()
    var isValid = true

    val width = bitmap.width
    val height = bitmap.height

    // Check minimum dimensions
    if (options.hasKey("minWidth") && width < options.getInt("minWidth")) {
      issues.pushString("Width ${width} is less than minimum ${options.getInt("minWidth")}")
      isValid = false
    }

    if (options.hasKey("minHeight") && height < options.getInt("minHeight")) {
      issues.pushString("Height ${height} is less than minimum ${options.getInt("minHeight")}")
      isValid = false
    }

    // Check maximum dimensions
    if (options.hasKey("maxWidth") && width > options.getInt("maxWidth")) {
      issues.pushString("Width ${width} exceeds maximum ${options.getInt("maxWidth")}")
      isValid = false
    }

    if (options.hasKey("maxHeight") && height > options.getInt("maxHeight")) {
      issues.pushString("Height ${height} exceeds maximum ${options.getInt("maxHeight")}")
      isValid = false
    }

    // Check aspect ratio
    val aspectRatio = width.toDouble() / height.toDouble()

    if (options.hasKey("minAspectRatio") && aspectRatio < options.getDouble("minAspectRatio")) {
      issues.pushString("Aspect ratio $aspectRatio is less than minimum ${options.getDouble("minAspectRatio")}")
      isValid = false
    }

    if (options.hasKey("maxAspectRatio") && aspectRatio > options.getDouble("maxAspectRatio")) {
      issues.pushString("Aspect ratio $aspectRatio exceeds maximum ${options.getDouble("maxAspectRatio")}")
      isValid = false
    }

    // Check required aspect ratio with tolerance
    if (options.hasKey("requiredAspectRatio")) {
      val required = options.getDouble("requiredAspectRatio")
      val tolerance = if (options.hasKey("aspectRatioTolerance"))
        options.getDouble("aspectRatioTolerance")
      else
        0.01

      if (kotlin.math.abs(aspectRatio - required) > tolerance) {
        issues.pushString("Aspect ratio $aspectRatio does not match required $required (tolerance: $tolerance)")
        isValid = false
      }
    }

    // Check channels
    val channels = if (bitmap.hasAlpha()) 4 else 3

    if (options.hasKey("requiredChannels") && channels != options.getInt("requiredChannels")) {
      issues.pushString("Image has $channels channels but $${options.getInt("requiredChannels")} required")
      isValid = false
    }

    return Arguments.createMap().apply {
      putBoolean("isValid", isValid)
      putArray("issues", issues)
      putInt("width", width)
      putInt("height", height)
      putInt("channels", channels)
    }
  }
}
