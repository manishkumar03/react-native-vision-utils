package com.visionutils

import android.graphics.Bitmap
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import java.io.ByteArrayOutputStream
import kotlin.math.roundToInt

/**
 * Convert tensor data back to images
 */
object TensorConverterAndroid {

  /**
   * Convert tensor data to a base64 encoded image
   */
  fun tensorToImage(
    data: FloatArray,
    width: Int,
    height: Int,
    options: ReadableMap
  ): WritableMap {
    val channels = if (options.hasKey("channels")) options.getInt("channels") else 3
    val format = if (options.hasKey("format")) options.getString("format") else "png"
    val quality = if (options.hasKey("quality")) options.getInt("quality") else 100

    // Denormalization parameters
    val denormalize = options.hasKey("denormalize") && options.getBoolean("denormalize")
    var mean = floatArrayOf(0f, 0f, 0f)
    var std = floatArrayOf(1f, 1f, 1f)

    if (denormalize) {
      if (options.hasKey("mean")) {
        val meanArr = options.getArray("mean")
        if (meanArr != null) {
          mean = FloatArray(meanArr.size()) { meanArr.getDouble(it).toFloat() }
        }
      }
      if (options.hasKey("std")) {
        val stdArr = options.getArray("std")
        if (stdArr != null) {
          std = FloatArray(stdArr.size()) { stdArr.getDouble(it).toFloat() }
        }
      }
    }

    // Create bitmap
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val pixels = IntArray(width * height)

    val dataLayout = if (options.hasKey("dataLayout")) options.getString("dataLayout") else "hwc"

    for (y in 0 until height) {
      for (x in 0 until width) {
        val pixelIdx = y * width + x

        val r: Float
        val g: Float
        val b: Float
        val a: Float

        when (dataLayout) {
          "hwc" -> {
            val offset = pixelIdx * channels
            r = data[offset]
            g = if (channels > 1) data[offset + 1] else r
            b = if (channels > 2) data[offset + 2] else r
            a = if (channels > 3) data[offset + 3] else 1f
          }
          "chw" -> {
            val pixelCount = width * height
            r = data[pixelIdx]
            g = if (channels > 1) data[pixelCount + pixelIdx] else r
            b = if (channels > 2) data[2 * pixelCount + pixelIdx] else r
            a = if (channels > 3) data[3 * pixelCount + pixelIdx] else 1f
          }
          else -> {
            throw VisionUtilsException("INVALID_LAYOUT", "Unknown data layout: $dataLayout")
          }
        }

        // Denormalize if needed
        var rVal = r
        var gVal = g
        var bVal = b

        if (denormalize) {
          rVal = r * std[0] + mean[0]
          gVal = g * std[minOf(1, std.size - 1)] + mean[minOf(1, mean.size - 1)]
          bVal = b * std[minOf(2, std.size - 1)] + mean[minOf(2, mean.size - 1)]
        }

        // Clamp to [0, 1] and convert to [0, 255]
        val rInt = (rVal.coerceIn(0f, 1f) * 255).roundToInt()
        val gInt = (gVal.coerceIn(0f, 1f) * 255).roundToInt()
        val bInt = (bVal.coerceIn(0f, 1f) * 255).roundToInt()
        val aInt = (a.coerceIn(0f, 1f) * 255).roundToInt()

        pixels[pixelIdx] = (aInt shl 24) or (rInt shl 16) or (gInt shl 8) or bInt
      }
    }

    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

    // Encode to base64
    val outputStream = ByteArrayOutputStream()
    val compressFormat = when (format?.lowercase()) {
      "jpeg", "jpg" -> Bitmap.CompressFormat.JPEG
      "webp" -> Bitmap.CompressFormat.WEBP
      else -> Bitmap.CompressFormat.PNG
    }

    bitmap.compress(compressFormat, quality, outputStream)
    val base64String = Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)

    bitmap.recycle()

    return Arguments.createMap().apply {
      putString("base64", base64String)
      putInt("width", width)
      putInt("height", height)
      putString("format", format ?: "png")
    }
  }
}
