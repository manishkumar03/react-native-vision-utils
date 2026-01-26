package com.visionutils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicBlur
import android.util.Base64
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import java.io.ByteArrayOutputStream
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Image augmentation operations for Android
 */
object ImageAugmenterAndroid {

  /**
   * Apply augmentations to an image
   */
  fun apply(bitmap: Bitmap, augmentations: ReadableMap): WritableMap {
    // Track processing time for parity with iOS
    val startTimeNs = System.nanoTime()

    var result = bitmap.copy(Bitmap.Config.ARGB_8888, true)

    // Rotation
    if (augmentations.hasKey("rotation")) {
      val rotation = augmentations.getDouble("rotation").toFloat()
      if (rotation != 0f) {
        result = rotate(result, rotation)
      }
    }

    // Flip horizontal
    if (augmentations.hasKey("flipHorizontal") && augmentations.getBoolean("flipHorizontal")) {
      result = flipHorizontal(result)
    }

    // Flip vertical
    if (augmentations.hasKey("flipVertical") && augmentations.getBoolean("flipVertical")) {
      result = flipVertical(result)
    }

    // Brightness
    if (augmentations.hasKey("brightness")) {
      val brightness = augmentations.getDouble("brightness").toFloat()
      if (brightness != 1f) {
        result = adjustBrightness(result, brightness)
      }
    }

    // Contrast
    if (augmentations.hasKey("contrast")) {
      val contrast = augmentations.getDouble("contrast").toFloat()
      if (contrast != 1f) {
        result = adjustContrast(result, contrast)
      }
    }

    // Saturation
    if (augmentations.hasKey("saturation")) {
      val saturation = augmentations.getDouble("saturation").toFloat()
      if (saturation != 1f) {
        result = adjustSaturation(result, saturation)
      }
    }

    // Blur
    if (augmentations.hasKey("blur")) {
      val blur = augmentations.getDouble("blur").toFloat()
      if (blur > 0f) {
        // Note: RenderScript blur requires Android API level check
        // For simplicity, we use a basic blur implementation
        result = applyBasicBlur(result, blur)
      }
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
      putDouble("processingTimeMs", processingTimeMs)
    }
  }

  private fun rotate(bitmap: Bitmap, degrees: Float): Bitmap {
    val matrix = Matrix().apply {
      postRotate(degrees, bitmap.width / 2f, bitmap.height / 2f)
    }

    // Calculate new dimensions
    val radians = Math.toRadians(degrees.toDouble())
    val sin = abs(sin(radians))
    val cos = abs(cos(radians))
    val newWidth = (bitmap.width * cos + bitmap.height * sin).toInt()
    val newHeight = (bitmap.width * sin + bitmap.height * cos).toInt()

    val result = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(result)

    // Translate to center the rotated image
    canvas.translate((newWidth - bitmap.width) / 2f, (newHeight - bitmap.height) / 2f)
    canvas.concat(matrix)
    canvas.drawBitmap(bitmap, 0f, 0f, null)

    bitmap.recycle()
    return result
  }

  private fun flipHorizontal(bitmap: Bitmap): Bitmap {
    val matrix = Matrix().apply {
      setScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
    }
    val result = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    bitmap.recycle()
    return result
  }

  private fun flipVertical(bitmap: Bitmap): Bitmap {
    val matrix = Matrix().apply {
      setScale(1f, -1f, bitmap.width / 2f, bitmap.height / 2f)
    }
    val result = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    bitmap.recycle()
    return result
  }

  private fun adjustBrightness(bitmap: Bitmap, factor: Float): Bitmap {
    // Factor 1.0 = no change, < 1.0 = darker, > 1.0 = brighter
    val offset = (factor - 1f) * 255f

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

  private fun applyBasicBlur(bitmap: Bitmap, radius: Float): Bitmap {
    // Simple box blur implementation
    // For better performance, use RenderScript on supported devices
    val width = bitmap.width
    val height = bitmap.height
    val pixels = IntArray(width * height)
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

    val blurRadius = radius.toInt().coerceIn(1, 25)
    val result = IntArray(width * height)

    // Horizontal pass
    for (y in 0 until height) {
      for (x in 0 until width) {
        var r = 0
        var g = 0
        var b = 0
        var a = 0
        var count = 0

        for (dx in -blurRadius..blurRadius) {
          val nx = x + dx
          if (nx >= 0 && nx < width) {
            val pixel = pixels[y * width + nx]
            a += (pixel shr 24) and 0xFF
            r += (pixel shr 16) and 0xFF
            g += (pixel shr 8) and 0xFF
            b += pixel and 0xFF
            count++
          }
        }

        result[y * width + x] = ((a / count) shl 24) or
                                ((r / count) shl 16) or
                                ((g / count) shl 8) or
                                (b / count)
      }
    }

    // Vertical pass
    System.arraycopy(result, 0, pixels, 0, pixels.size)

    for (y in 0 until height) {
      for (x in 0 until width) {
        var r = 0
        var g = 0
        var b = 0
        var a = 0
        var count = 0

        for (dy in -blurRadius..blurRadius) {
          val ny = y + dy
          if (ny >= 0 && ny < height) {
            val pixel = pixels[ny * width + x]
            a += (pixel shr 24) and 0xFF
            r += (pixel shr 16) and 0xFF
            g += (pixel shr 8) and 0xFF
            b += pixel and 0xFF
            count++
          }
        }

        result[y * width + x] = ((a / count) shl 24) or
                                ((r / count) shl 16) or
                                ((g / count) shl 8) or
                                (b / count)
      }
    }

    val blurredBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    blurredBitmap.setPixels(result, 0, width, 0, 0, width, height)
    bitmap.recycle()

    return blurredBitmap
  }
}
