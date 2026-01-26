package com.visionutils

import android.graphics.Bitmap
import android.graphics.Matrix
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap

/**
 * Multi-crop operations for Android
 */
object MultiCropAndroid {

  /**
   * Perform five-crop operation (4 corners + center)
   */
  fun fiveCrop(
    bitmap: Bitmap,
    options: ReadableMap,
    pixelOptions: ReadableMap
  ): WritableMap {
    val cropWidth = options.getInt("width")
    val cropHeight = options.getInt("height")

    val imageWidth = bitmap.width
    val imageHeight = bitmap.height

    if (cropWidth > imageWidth || cropHeight > imageHeight) {
      throw VisionUtilsException(
        "INVALID_CROP_SIZE",
        "Crop size ($cropWidth x $cropHeight) exceeds image size ($imageWidth x $imageHeight)"
      )
    }

    val crops = mutableListOf<Bitmap>()

    // Top-left
    crops.add(Bitmap.createBitmap(bitmap, 0, 0, cropWidth, cropHeight))

    // Top-right
    crops.add(Bitmap.createBitmap(bitmap, imageWidth - cropWidth, 0, cropWidth, cropHeight))

    // Bottom-left
    crops.add(Bitmap.createBitmap(bitmap, 0, imageHeight - cropHeight, cropWidth, cropHeight))

    // Bottom-right
    crops.add(Bitmap.createBitmap(bitmap, imageWidth - cropWidth, imageHeight - cropHeight, cropWidth, cropHeight))

    // Center
    val centerX = (imageWidth - cropWidth) / 2
    val centerY = (imageHeight - cropHeight) / 2
    crops.add(Bitmap.createBitmap(bitmap, centerX, centerY, cropWidth, cropHeight))

    // Process each crop
    val results = Arguments.createArray()
    val parsedOptions = GetPixelDataOptions.fromMap(pixelOptions)

    for (crop in crops) {
      val result = PixelProcessor.process(crop, parsedOptions)
      results.pushMap(result.toWritableMap())
      crop.recycle()
    }

    return Arguments.createMap().apply {
      putArray("results", results)
      putInt("cropCount", 5)
      putInt("cropWidth", cropWidth)
      putInt("cropHeight", cropHeight)
    }
  }

  /**
   * Perform ten-crop operation (five-crop + horizontal flips)
   */
  fun tenCrop(
    bitmap: Bitmap,
    options: ReadableMap,
    pixelOptions: ReadableMap
  ): WritableMap {
    val cropWidth = options.getInt("width")
    val cropHeight = options.getInt("height")

    val imageWidth = bitmap.width
    val imageHeight = bitmap.height

    if (cropWidth > imageWidth || cropHeight > imageHeight) {
      throw VisionUtilsException(
        "INVALID_CROP_SIZE",
        "Crop size ($cropWidth x $cropHeight) exceeds image size ($imageWidth x $imageHeight)"
      )
    }

    val crops = mutableListOf<Bitmap>()

    // Create flip matrix
    val flipMatrix = Matrix().apply {
      setScale(-1f, 1f, cropWidth / 2f, cropHeight / 2f)
    }

    // Define crop positions
    val positions = listOf(
      Pair(0, 0), // Top-left
      Pair(imageWidth - cropWidth, 0), // Top-right
      Pair(0, imageHeight - cropHeight), // Bottom-left
      Pair(imageWidth - cropWidth, imageHeight - cropHeight), // Bottom-right
      Pair((imageWidth - cropWidth) / 2, (imageHeight - cropHeight) / 2) // Center
    )

    // Extract crops and their flips
    for ((x, y) in positions) {
      val crop = Bitmap.createBitmap(bitmap, x, y, cropWidth, cropHeight)
      crops.add(crop)

      // Create flipped version
      val flipped = Bitmap.createBitmap(crop, 0, 0, cropWidth, cropHeight, flipMatrix, true)
      crops.add(flipped)
    }

    // Process each crop
    val results = Arguments.createArray()
    val parsedOptions = GetPixelDataOptions.fromMap(pixelOptions)

    for (crop in crops) {
      val result = PixelProcessor.process(crop, parsedOptions)
      results.pushMap(result.toWritableMap())
      crop.recycle()
    }

    return Arguments.createMap().apply {
      putArray("results", results)
      putInt("cropCount", 10)
      putInt("cropWidth", cropWidth)
      putInt("cropHeight", cropHeight)
    }
  }
}
