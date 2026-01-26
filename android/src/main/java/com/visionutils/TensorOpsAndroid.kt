package com.visionutils

import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import kotlin.math.min

/**
 * Tensor operations for Android
 */
object TensorOpsAndroid {

  /**
   * Extract a single channel from pixel data
   */
  fun extractChannel(
    data: FloatArray,
    width: Int,
    height: Int,
    channels: Int,
    channelIndex: Int,
    dataLayout: String
  ): WritableMap {
    if (channelIndex < 0 || channelIndex >= channels) {
      throw VisionUtilsException("INVALID_CHANNEL", "Channel index $channelIndex out of range [0, $channels)")
    }

    val pixelCount = width * height
    val channelData = FloatArray(pixelCount)

    when (dataLayout) {
      "hwc" -> {
        // Height x Width x Channel layout (interleaved)
        for (i in 0 until pixelCount) {
          channelData[i] = data[i * channels + channelIndex]
        }
      }
      "chw" -> {
        // Channel x Height x Width layout (planar)
        val offset = channelIndex * pixelCount
        System.arraycopy(data, offset, channelData, 0, pixelCount)
      }
      else -> {
        throw VisionUtilsException("INVALID_LAYOUT", "Unknown data layout: $dataLayout")
      }
    }

    val resultArray = Arguments.createArray()
    for (value in channelData) {
      resultArray.pushDouble(value.toDouble())
    }

    return Arguments.createMap().apply {
      putArray("data", resultArray)
      putInt("width", width)
      putInt("height", height)
    }
  }

  /**
   * Extract a rectangular patch from pixel data
   */
  fun extractPatch(
    data: FloatArray,
    width: Int,
    height: Int,
    channels: Int,
    patchOptions: ReadableMap,
    dataLayout: String
  ): WritableMap {
    val x = patchOptions.getInt("x")
    val y = patchOptions.getInt("y")
    val patchWidth = patchOptions.getInt("width")
    val patchHeight = patchOptions.getInt("height")

    // Validate bounds
    if (x < 0 || y < 0 || x + patchWidth > width || y + patchHeight > height) {
      throw VisionUtilsException(
        "INVALID_PATCH",
        "Patch ($x, $y, $patchWidth, $patchHeight) exceeds image bounds ($width, $height)"
      )
    }

    val patchSize = patchWidth * patchHeight * channels
    val patchData = FloatArray(patchSize)

    when (dataLayout) {
      "hwc" -> {
        var idx = 0
        for (py in y until y + patchHeight) {
          for (px in x until x + patchWidth) {
            val srcOffset = (py * width + px) * channels
            for (c in 0 until channels) {
              patchData[idx++] = data[srcOffset + c]
            }
          }
        }
      }
      "chw" -> {
        val pixelCount = width * height
        var idx = 0
        for (c in 0 until channels) {
          val channelOffset = c * pixelCount
          for (py in y until y + patchHeight) {
            for (px in x until x + patchWidth) {
              patchData[idx++] = data[channelOffset + py * width + px]
            }
          }
        }
      }
      else -> {
        throw VisionUtilsException("INVALID_LAYOUT", "Unknown data layout: $dataLayout")
      }
    }

    val resultArray = Arguments.createArray()
    for (value in patchData) {
      resultArray.pushDouble(value.toDouble())
    }

    return Arguments.createMap().apply {
      putArray("data", resultArray)
      putInt("width", patchWidth)
      putInt("height", patchHeight)
      putInt("channels", channels)
    }
  }

  /**
   * Concatenate multiple results into a batch tensor
   */
  fun concatenateToBatch(results: ReadableArray): WritableMap {
    if (results.size() == 0) {
      throw VisionUtilsException("EMPTY_BATCH", "Cannot create batch from empty array")
    }

    // Get dimensions from first result
    val first = results.getMap(0) ?: throw VisionUtilsException("INVALID_RESULT", "Invalid result at index 0")
    val width = first.getInt("width")
    val height = first.getInt("height")
    val channels = first.getInt("channels")
    val perImageSize = width * height * channels

    val batchSize = results.size()
    val batchData = Arguments.createArray()

    for (i in 0 until batchSize) {
      val result = results.getMap(i) ?: throw VisionUtilsException("INVALID_RESULT", "Invalid result at index $i")

      // Validate dimensions match
      if (result.getInt("width") != width ||
          result.getInt("height") != height ||
          result.getInt("channels") != channels) {
        throw VisionUtilsException(
          "DIMENSION_MISMATCH",
          "Result at index $i has different dimensions"
        )
      }

      val data = result.getArray("data") ?: throw VisionUtilsException("INVALID_DATA", "Missing data at index $i")
      for (j in 0 until data.size()) {
        batchData.pushDouble(data.getDouble(j))
      }
    }

    val shape = Arguments.createArray().apply {
      pushInt(batchSize)
      pushInt(channels)
      pushInt(height)
      pushInt(width)
    }

    return Arguments.createMap().apply {
      putArray("data", batchData)
      putArray("shape", shape)
      putInt("batchSize", batchSize)
    }
  }

  /**
   * Permute tensor dimensions
   */
  fun permute(data: FloatArray, shape: IntArray, order: IntArray): WritableMap {
    // Validate order
    if (order.size != shape.size) {
      throw VisionUtilsException(
        "INVALID_ORDER",
        "Order length ${order.size} doesn't match shape dimensions ${shape.size}"
      )
    }

    // Calculate strides for original shape
    val ndim = shape.size
    val strides = IntArray(ndim)
    strides[ndim - 1] = 1
    for (i in ndim - 2 downTo 0) {
      strides[i] = strides[i + 1] * shape[i + 1]
    }

    // Calculate new shape and strides
    val newShape = IntArray(ndim) { shape[order[it]] }
    val newStrides = IntArray(ndim)
    newStrides[ndim - 1] = 1
    for (i in ndim - 2 downTo 0) {
      newStrides[i] = newStrides[i + 1] * newShape[i + 1]
    }

    // Permute data
    val totalSize = data.size
    val result = FloatArray(totalSize)
    val indices = IntArray(ndim)

    for (flatIdx in 0 until totalSize) {
      // Convert flat index to multi-dimensional indices in new shape
      var remaining = flatIdx
      for (i in 0 until ndim) {
        indices[i] = remaining / newStrides[i]
        remaining %= newStrides[i]
      }

      // Map to original indices
      var srcIdx = 0
      for (i in 0 until ndim) {
        srcIdx += indices[i] * strides[order[i]]
      }

      result[flatIdx] = data[srcIdx]
    }

    val resultArray = Arguments.createArray()
    for (value in result) {
      resultArray.pushDouble(value.toDouble())
    }

    val shapeArray = Arguments.createArray()
    for (dim in newShape) {
      shapeArray.pushInt(dim)
    }

    return Arguments.createMap().apply {
      putArray("data", resultArray)
      putArray("shape", shapeArray)
    }
  }
}
