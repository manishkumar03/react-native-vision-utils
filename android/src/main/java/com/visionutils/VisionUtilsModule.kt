package com.visionutils

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.WritableMap
import kotlinx.coroutines.*
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicBlur
import android.util.Base64
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.pow

class VisionUtilsModule(reactContext: ReactApplicationContext) :
  NativeVisionUtilsSpec(reactContext) {

  private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

  // Simple image cache
  private val imageCache = mutableMapOf<String, Pair<FloatArray, Long>>()
  private var cacheHitCount = 0
  private var cacheMissCount = 0
  private val maxCacheSize = 50

  override fun invalidate() {
    super.invalidate()
    scope.cancel()
  }

  /**
   * Get pixel data from a single image
   */
  override fun getPixelData(options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val parsedOptions = GetPixelDataOptions.fromMap(options)
        val context = reactApplicationContext.applicationContext

        val bitmap = ImageLoader.loadImage(context, parsedOptions.source)
        val result = PixelProcessor.process(bitmap, parsedOptions)

        withContext(Dispatchers.Main) {
          promise.resolve(result.toWritableMap())
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Get pixel data from multiple images with concurrency control
   */
  override fun batchGetPixelData(
    optionsArray: ReadableArray,
    batchOptions: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      val startTime = System.nanoTime()
      val context = reactApplicationContext.applicationContext
      val batchOpts = BatchOptions.fromMap(batchOptions)

      val results = Arguments.createArray()
      val semaphore = kotlinx.coroutines.sync.Semaphore(batchOpts.concurrency)

      val jobs = (0 until optionsArray.size()).map { index ->
        async {
          semaphore.acquire()
          try {
            val optionsMap = optionsArray.getMap(index)
              ?: throw VisionUtilsException("INVALID_SOURCE", "Invalid options at index $index")

            val parsedOptions = GetPixelDataOptions.fromMap(optionsMap)
            val bitmap = ImageLoader.loadImage(context, parsedOptions.source)
            val result = PixelProcessor.process(bitmap, parsedOptions)

            Pair(index, result.toWritableMap())
          } catch (e: VisionUtilsException) {
            val errorMap = Arguments.createMap().apply {
              putBoolean("error", true)
              putString("message", e.message)
              putString("code", e.code)
              putInt("index", index)
            }
            Pair(index, errorMap)
          } catch (e: Exception) {
            val errorMap = Arguments.createMap().apply {
              putBoolean("error", true)
              putString("message", e.message ?: "Unknown error")
              putString("code", "UNKNOWN")
              putInt("index", index)
            }
            Pair(index, errorMap)
          } finally {
            semaphore.release()
          }
        }
      }

      // Collect results in order
      val orderedResults = jobs.map { it.await() }
        .sortedBy { it.first }
        .map { it.second }

      orderedResults.forEach { results.pushMap(it) }

      val endTime = System.nanoTime()
      val totalTimeMs = (endTime - startTime) / 1_000_000.0

      val response = Arguments.createMap().apply {
        putArray("results", results)
        putDouble("totalTimeMs", totalTimeMs)
      }

      withContext(Dispatchers.Main) {
        promise.resolve(response)
      }
    }
  }

  /**
   * Get image statistics
   */
  override fun getImageStatistics(source: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)
        val stats = ImageAnalyzerAndroid.getStatistics(bitmap)

        withContext(Dispatchers.Main) {
          promise.resolve(stats)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Get image metadata
   */
  override fun getImageMetadata(source: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)
        val metadata = ImageAnalyzerAndroid.getMetadata(bitmap)

        withContext(Dispatchers.Main) {
          promise.resolve(metadata)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Validate image against criteria
   */
  override fun validateImage(source: ReadableMap, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)
        val result = ImageAnalyzerAndroid.validate(bitmap, options)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Convert tensor data to image
   */
  override fun tensorToImage(
    data: ReadableArray,
    width: Double,
    height: Double,
    options: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val floatData = FloatArray(data.size()) { data.getDouble(it).toFloat() }
        val result = TensorConverterAndroid.tensorToImage(
          floatData,
          width.toInt(),
          height.toInt(),
          options
        )

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Five-crop operation
   */
  override fun fiveCrop(
    source: ReadableMap,
    options: ReadableMap,
    pixelOptions: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)
        val result = MultiCropAndroid.fiveCrop(bitmap, options, pixelOptions)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Ten-crop operation
   */
  override fun tenCrop(
    source: ReadableMap,
    options: ReadableMap,
    pixelOptions: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)
        val result = MultiCropAndroid.tenCrop(bitmap, options, pixelOptions)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Extract channel from pixel data
   */
  override fun extractChannel(
    data: ReadableArray,
    width: Double,
    height: Double,
    channels: Double,
    channelIndex: Double,
    dataLayout: String,
    promise: Promise
  ) {
    scope.launch {
      try {
        val floatData = FloatArray(data.size()) { data.getDouble(it).toFloat() }
        val result = TensorOpsAndroid.extractChannel(
          floatData,
          width.toInt(),
          height.toInt(),
          channels.toInt(),
          channelIndex.toInt(),
          dataLayout
        )

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Extract patch from pixel data
   */
  override fun extractPatch(
    data: ReadableArray,
    width: Double,
    height: Double,
    channels: Double,
    patchOptions: ReadableMap,
    dataLayout: String,
    promise: Promise
  ) {
    scope.launch {
      try {
        val floatData = FloatArray(data.size()) { data.getDouble(it).toFloat() }
        val result = TensorOpsAndroid.extractPatch(
          floatData,
          width.toInt(),
          height.toInt(),
          channels.toInt(),
          patchOptions,
          dataLayout
        )

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Concatenate results to batch
   */
  override fun concatenateToBatch(results: ReadableArray, promise: Promise) {
    scope.launch {
      try {
        val result = TensorOpsAndroid.concatenateToBatch(results)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Permute tensor dimensions
   */
  override fun permute(
    data: ReadableArray,
    shape: ReadableArray,
    order: ReadableArray,
    promise: Promise
  ) {
    scope.launch {
      try {
        val floatData = FloatArray(data.size()) { data.getDouble(it).toFloat() }
        val shapeArray = IntArray(shape.size()) { shape.getInt(it) }
        val orderArray = IntArray(order.size()) { order.getInt(it) }

        val result = TensorOpsAndroid.permute(floatData, shapeArray, orderArray)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Apply augmentations
   */
  override fun applyAugmentations(
    source: ReadableMap,
    augmentations: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)
        val result = ImageAugmenterAndroid.apply(bitmap, augmentations)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
        }
      }
    }
  }

  /**
   * Clear cache
   */
  override fun clearCache(promise: Promise) {
    synchronized(imageCache) {
      imageCache.clear()
      cacheHitCount = 0
      cacheMissCount = 0
    }
    promise.resolve(null)
  }

  /**
   * Get cache statistics
   */
  override fun getCacheStats(promise: Promise) {
    val stats = Arguments.createMap().apply {
      synchronized(imageCache) {
        putInt("hitCount", cacheHitCount)
        putInt("missCount", cacheMissCount)
        putInt("size", imageCache.size)
        putInt("maxSize", maxCacheSize)
      }
    }
    promise.resolve(stats)
  }

  companion object {
    const val NAME = NativeVisionUtilsSpec.NAME
  }
}
