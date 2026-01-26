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
        val loadResult = ImageLoader.loadImageWithMetadata(context, imageSource)
        val metadata = ImageAnalyzerAndroid.getMetadata(
          loadResult.bitmap,
          loadResult.fileSize,
          loadResult.format
        )

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
        // Debug logging
        android.util.Log.d("VisionUtils", "fiveCrop source: ${source.toHashMap()}")
        android.util.Log.d("VisionUtils", "fiveCrop source has type: ${source.hasKey("type")}")
        if (source.hasKey("type")) {
          android.util.Log.d("VisionUtils", "fiveCrop source type value: ${source.getString("type")}")
        }

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
   * Apply color jitter augmentation with granular control
   */
  override fun colorJitter(
    source: ReadableMap,
    options: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)
        val result = ColorJitterAndroid.apply(bitmap, options)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("COLOR_JITTER_ERROR", e.message ?: "Color jitter failed")
        }
      }
    }
  }

  /**
   * Apply cutout/random erasing augmentation
   */
  override fun cutout(
    source: ReadableMap,
    options: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)
        val result = CutoutAndroid.apply(bitmap, options)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("CUTOUT_ERROR", e.message ?: "Cutout failed")
        }
      }
    }
  }

  /**
   * Quantize float data to integer format
   */
  override fun quantize(
    data: ReadableArray,
    options: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val floatData = FloatArray(data.size()) { data.getDouble(it).toFloat() }
        val result = QuantizationAndroid.quantize(floatData, options)

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
   * Dequantize integer data back to float
   */
  override fun dequantize(
    data: ReadableArray,
    options: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val intData = IntArray(data.size()) { data.getInt(it) }
        val result = QuantizationAndroid.dequantize(intData, options)

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
   * Calculate optimal quantization parameters from data
   */
  override fun calculateQuantizationParams(
    data: ReadableArray,
    options: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val floatData = FloatArray(data.size()) { data.getDouble(it).toFloat() }
        val result = QuantizationAndroid.calculateQuantizationParams(floatData, options)

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

  // MARK: - Label Database

  /**
   * Get label by index from a dataset
   */
  override fun getLabel(index: Double, dataset: String, includeMetadata: Boolean, promise: Promise) {
    try {
      val options = Arguments.createMap().apply {
        putString("dataset", dataset)
        putBoolean("includeMetadata", includeMetadata)
      }
      val result = LabelDatabaseAndroid.getLabel(index.toInt(), options)
      promise.resolve(result)
    } catch (e: VisionUtilsException) {
      promise.reject(e.code, e.message)
    } catch (e: Exception) {
      promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
    }
  }

  /**
   * Get top labels from prediction scores
   */
  override fun getTopLabels(scores: ReadableArray, options: ReadableMap, promise: Promise) {
    try {
      val result = LabelDatabaseAndroid.getTopLabels(scores, options)
      promise.resolve(result)
    } catch (e: VisionUtilsException) {
      promise.reject(e.code, e.message)
    } catch (e: Exception) {
      promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
    }
  }

  /**
   * Get all labels for a dataset
   */
  override fun getAllLabels(dataset: String, promise: Promise) {
    try {
      val result = LabelDatabaseAndroid.getAllLabels(dataset)
      promise.resolve(result)
    } catch (e: VisionUtilsException) {
      promise.reject(e.code, e.message)
    } catch (e: Exception) {
      promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
    }
  }

  /**
   * Get dataset information
   */
  override fun getDatasetInfo(dataset: String, promise: Promise) {
    try {
      val result = LabelDatabaseAndroid.getDatasetInfo(dataset)
      promise.resolve(result)
    } catch (e: VisionUtilsException) {
      promise.reject(e.code, e.message)
    } catch (e: Exception) {
      promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
    }
  }

  /**
   * Get available datasets
   */
  override fun getAvailableDatasets(promise: Promise) {
    try {
      val result = LabelDatabaseAndroid.getAvailableDatasets()
      promise.resolve(result)
    } catch (e: Exception) {
      promise.reject("UNKNOWN", e.message ?: "Unknown error occurred")
    }
  }

  // MARK: - Camera Frame Processing

  /**
   * Process camera frame into ML-ready tensor
   */
  override fun processCameraFrame(source: ReadableMap, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val result = CameraFrameAndroid.processCameraFrame(source, options)
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
   * Direct YUV to RGB conversion
   */
  override fun convertYUVToRGB(
    yBuffer: String,
    uBuffer: String,
    vBuffer: String,
    width: Double,
    height: Double,
    pixelFormat: String,
    promise: Promise
  ) {
    scope.launch {
      try {
        val options = Arguments.createMap().apply {
          putString("yBuffer", yBuffer)
          putString("uBuffer", uBuffer)
          putString("vBuffer", vBuffer)
          putDouble("width", width)
          putDouble("height", height)
          putString("pixelFormat", pixelFormat)
        }
        val result = CameraFrameAndroid.convertYUVToRGB(options)
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

  // MARK: - Bounding Box Utilities

  /**
   * Convert bounding boxes between formats
   */
  override fun convertBoxFormat(
    boxes: ReadableArray,
    sourceFormat: String,
    targetFormat: String,
    promise: Promise
  ) {
    scope.launch {
      try {
        val startTime = System.nanoTime()

        val boxesList = mutableListOf<List<Double>>()
        for (i in 0 until boxes.size()) {
          val box = boxes.getArray(i)
          if (box != null && box.size() == 4) {
            boxesList.add(listOf(
              box.getDouble(0), box.getDouble(1),
              box.getDouble(2), box.getDouble(3)
            ))
          }
        }

        val convertedBoxes = BoundingBoxUtilsAndroid.convertBoxFormat(boxesList, sourceFormat, targetFormat)

        val endTime = System.nanoTime()
        val processingTimeMs = (endTime - startTime) / 1_000_000.0

        val response = Arguments.createMap().apply {
          val boxesArray = Arguments.createArray()
          convertedBoxes.forEach { box ->
            val arr = Arguments.createArray()
            box.forEach { arr.pushDouble(it) }
            boxesArray.pushArray(arr)
          }
          putArray("boxes", boxesArray)
          putString("format", targetFormat)
          putDouble("processingTimeMs", processingTimeMs)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("BOX_CONVERT_ERROR", e.message ?: "Failed to convert box format")
        }
      }
    }
  }

  /**
   * Scale bounding boxes
   */
  override fun scaleBoxes(boxes: ReadableArray, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val boxesList = mutableListOf<List<Double>>()
        for (i in 0 until boxes.size()) {
          val box = boxes.getArray(i)
          if (box != null && box.size() == 4) {
            boxesList.add(listOf(
              box.getDouble(0), box.getDouble(1),
              box.getDouble(2), box.getDouble(3)
            ))
          }
        }

        val fromWidth = options.getDouble("fromWidth")
        val fromHeight = options.getDouble("fromHeight")
        val toWidth = options.getDouble("toWidth")
        val toHeight = options.getDouble("toHeight")
        val format = if (options.hasKey("format")) options.getString("format") ?: "xyxy" else "xyxy"

        val result = BoundingBoxUtilsAndroid.scaleBoxes(
          boxesList, fromWidth, fromHeight, toWidth, toHeight, format
        )

        val response = Arguments.createMap().apply {
          val boxesArray = Arguments.createArray()
          result.boxes.forEach { box ->
            val arr = Arguments.createArray()
            box.forEach { arr.pushDouble(it) }
            boxesArray.pushArray(arr)
          }
          putArray("boxes", boxesArray)
          putString("format", result.format)
          putDouble("processingTimeMs", result.processingTimeMs)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("BOX_SCALE_ERROR", e.message ?: "Failed to scale boxes")
        }
      }
    }
  }

  /**
   * Clip bounding boxes to image boundaries
   */
  override fun clipBoxes(boxes: ReadableArray, width: Double, height: Double, format: String, promise: Promise) {
    scope.launch {
      try {
        val boxesList = mutableListOf<List<Double>>()
        for (i in 0 until boxes.size()) {
          val box = boxes.getArray(i)
          if (box != null && box.size() == 4) {
            boxesList.add(listOf(
              box.getDouble(0), box.getDouble(1),
              box.getDouble(2), box.getDouble(3)
            ))
          }
        }

        val result = BoundingBoxUtilsAndroid.clipBoxes(boxesList, width, height, format)

        val response = Arguments.createMap().apply {
          val boxesArray = Arguments.createArray()
          result.boxes.forEach { box ->
            val arr = Arguments.createArray()
            box.forEach { arr.pushDouble(it) }
            boxesArray.pushArray(arr)
          }
          putArray("boxes", boxesArray)
          putString("format", result.format)
          putInt("removedCount", result.removedCount)
          putDouble("processingTimeMs", result.processingTimeMs)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("BOX_CLIP_ERROR", e.message ?: "Failed to clip boxes")
        }
      }
    }
  }

  /**
   * Calculate Intersection over Union between two boxes
   */
  override fun calculateIoU(box1: ReadableArray, box2: ReadableArray, format: String, promise: Promise) {
    scope.launch {
      try {
        val boxList1 = listOf(
          box1.getDouble(0), box1.getDouble(1),
          box1.getDouble(2), box1.getDouble(3)
        )
        val boxList2 = listOf(
          box2.getDouble(0), box2.getDouble(1),
          box2.getDouble(2), box2.getDouble(3)
        )

        val result = BoundingBoxUtilsAndroid.calculateIoU(boxList1, boxList2, format)

        val response = Arguments.createMap().apply {
          putDouble("iou", result["iou"] ?: 0.0)
          putDouble("intersection", result["intersection"] ?: 0.0)
          putDouble("union", result["union"] ?: 0.0)
          putDouble("processingTimeMs", result["processingTimeMs"] ?: 0.0)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("IOU_ERROR", e.message ?: "Failed to calculate IoU")
        }
      }
    }
  }

  /**
   * Non-Maximum Suppression
   */
  override fun nonMaxSuppression(detections: ReadableArray, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val detectionsList = mutableListOf<Map<String, Any>>()
        for (i in 0 until detections.size()) {
          val det = detections.getMap(i) ?: continue
          val boxArr = det.getArray("box") ?: continue
          if (boxArr.size() < 4) continue

          val detection = mutableMapOf<String, Any>(
            "box" to listOf(
              boxArr.getDouble(0), boxArr.getDouble(1),
              boxArr.getDouble(2), boxArr.getDouble(3)
            ),
            "score" to det.getDouble("score")
          )
          if (det.hasKey("classIndex")) {
            detection["classIndex"] = det.getInt("classIndex")
          }
          if (det.hasKey("label")) {
            detection["label"] = det.getString("label") ?: ""
          }
          detectionsList.add(detection)
        }

        val iouThreshold = if (options.hasKey("iouThreshold")) options.getDouble("iouThreshold") else 0.5
        val scoreThreshold = if (options.hasKey("scoreThreshold")) options.getDouble("scoreThreshold") else 0.0
        val maxDetections = if (options.hasKey("maxDetections")) options.getInt("maxDetections") else null
        val format = if (options.hasKey("format")) options.getString("format") ?: "xyxy" else "xyxy"

        val result = BoundingBoxUtilsAndroid.nonMaxSuppression(
          detectionsList, iouThreshold, scoreThreshold, maxDetections, format
        )

        val response = Arguments.createMap().apply {
          val detsArray = Arguments.createArray()
          result.detections.forEach { det ->
            val detMap = Arguments.createMap()
            val boxArr = Arguments.createArray()
            @Suppress("UNCHECKED_CAST")
            (det["box"] as List<Double>).forEach { boxArr.pushDouble(it) }
            detMap.putArray("box", boxArr)
            detMap.putDouble("score", det["score"] as Double)
            if (det.containsKey("classIndex")) {
              detMap.putInt("classIndex", det["classIndex"] as Int)
            }
            if (det.containsKey("label")) {
              detMap.putString("label", det["label"] as String)
            }
            detsArray.pushMap(detMap)
          }
          putArray("detections", detsArray)

          val indicesArray = Arguments.createArray()
          result.indices.forEach { indicesArray.pushInt(it) }
          putArray("indices", indicesArray)

          putInt("suppressedCount", result.suppressedCount)
          putInt("totalBefore", result.totalBefore)
          putInt("totalAfter", result.totalAfter)
          putDouble("processingTimeMs", result.processingTimeMs)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("NMS_ERROR", e.message ?: "Failed to run NMS")
        }
      }
    }
  }

  // MARK: - Letterbox Utilities

  /**
   * Letterbox an image for YOLO-style models
   */
  override fun letterbox(source: ReadableMap, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)

        val targetWidth = options.getDouble("targetWidth").toInt()
        val targetHeight = options.getDouble("targetHeight").toInt()

        val fillColor = if (options.hasKey("fillColor")) {
          val arr = options.getArray("fillColor")
          if (arr != null && arr.size() >= 3) {
            listOf(arr.getInt(0), arr.getInt(1), arr.getInt(2))
          } else listOf(114, 114, 114)
        } else listOf(114, 114, 114)

        val scaleUp = if (options.hasKey("scaleUp")) options.getBoolean("scaleUp") else true

        val result = LetterboxUtilsAndroid.letterbox(
          bitmap, targetWidth, targetHeight, fillColor, scaleUp,
          autoStride = false, stride = 32, center = true
        )

        // Convert to proper React Native response
        val response = Arguments.createMap().apply {
          putString("imageBase64", result["imageBase64"] as String)
          putInt("width", result["width"] as Int)
          putInt("height", result["height"] as Int)
          putDouble("scale", result["scale"] as Double)

          @Suppress("UNCHECKED_CAST")
          val padding = result["padding"] as List<Int>
          putArray("padding", Arguments.createArray().apply {
            padding.forEach { pushInt(it) }
          })

          @Suppress("UNCHECKED_CAST")
          val offset = result["offset"] as List<Int>
          putArray("offset", Arguments.createArray().apply {
            offset.forEach { pushInt(it) }
          })

          @Suppress("UNCHECKED_CAST")
          val originalSize = result["originalSize"] as List<Int>
          putArray("originalSize", Arguments.createArray().apply {
            originalSize.forEach { pushInt(it) }
          })

          // Add letterboxInfo
          @Suppress("UNCHECKED_CAST")
          val letterboxInfo = result["letterboxInfo"] as Map<String, Any>
          putMap("letterboxInfo", Arguments.createMap().apply {
            putDouble("scale", letterboxInfo["scale"] as Double)

            @Suppress("UNCHECKED_CAST")
            val infoPadding = letterboxInfo["padding"] as List<Int>
            putArray("padding", Arguments.createArray().apply {
              infoPadding.forEach { pushInt(it) }
            })

            @Suppress("UNCHECKED_CAST")
            val infoOffset = letterboxInfo["offset"] as List<Int>
            putArray("offset", Arguments.createArray().apply {
              infoOffset.forEach { pushInt(it) }
            })

            @Suppress("UNCHECKED_CAST")
            val infoOriginalSize = letterboxInfo["originalSize"] as List<Int>
            putArray("originalSize", Arguments.createArray().apply {
              infoOriginalSize.forEach { pushInt(it) }
            })

            @Suppress("UNCHECKED_CAST")
            val infoLetterboxedSize = letterboxInfo["letterboxedSize"] as List<Int>
            putArray("letterboxedSize", Arguments.createArray().apply {
              infoLetterboxedSize.forEach { pushInt(it) }
            })
          })

          putDouble("processingTimeMs", result["processingTimeMs"] as Double)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("LETTERBOX_ERROR", e.message ?: "Failed to letterbox image")
        }
      }
    }
  }

  /**
   * Reverse letterbox transformation on coordinates
   */
  override fun reverseLetterbox(boxes: ReadableArray, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val startTime = System.nanoTime()

        val boxesList = mutableListOf<List<Double>>()
        for (i in 0 until boxes.size()) {
          val box = boxes.getArray(i)
          if (box != null && box.size() == 4) {
            boxesList.add(listOf(
              box.getDouble(0), box.getDouble(1),
              box.getDouble(2), box.getDouble(3)
            ))
          }
        }

        val scale = options.getDouble("scale")
        val offsetArr = options.getArray("offset")
        val padding = if (offsetArr != null && offsetArr.size() >= 2) {
          listOf(offsetArr.getDouble(0).toInt(), offsetArr.getDouble(1).toInt())
        } else listOf(0, 0)

        val originalSizeArr = options.getArray("originalSize")
        val originalWidth = if (originalSizeArr != null && originalSizeArr.size() >= 1) originalSizeArr.getInt(0) else 0
        val originalHeight = if (originalSizeArr != null && originalSizeArr.size() >= 2) originalSizeArr.getInt(1) else 0

        val format = if (options.hasKey("format")) options.getString("format") ?: "xyxy" else "xyxy"
        val clip = if (options.hasKey("clip")) options.getBoolean("clip") else true

        val transformedBoxes = LetterboxUtilsAndroid.reverseLetterbox(
          boxesList, scale, padding, originalWidth, originalHeight, format, clip
        )

        val endTime = System.nanoTime()
        val processingTimeMs = (endTime - startTime) / 1_000_000.0

        val response = Arguments.createMap().apply {
          val boxesArray = Arguments.createArray()
          transformedBoxes.forEach { box ->
            val arr = Arguments.createArray()
            box.forEach { arr.pushDouble(it) }
            boxesArray.pushArray(arr)
          }
          putArray("boxes", boxesArray)
          putString("format", format)
          putDouble("processingTimeMs", processingTimeMs)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("REVERSE_LETTERBOX_ERROR", e.message ?: "Failed to reverse letterbox")
        }
      }
    }
  }

  // MARK: - Drawing Utilities

  /**
   * Draw bounding boxes on an image
   */
  override fun drawBoxes(source: ReadableMap, boxes: ReadableArray, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)

        val boxesList = mutableListOf<Map<String, Any>>()
        for (i in 0 until boxes.size()) {
          val boxMap = boxes.getMap(i) ?: continue
          val boxArr = boxMap.getArray("box") ?: continue
          if (boxArr.size() < 4) continue

          val box = mutableMapOf<String, Any>(
            "box" to listOf(
              boxArr.getDouble(0), boxArr.getDouble(1),
              boxArr.getDouble(2), boxArr.getDouble(3)
            )
          )
          if (boxMap.hasKey("label")) {
            box["label"] = boxMap.getString("label") ?: ""
          }
          if (boxMap.hasKey("score")) {
            box["score"] = boxMap.getDouble("score")
          }
          if (boxMap.hasKey("classIndex")) {
            box["classIndex"] = boxMap.getInt("classIndex")
          }
          if (boxMap.hasKey("color")) {
            val colorArr = boxMap.getArray("color")
            if (colorArr != null && colorArr.size() >= 3) {
              box["color"] = listOf(colorArr.getInt(0), colorArr.getInt(1), colorArr.getInt(2))
            }
          }
          boxesList.add(box)
        }

        val lineWidth = if (options.hasKey("lineWidth")) options.getDouble("lineWidth").toFloat() else 2f
        val fontSize = if (options.hasKey("fontSize")) options.getDouble("fontSize").toFloat() else 12f
        val drawLabels = if (options.hasKey("drawLabels")) options.getBoolean("drawLabels") else true
        val labelBackgroundAlpha = if (options.hasKey("labelBackgroundAlpha")) options.getDouble("labelBackgroundAlpha").toFloat() else 0.7f

        val labelColor = if (options.hasKey("labelColor")) {
          val arr = options.getArray("labelColor")
          if (arr != null && arr.size() >= 3) {
            listOf(arr.getInt(0), arr.getInt(1), arr.getInt(2))
          } else listOf(255, 255, 255)
        } else listOf(255, 255, 255)

        val defaultColor = if (options.hasKey("defaultColor")) {
          val arr = options.getArray("defaultColor")
          if (arr != null && arr.size() >= 3) {
            listOf(arr.getInt(0), arr.getInt(1), arr.getInt(2))
          } else null
        } else null

        val quality = if (options.hasKey("quality")) options.getInt("quality") else 90

        val result = DrawingUtilsAndroid.drawBoxes(
          bitmap, boxesList, lineWidth, fontSize, drawLabels,
          labelBackgroundAlpha, labelColor, defaultColor, quality
        )

        val response = Arguments.createMap().apply {
          putString("imageBase64", result["imageBase64"] as String)
          putInt("width", result["width"] as Int)
          putInt("height", result["height"] as Int)
          putInt("boxesDrawn", result["boxesDrawn"] as Int)
          putDouble("processingTimeMs", result["processingTimeMs"] as Double)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("DRAW_BOXES_ERROR", e.message ?: "Failed to draw boxes")
        }
      }
    }
  }

  /**
   * Draw keypoints and skeleton on an image
   */
  override fun drawKeypoints(source: ReadableMap, keypoints: ReadableArray, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)

        val keypointsList = mutableListOf<Map<String, Any>>()
        for (i in 0 until keypoints.size()) {
          val kp = keypoints.getMap(i) ?: continue
          val point = mutableMapOf<String, Any>(
            "x" to kp.getDouble("x"),
            "y" to kp.getDouble("y")
          )
          if (kp.hasKey("confidence")) {
            point["confidence"] = kp.getDouble("confidence")
          }
          keypointsList.add(point)
        }

        val pointRadius = if (options.hasKey("pointRadius")) options.getDouble("pointRadius").toFloat() else 4f

        val pointColors = if (options.hasKey("pointColors")) {
          val arr = options.getArray("pointColors")
          if (arr != null) {
            val colors = mutableListOf<List<Int>>()
            for (i in 0 until arr.size()) {
              val c = arr.getArray(i)
              if (c != null && c.size() >= 3) {
                colors.add(listOf(c.getInt(0), c.getInt(1), c.getInt(2)))
              }
            }
            colors
          } else null
        } else null

        val skeleton = if (options.hasKey("skeleton")) {
          val arr = options.getArray("skeleton")
          if (arr != null) {
            val conns = mutableListOf<Map<String, Any>>()
            for (i in 0 until arr.size()) {
              val conn = arr.getMap(i) ?: continue
              val connection = mutableMapOf<String, Any>(
                "from" to conn.getInt("from"),
                "to" to conn.getInt("to")
              )
              if (conn.hasKey("color")) {
                val c = conn.getArray("color")
                if (c != null && c.size() >= 3) {
                  connection["color"] = listOf(c.getInt(0), c.getInt(1), c.getInt(2))
                }
              }
              conns.add(connection)
            }
            conns
          } else null
        } else null

        val lineWidth = if (options.hasKey("lineWidth")) options.getDouble("lineWidth").toFloat() else 2f
        val minConfidence = if (options.hasKey("minConfidence")) options.getDouble("minConfidence").toFloat() else 0f
        val quality = if (options.hasKey("quality")) options.getInt("quality") else 90

        val result = DrawingUtilsAndroid.drawKeypoints(
          bitmap, keypointsList, pointRadius, pointColors,
          skeleton, lineWidth, minConfidence, quality
        )

        val response = Arguments.createMap().apply {
          putString("imageBase64", result["imageBase64"] as String)
          putInt("width", result["width"] as Int)
          putInt("height", result["height"] as Int)
          putInt("pointsDrawn", result["pointsDrawn"] as Int)
          putDouble("processingTimeMs", result["processingTimeMs"] as Double)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("DRAW_KEYPOINTS_ERROR", e.message ?: "Failed to draw keypoints")
        }
      }
    }
  }

  /**
   * Overlay a segmentation mask on an image
   */
  override fun overlayMask(source: ReadableMap, mask: ReadableArray, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)

        val maskList = mutableListOf<Int>()
        for (i in 0 until mask.size()) {
          maskList.add(mask.getInt(i))
        }

        val maskWidth = options.getInt("maskWidth")
        val maskHeight = options.getInt("maskHeight")
        val alpha = if (options.hasKey("alpha")) options.getDouble("alpha").toFloat() else 0.5f

        val colorMap = if (options.hasKey("colorMap")) {
          val arr = options.getArray("colorMap")
          if (arr != null) {
            val colors = mutableListOf<List<Int>>()
            for (i in 0 until arr.size()) {
              val c = arr.getArray(i)
              if (c != null && c.size() >= 3) {
                colors.add(listOf(c.getInt(0), c.getInt(1), c.getInt(2)))
              }
            }
            colors
          } else null
        } else null

        val singleColor = if (options.hasKey("singleColor")) {
          val arr = options.getArray("singleColor")
          if (arr != null && arr.size() >= 3) {
            listOf(arr.getInt(0), arr.getInt(1), arr.getInt(2))
          } else null
        } else null

        val isClassMask = if (options.hasKey("isClassMask")) options.getBoolean("isClassMask") else true
        val quality = if (options.hasKey("quality")) options.getInt("quality") else 90

        val result = DrawingUtilsAndroid.overlayMask(
          bitmap, maskList, maskWidth, maskHeight, alpha,
          colorMap, singleColor, isClassMask, quality
        )

        val response = Arguments.createMap().apply {
          putString("imageBase64", result["imageBase64"] as String)
          putInt("width", result["width"] as Int)
          putInt("height", result["height"] as Int)
          putDouble("processingTimeMs", result["processingTimeMs"] as Double)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("OVERLAY_MASK_ERROR", e.message ?: "Failed to overlay mask")
        }
      }
    }
  }

  /**
   * Overlay a heatmap on an image
   */
  override fun overlayHeatmap(source: ReadableMap, heatmap: ReadableArray, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)

        val heatmapList = mutableListOf<Double>()
        for (i in 0 until heatmap.size()) {
          heatmapList.add(heatmap.getDouble(i))
        }

        val heatmapWidth = options.getInt("heatmapWidth")
        val heatmapHeight = options.getInt("heatmapHeight")
        val alpha = if (options.hasKey("alpha")) options.getDouble("alpha").toFloat() else 0.5f
        val colorScheme = if (options.hasKey("colorScheme")) options.getString("colorScheme") ?: "jet" else "jet"
        val minValue = if (options.hasKey("minValue")) options.getDouble("minValue") else null
        val maxValue = if (options.hasKey("maxValue")) options.getDouble("maxValue") else null
        val quality = if (options.hasKey("quality")) options.getInt("quality") else 90

        val result = DrawingUtilsAndroid.overlayHeatmap(
          bitmap, heatmapList, heatmapWidth, heatmapHeight,
          alpha, colorScheme, minValue, maxValue, quality
        )

        val response = Arguments.createMap().apply {
          putString("imageBase64", result["imageBase64"] as String)
          putInt("width", result["width"] as Int)
          putInt("height", result["height"] as Int)
          putDouble("processingTimeMs", result["processingTimeMs"] as Double)
        }

        withContext(Dispatchers.Main) {
          promise.resolve(response)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("OVERLAY_HEATMAP_ERROR", e.message ?: "Failed to overlay heatmap")
        }
      }
    }
  }

  /**
   * Detect blur in an image using Laplacian variance
   */
  override fun detectBlur(source: ReadableMap, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val context = reactApplicationContext.applicationContext
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)

        val threshold = if (options.hasKey("threshold")) options.getDouble("threshold") else 100.0
        val downsampleSize = if (options.hasKey("downsampleSize")) options.getInt("downsampleSize") else null

        val result = BlurDetectorAndroid.detectBlur(bitmap, threshold, downsampleSize)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("BLUR_DETECTION_ERROR", e.message ?: "Failed to detect blur")
        }
      }
    }
  }

  /**
   * Extract frames from a video file at specific timestamps or intervals
   */
  override fun extractVideoFrames(source: ReadableMap, options: ReadableMap, promise: Promise) {
    scope.launch {
      try {
        val result = VideoFrameExtractorAndroid.extractFrames(source, options)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("VIDEO_EXTRACTION_ERROR", e.message ?: "Failed to extract video frames")
        }
      }
    }
  }

  /**
   * Extract patches from an image in a grid pattern.
   * Useful for sliding window inference or creating training patches.
   *
   * @param source Image source (uri or base64)
   * @param gridOptions Grid extraction options (patchWidth, patchHeight, strideX, strideY, includePartial)
   * @param pixelOptions Pixel data output options
   */
  override fun extractGrid(
    source: ReadableMap,
    gridOptions: ReadableMap,
    pixelOptions: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val result = GridExtractorAndroid.extractGrid(source, gridOptions, pixelOptions, reactApplicationContext)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("GRID_EXTRACTION_ERROR", e.message ?: "Failed to extract grid")
        }
      }
    }
  }

  /**
   * Extract random crops from an image.
   * Useful for data augmentation with reproducible results when using seeds.
   *
   * @param source Image source (uri or base64)
   * @param cropOptions Random crop options (width, height, count, seed)
   * @param pixelOptions Pixel data output options
   */
  override fun randomCrop(
    source: ReadableMap,
    cropOptions: ReadableMap,
    pixelOptions: ReadableMap,
    promise: Promise
  ) {
    scope.launch {
      try {
        val result = RandomCropperAndroid.randomCrop(source, cropOptions, pixelOptions, reactApplicationContext)

        withContext(Dispatchers.Main) {
          promise.resolve(result)
        }
      } catch (e: VisionUtilsException) {
        withContext(Dispatchers.Main) {
          promise.reject(e.code, e.message)
        }
      } catch (e: Exception) {
        withContext(Dispatchers.Main) {
          promise.reject("RANDOM_CROP_ERROR", e.message ?: "Failed to extract random crop")
        }
      }
    }
  }

  companion object {
    const val NAME = NativeVisionUtilsSpec.NAME
  }
}
