package com.visionutils

import android.content.Context
import android.graphics.Bitmap
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.random.Random

/**
 * RandomCropper provides functionality for extracting random crops from images.
 *
 * This is useful for:
 * - Data augmentation during training
 * - Creating diverse training samples
 * - Reproducible random crops using seeds
 *
 * Example Usage (from JS):
 * ```typescript
 * const result = await randomCrop(
 *   { uri: 'file://image.jpg' },
 *   { width: 224, height: 224, count: 5, seed: 42 },
 *   { outputFormat: 'float32', layout: 'NCHW' }
 * );
 * ```
 */
object RandomCropperAndroid {

    /**
     * Extract random crops from an image.
     *
     * @param source Source specification (uri or base64)
     * @param cropOptions Random crop options
     * @param pixelOptions Pixel data output options
     * @param context Android context for loading images
     * @return WritableMap containing crops array and metadata
     * @throws VisionUtilsException on invalid input or processing failure
     */
    suspend fun randomCrop(
        source: ReadableMap,
        cropOptions: ReadableMap,
        pixelOptions: ReadableMap,
        context: Context
    ): WritableMap = withContext(Dispatchers.Default) {
        val startTimeNs = System.nanoTime()

        // Parse and load the source image
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)

        try {
            // Parse crop options with defaults
            val cropWidth = if (cropOptions.hasKey("width")) cropOptions.getInt("width") else 224
            val cropHeight = if (cropOptions.hasKey("height")) cropOptions.getInt("height") else 224
            val count = if (cropOptions.hasKey("count")) cropOptions.getInt("count") else 1
            val seed = if (cropOptions.hasKey("seed") && !cropOptions.isNull("seed")) cropOptions.getInt("seed") else null

            // Validate dimensions
            if (cropWidth <= 0 || cropHeight <= 0) {
                throw VisionUtilsException("INVALID_DIMENSIONS", "Crop dimensions must be positive")
            }
            if (count <= 0) {
                throw VisionUtilsException("INVALID_COUNT", "Count must be positive")
            }

            val imageWidth = bitmap.width
            val imageHeight = bitmap.height

            // Check if image is large enough for the requested crop size
            if (imageWidth < cropWidth || imageHeight < cropHeight) {
                throw VisionUtilsException(
                    "IMAGE_TOO_SMALL",
                    "Image (${imageWidth}x${imageHeight}) is smaller than requested crop size (${cropWidth}x${cropHeight})"
                )
            }

            // Setup random number generator with optional seed
            val baseSeed = seed?.toLong() ?: Random.nextLong()
            val random = Random(baseSeed)
            val effectiveSeed = ((baseSeed % Int.MAX_VALUE).toInt()).let { if (it < 0) -it else it }
            val parsedPixelOptions = GetPixelDataOptions.fromMap(pixelOptions)

            // Generate random crops
            val crops = Arguments.createArray()

            for (i in 0 until count) {
                // Generate random position
                val maxX = imageWidth - cropWidth
                val maxY = imageHeight - cropHeight
                val x = if (maxX > 0) random.nextInt(maxX + 1) else 0
                val y = if (maxY > 0) random.nextInt(maxY + 1) else 0

                // Extract the crop
                val cropBitmap = Bitmap.createBitmap(bitmap, x, y, cropWidth, cropHeight)

                // Get pixel data for this crop
                val pixelResult = PixelProcessor.process(cropBitmap, parsedPixelOptions)

                // Convert FloatArray to WritableArray
                val dataArray = Arguments.createArray()
                pixelResult.data.forEach { dataArray.pushDouble(it.toDouble()) }

                // Build crop info
                val cropInfo = Arguments.createMap().apply {
                    putInt("index", i)
                    putInt("x", x)
                    putInt("y", y)
                    putInt("width", cropWidth)
                    putInt("height", cropHeight)
                    putArray("data", dataArray)
                    putInt("seed", effectiveSeed)
                }

                crops.pushMap(cropInfo)

                // Clean up
                cropBitmap.recycle()
            }

            val processingTimeMs = (System.nanoTime() - startTimeNs) / 1_000_000.0

            Arguments.createMap().apply {
                putArray("crops", crops)
                putInt("cropCount", count)
                putInt("cropWidth", cropWidth)
                putInt("cropHeight", cropHeight)
                putInt("seed", effectiveSeed)
                putInt("originalWidth", imageWidth)
                putInt("originalHeight", imageHeight)
                putDouble("processingTimeMs", processingTimeMs)
            }
        } finally {
            bitmap.recycle()
        }
    }
}
