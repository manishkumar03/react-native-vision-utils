package com.visionutils

import android.content.Context
import android.graphics.Bitmap
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.min

/**
 * GridExtractor provides functionality for extracting image patches in a grid pattern.
 *
 * This is useful for:
 * - Sliding window inference on large images
 * - Creating training patches from images
 * - Tiled processing of high-resolution images
 *
 * Example Usage (from JS):
 * ```typescript
 * const result = await extractGrid(
 *   { uri: 'file://image.jpg' },
 *   { rows: 4, columns: 4, overlap: 32, includePartial: false },
 *   { outputFormat: 'float32', layout: 'NCHW' }
 * );
 * ```
 */
object GridExtractorAndroid {

    /**
     * Extract patches from an image in a grid pattern.
     *
     * @param source Source specification (uri or base64)
     * @param gridOptions Grid extraction options (rows, columns, overlap, includePartial)
     * @param pixelOptions Pixel data output options
     * @param context Android context for loading images
     * @return WritableMap containing patches array and metadata
     * @throws VisionUtilsException on invalid input or processing failure
     */
    suspend fun extractGrid(
        source: ReadableMap,
        gridOptions: ReadableMap,
        pixelOptions: ReadableMap,
        context: Context
    ): WritableMap = withContext(Dispatchers.Default) {
        val startTimeNs = System.nanoTime()

        // Parse and load the source image
        val imageSource = ImageSource.fromMap(source)
        val bitmap = ImageLoader.loadImage(context, imageSource)

        try {
            // Parse grid options
            val columns = if (gridOptions.hasKey("columns")) gridOptions.getInt("columns") else {
                throw VisionUtilsException("INVALID_OPTIONS", "columns must be specified")
            }
            val rows = if (gridOptions.hasKey("rows")) gridOptions.getInt("rows") else {
                throw VisionUtilsException("INVALID_OPTIONS", "rows must be specified")
            }

            if (columns < 1 || rows < 1) {
                throw VisionUtilsException("INVALID_OPTIONS", "rows and columns must be at least 1")
            }

            val overlap = if (gridOptions.hasKey("overlap")) gridOptions.getInt("overlap") else 0
            val overlapPercent = if (gridOptions.hasKey("overlapPercent") && !gridOptions.isNull("overlapPercent"))
                gridOptions.getDouble("overlapPercent") else null
            val includePartial = if (gridOptions.hasKey("includePartial")) gridOptions.getBoolean("includePartial") else false

            val imageWidth = bitmap.width
            val imageHeight = bitmap.height
            val parsedPixelOptions = GetPixelDataOptions.fromMap(pixelOptions)

            // Calculate effective overlap
            val effectiveOverlap = if (overlapPercent != null) {
                val basePatchWidth = imageWidth / columns
                val basePatchHeight = imageHeight / rows
                (min(basePatchWidth, basePatchHeight) * overlapPercent).toInt()
            } else {
                overlap
            }

            // Calculate patch dimensions
            // imageSize = patchSize * count - overlap * (count - 1)
            // patchSize = (imageSize + overlap * (count - 1)) / count
            val patchWidth = (imageWidth + effectiveOverlap * (columns - 1)) / columns
            val patchHeight = (imageHeight + effectiveOverlap * (rows - 1)) / rows

            if (patchWidth <= 0 || patchHeight <= 0) {
                throw VisionUtilsException("INVALID_DIMENSIONS", "Calculated patch dimensions are invalid")
            }

            // Calculate stride
            val strideX = max(1, patchWidth - effectiveOverlap)
            val strideY = max(1, patchHeight - effectiveOverlap)

            // Extract patches
            val patches = Arguments.createArray()

            for (row in 0 until rows) {
                for (col in 0 until columns) {
                    val x = col * strideX
                    val y = row * strideY

                    // Check if this patch would be partial
                    val isPartialX = (x + patchWidth) > imageWidth
                    val isPartialY = (y + patchHeight) > imageHeight

                    if ((isPartialX || isPartialY) && !includePartial) {
                        continue
                    }

                    // Calculate actual crop dimensions
                    val actualWidth = min(patchWidth, imageWidth - x)
                    val actualHeight = min(patchHeight, imageHeight - y)

                    // Extract the patch
                    val patchBitmap = Bitmap.createBitmap(bitmap, x, y, actualWidth, actualHeight)

                    // Handle partial patches by padding if needed
                    val finalBitmap = if (actualWidth < patchWidth || actualHeight < patchHeight) {
                        padBitmap(patchBitmap, patchWidth, patchHeight)
                    } else {
                        patchBitmap
                    }

                    // Get pixel data for this patch
                    val pixelResult = PixelProcessor.process(finalBitmap, parsedPixelOptions)

                    // Convert FloatArray to WritableArray
                    val dataArray = Arguments.createArray()
                    pixelResult.data.forEach { dataArray.pushDouble(it.toDouble()) }

                    // Build patch info matching GridPatch interface
                    val patchInfo = Arguments.createMap().apply {
                        putInt("row", row)
                        putInt("column", col)
                        putInt("x", x)
                        putInt("y", y)
                        putInt("width", actualWidth)
                        putInt("height", actualHeight)
                        putArray("data", dataArray)
                    }

                    patches.pushMap(patchInfo)

                    // Clean up intermediate bitmaps
                    if (finalBitmap !== patchBitmap && finalBitmap !== bitmap) {
                        finalBitmap.recycle()
                    }
                    if (patchBitmap !== bitmap) {
                        patchBitmap.recycle()
                    }
                }
            }

            val totalTimeMs = (System.nanoTime() - startTimeNs) / 1_000_000.0
            val patchCount = patches.size()

            Arguments.createMap().apply {
                putArray("patches", patches)
                putInt("patchCount", patchCount)
                putInt("columns", columns)
                putInt("rows", rows)
                putInt("originalWidth", imageWidth)
                putInt("originalHeight", imageHeight)
                putInt("patchWidth", patchWidth)
                putInt("patchHeight", patchHeight)
                putDouble("processingTimeMs", totalTimeMs)
            }
        } finally {
            bitmap.recycle()
        }
    }

    /**
     * Pad a bitmap to the target size (adds black padding to right/bottom).
     */
    private fun padBitmap(source: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val padded = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(padded)
        // Fill with black background (default is transparent, so fill explicitly)
        canvas.drawColor(android.graphics.Color.BLACK)
        // Draw the source at top-left
        canvas.drawBitmap(source, 0f, 0f, null)
        return padded
    }
}
