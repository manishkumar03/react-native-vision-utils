package com.visionutils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import kotlin.math.min
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt

/**
 * Processes bitmap images to extract and transform pixel data
 */
object PixelProcessor {

    /**
     * Process an image with the given options
     */
    fun process(bitmap: Bitmap, options: GetPixelDataOptions): VisionUtilsResult {
        val startTime = System.nanoTime()

        var processedBitmap = bitmap

        // Apply ROI if specified
        options.roi?.let { roi ->
            processedBitmap = applyRoi(processedBitmap, roi)
        }

        // Apply resize if specified
        options.resize?.let { resize ->
            processedBitmap = applyResize(processedBitmap, resize)
        }

        // Extract pixel data as RGBA
        val rgbaPixels = extractRgbaPixels(processedBitmap)

        // Convert to target color format
        val colorData = convertColorFormat(
            rgbaPixels,
            processedBitmap.width,
            processedBitmap.height,
            options.colorFormat
        )

        // Apply normalization
        val normalizedData = applyNormalization(
            colorData,
            options.colorFormat,
            options.normalization
        )

        // Convert to target data layout
        val layoutData = convertLayout(
            normalizedData,
            processedBitmap.width,
            processedBitmap.height,
            options.colorFormat.channels,
            options.dataLayout
        )

        // Calculate shape based on layout
        val shape = calculateShape(
            processedBitmap.width,
            processedBitmap.height,
            options.colorFormat.channels,
            options.dataLayout
        )

        val endTime = System.nanoTime()
        val processingTimeMs = (endTime - startTime) / 1_000_000.0

        return VisionUtilsResult(
            data = layoutData,
            width = processedBitmap.width,
            height = processedBitmap.height,
            channels = options.colorFormat.channels,
            colorFormat = options.colorFormat,
            dataLayout = options.dataLayout,
            shape = shape,
            processingTimeMs = processingTimeMs
        )
    }

    /**
     * Apply region of interest cropping
     */
    private fun applyRoi(bitmap: Bitmap, roi: Roi): Bitmap {
        // Validate ROI bounds
        if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0) {
            throw VisionUtilsException(
                "INVALID_ROI",
                "ROI dimensions must be positive and coordinates non-negative"
            )
        }

        if (roi.x + roi.width > bitmap.width || roi.y + roi.height > bitmap.height) {
            throw VisionUtilsException(
                "INVALID_ROI",
                "ROI extends beyond image bounds (image: ${bitmap.width}x${bitmap.height}, ROI: x=${roi.x}, y=${roi.y}, w=${roi.width}, h=${roi.height})"
            )
        }

        return Bitmap.createBitmap(bitmap, roi.x, roi.y, roi.width, roi.height)
    }

    /**
     * Apply resize with the specified strategy
     */
    private fun applyResize(bitmap: Bitmap, resize: ResizeOptions): Bitmap {
        val targetWidth = resize.width
        val targetHeight = resize.height

        return when (resize.strategy) {
            ResizeStrategy.STRETCH -> {
                Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
            }
            ResizeStrategy.COVER -> {
                resizeCover(bitmap, targetWidth, targetHeight)
            }
            ResizeStrategy.CONTAIN -> {
                resizeContain(bitmap, targetWidth, targetHeight, resize.padColor)
            }
            ResizeStrategy.LETTERBOX -> {
                resizeLetterbox(bitmap, targetWidth, targetHeight, resize.letterboxColor)
            }
        }
    }

    /**
     * Resize with cover strategy (fill target, crop excess)
     */
    private fun resizeCover(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val sourceWidth = bitmap.width.toFloat()
        val sourceHeight = bitmap.height.toFloat()

        val scaleX = targetWidth / sourceWidth
        val scaleY = targetHeight / sourceHeight
        val scale = max(scaleX, scaleY)

        val scaledWidth = (sourceWidth * scale).roundToInt()
        val scaledHeight = (sourceHeight * scale).roundToInt()

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)

        // Center crop
        val x = (scaledWidth - targetWidth) / 2
        val y = (scaledHeight - targetHeight) / 2

        return Bitmap.createBitmap(scaledBitmap, x, y, targetWidth, targetHeight)
    }

    /**
     * Resize with contain strategy (fit within target, letterbox)
     */
    private fun resizeContain(bitmap: Bitmap, targetWidth: Int, targetHeight: Int, padColor: IntArray): Bitmap {
        val sourceWidth = bitmap.width.toFloat()
        val sourceHeight = bitmap.height.toFloat()

        val scaleX = targetWidth / sourceWidth
        val scaleY = targetHeight / sourceHeight
        val scale = min(scaleX, scaleY)

        val scaledWidth = (sourceWidth * scale).roundToInt()
        val scaledHeight = (sourceHeight * scale).roundToInt()

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)

        // Create target bitmap with padding color
        val result = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val r = padColor.getOrElse(0) { 0 }
        val g = padColor.getOrElse(1) { 0 }
        val b = padColor.getOrElse(2) { 0 }
        canvas.drawColor(Color.rgb(r, g, b))

        // Center the scaled bitmap
        val x = (targetWidth - scaledWidth) / 2
        val y = (targetHeight - scaledHeight) / 2
        canvas.drawBitmap(scaledBitmap, x.toFloat(), y.toFloat(), null)

        return result
    }

    /**
     * Resize with letterbox strategy (same as contain but with explicit padding)
     * Similar to YOLO preprocessing - maintains aspect ratio with padding
     */
    private fun resizeLetterbox(bitmap: Bitmap, targetWidth: Int, targetHeight: Int, letterboxColor: IntArray): Bitmap {
        val sourceWidth = bitmap.width.toFloat()
        val sourceHeight = bitmap.height.toFloat()

        val scaleX = targetWidth / sourceWidth
        val scaleY = targetHeight / sourceHeight
        val scale = min(scaleX, scaleY)

        val scaledWidth = (sourceWidth * scale).roundToInt()
        val scaledHeight = (sourceHeight * scale).roundToInt()

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)

        // Create target bitmap with letterbox padding color
        val result = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val r = letterboxColor.getOrElse(0) { 114 }
        val g = letterboxColor.getOrElse(1) { 114 }
        val b = letterboxColor.getOrElse(2) { 114 }
        canvas.drawColor(Color.rgb(r, g, b)) // Default YOLO-style gray padding

        // Center the scaled bitmap
        val x = (targetWidth - scaledWidth) / 2
        val y = (targetHeight - scaledHeight) / 2
        canvas.drawBitmap(scaledBitmap, x.toFloat(), y.toFloat(), null)

        return result
    }

    /**
     * Extract RGBA pixel values from bitmap
     */
    private fun extractRgbaPixels(bitmap: Bitmap): IntArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return pixels
    }

    /**
     * Convert ARGB pixels to target color format
     */
    private fun convertColorFormat(
        argbPixels: IntArray,
        width: Int,
        height: Int,
        format: ColorFormat
    ): FloatArray {
        val channels = format.channels
        val result = FloatArray(width * height * channels)

        for (i in argbPixels.indices) {
            val argb = argbPixels[i]
            val a = (argb shr 24) and 0xFF
            val r = (argb shr 16) and 0xFF
            val g = (argb shr 8) and 0xFF
            val b = argb and 0xFF

            val baseIdx = i * channels

            when (format) {
                ColorFormat.RGB -> {
                    result[baseIdx] = r.toFloat()
                    result[baseIdx + 1] = g.toFloat()
                    result[baseIdx + 2] = b.toFloat()
                }
                ColorFormat.RGBA -> {
                    result[baseIdx] = r.toFloat()
                    result[baseIdx + 1] = g.toFloat()
                    result[baseIdx + 2] = b.toFloat()
                    result[baseIdx + 3] = a.toFloat()
                }
                ColorFormat.BGR -> {
                    result[baseIdx] = b.toFloat()
                    result[baseIdx + 1] = g.toFloat()
                    result[baseIdx + 2] = r.toFloat()
                }
                ColorFormat.BGRA -> {
                    result[baseIdx] = b.toFloat()
                    result[baseIdx + 1] = g.toFloat()
                    result[baseIdx + 2] = r.toFloat()
                    result[baseIdx + 3] = a.toFloat()
                }
                ColorFormat.GRAYSCALE -> {
                    // ITU-R BT.601 formula for luminance
                    result[baseIdx] = 0.299f * r + 0.587f * g + 0.114f * b
                }
                ColorFormat.HSV -> {
                    val hsv = rgbToHsv(r, g, b)
                    result[baseIdx] = hsv[0]
                    result[baseIdx + 1] = hsv[1]
                    result[baseIdx + 2] = hsv[2]
                }
                ColorFormat.HSL -> {
                    val hsl = rgbToHsl(r, g, b)
                    result[baseIdx] = hsl[0]
                    result[baseIdx + 1] = hsl[1]
                    result[baseIdx + 2] = hsl[2]
                }
                ColorFormat.LAB -> {
                    val lab = rgbToLab(r, g, b)
                    result[baseIdx] = lab[0]
                    result[baseIdx + 1] = lab[1]
                    result[baseIdx + 2] = lab[2]
                }
                ColorFormat.YUV -> {
                    val yuv = rgbToYuv(r, g, b)
                    result[baseIdx] = yuv[0]
                    result[baseIdx + 1] = yuv[1]
                    result[baseIdx + 2] = yuv[2]
                }
                ColorFormat.YCBCR -> {
                    val ycbcr = rgbToYCbCr(r, g, b)
                    result[baseIdx] = ycbcr[0]
                    result[baseIdx + 1] = ycbcr[1]
                    result[baseIdx + 2] = ycbcr[2]
                }
            }
        }

        return result
    }

    /**
     * Convert RGB to HSV color space
     */
    private fun rgbToHsv(r: Int, g: Int, b: Int): FloatArray {
        val rf = r / 255f
        val gf = g / 255f
        val bf = b / 255f

        val maxVal = maxOf(rf, gf, bf)
        val minVal = minOf(rf, gf, bf)
        val delta = maxVal - minVal

        var h = 0f
        val s = if (maxVal == 0f) 0f else delta / maxVal
        val v = maxVal

        if (delta != 0f) {
            h = when (maxVal) {
                rf -> 60f * (((gf - bf) / delta) % 6)
                gf -> 60f * ((bf - rf) / delta + 2)
                else -> 60f * ((rf - gf) / delta + 4)
            }
            if (h < 0) h += 360f
        }

        return floatArrayOf(h, s * 255f, v * 255f)
    }

    /**
     * Convert RGB to HSL color space
     */
    private fun rgbToHsl(r: Int, g: Int, b: Int): FloatArray {
        val rf = r / 255f
        val gf = g / 255f
        val bf = b / 255f

        val maxVal = maxOf(rf, gf, bf)
        val minVal = minOf(rf, gf, bf)
        val delta = maxVal - minVal
        val l = (maxVal + minVal) / 2

        var h = 0f
        var s = 0f

        if (delta != 0f) {
            s = if (l <= 0.5f) delta / (maxVal + minVal) else delta / (2 - maxVal - minVal)

            h = when (maxVal) {
                rf -> 60f * (((gf - bf) / delta) % 6)
                gf -> 60f * ((bf - rf) / delta + 2)
                else -> 60f * ((rf - gf) / delta + 4)
            }
            if (h < 0) h += 360f
        }

        return floatArrayOf(h, s * 255f, l * 255f)
    }

    /**
     * Convert RGB to LAB color space
     */
    private fun rgbToLab(r: Int, g: Int, b: Int): FloatArray {
        // First convert to XYZ
        var rf = r / 255f
        var gf = g / 255f
        var bf = b / 255f

        // Apply gamma correction
        rf = if (rf > 0.04045f) ((rf + 0.055f) / 1.055f).toDouble().pow(2.4).toFloat() else rf / 12.92f
        gf = if (gf > 0.04045f) ((gf + 0.055f) / 1.055f).toDouble().pow(2.4).toFloat() else gf / 12.92f
        bf = if (bf > 0.04045f) ((bf + 0.055f) / 1.055f).toDouble().pow(2.4).toFloat() else bf / 12.92f

        // Convert to XYZ using D65 illuminant
        var x = (rf * 0.4124564f + gf * 0.3575761f + bf * 0.1804375f) / 0.95047f
        var y = (rf * 0.2126729f + gf * 0.7151522f + bf * 0.0721750f) / 1.00000f
        var z = (rf * 0.0193339f + gf * 0.1191920f + bf * 0.9503041f) / 1.08883f

        // Convert to LAB
        x = if (x > 0.008856f) x.toDouble().pow(1.0/3.0).toFloat() else (7.787f * x) + (16f / 116f)
        y = if (y > 0.008856f) y.toDouble().pow(1.0/3.0).toFloat() else (7.787f * y) + (16f / 116f)
        z = if (z > 0.008856f) z.toDouble().pow(1.0/3.0).toFloat() else (7.787f * z) + (16f / 116f)

        val lVal = (116f * y) - 16f
        val aVal = 500f * (x - y)
        val bVal = 200f * (y - z)

        return floatArrayOf(lVal, aVal + 128f, bVal + 128f)  // Shift a,b to 0-255 range
    }

    /**
     * Convert RGB to YUV color space
     */
    private fun rgbToYuv(r: Int, g: Int, b: Int): FloatArray {
        val yVal = 0.299f * r + 0.587f * g + 0.114f * b
        val uVal = -0.14713f * r - 0.28886f * g + 0.436f * b + 128f
        val vVal = 0.615f * r - 0.51499f * g - 0.10001f * b + 128f

        return floatArrayOf(yVal, uVal, vVal)
    }

    /**
     * Convert RGB to YCbCr color space (BT.601)
     */
    private fun rgbToYCbCr(r: Int, g: Int, b: Int): FloatArray {
        val yVal = 16f + (65.481f * r + 128.553f * g + 24.966f * b) / 255f
        val cbVal = 128f + (-37.797f * r - 74.203f * g + 112f * b) / 255f
        val crVal = 128f + (112f * r - 93.786f * g - 18.214f * b) / 255f

        return floatArrayOf(yVal, cbVal, crVal)
    }

    /**
     * Apply normalization to pixel data
     */
    private fun applyNormalization(
        data: FloatArray,
        format: ColorFormat,
        normalization: Normalization
    ): FloatArray {
        val channels = format.channels

        // Get mean and std based on preset
        val (mean, std) = when (normalization.preset) {
            NormalizationPreset.IMAGENET -> {
                Pair(
                    floatArrayOf(0.485f * 255f, 0.456f * 255f, 0.406f * 255f),
                    floatArrayOf(0.229f * 255f, 0.224f * 255f, 0.225f * 255f)
                )
            }
            NormalizationPreset.TENSORFLOW -> {
                Pair(
                    floatArrayOf(127.5f, 127.5f, 127.5f),
                    floatArrayOf(127.5f, 127.5f, 127.5f)
                )
            }
            NormalizationPreset.SCALE -> {
                Pair(
                    floatArrayOf(0f, 0f, 0f),
                    floatArrayOf(255f, 255f, 255f)
                )
            }
            NormalizationPreset.RAW -> {
                return data // No normalization
            }
            NormalizationPreset.CUSTOM -> {
                Pair(normalization.mean, normalization.std)
            }
        }

        val result = FloatArray(data.size)
        val numPixels = data.size / channels

        for (i in 0 until numPixels) {
            for (c in 0 until channels) {
                val idx = i * channels + c
                val channelIdx = min(c, mean.size - 1)
                result[idx] = (data[idx] - mean[channelIdx]) / std[channelIdx]
            }
        }

        return result
    }

    /**
     * Convert data layout from HWC to target format
     */
    private fun convertLayout(
        data: FloatArray,
        width: Int,
        height: Int,
        channels: Int,
        layout: DataLayout
    ): FloatArray {
        return when (layout) {
            DataLayout.HWC -> data // Already in HWC format
            DataLayout.CHW -> convertHwcToChw(data, width, height, channels)
            DataLayout.NHWC -> data // Same as HWC for single image (batch size 1)
            DataLayout.NCHW -> convertHwcToChw(data, width, height, channels)
        }
    }

    /**
     * Convert from HWC (Height x Width x Channels) to CHW (Channels x Height x Width)
     */
    private fun convertHwcToChw(
        hwcData: FloatArray,
        width: Int,
        height: Int,
        channels: Int
    ): FloatArray {
        val chwData = FloatArray(hwcData.size)

        for (h in 0 until height) {
            for (w in 0 until width) {
                for (c in 0 until channels) {
                    val hwcIdx = (h * width + w) * channels + c
                    val chwIdx = c * height * width + h * width + w
                    chwData[chwIdx] = hwcData[hwcIdx]
                }
            }
        }

        return chwData
    }

    /**
     * Calculate shape array based on data layout
     */
    private fun calculateShape(
        width: Int,
        height: Int,
        channels: Int,
        layout: DataLayout
    ): IntArray {
        return when (layout) {
            DataLayout.HWC -> intArrayOf(height, width, channels)
            DataLayout.CHW -> intArrayOf(channels, height, width)
            DataLayout.NHWC -> intArrayOf(1, height, width, channels)
            DataLayout.NCHW -> intArrayOf(1, channels, height, width)
        }
    }
}
