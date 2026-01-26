package com.visionutils

import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import com.facebook.react.bridge.Arguments

/**
 * Image source types supported by the library
 */
enum class ImageSourceType {
    URL,
    FILE,
    BASE64,
    ASSET,
    PHOTO_LIBRARY;

    companion object {
        fun fromString(value: String): ImageSourceType {
            return when (value.lowercase()) {
                "url" -> URL
                "file" -> FILE
                "base64" -> BASE64
                "asset" -> ASSET
                "photolibrary", "photo_library" -> PHOTO_LIBRARY
                else -> throw VisionUtilsException("INVALID_SOURCE", "Unknown image source type: $value")
            }
        }
    }
}

/**
 * Image source with type and value
 */
data class ImageSource(
    val type: ImageSourceType,
    val value: String
) {
    companion object {
        fun fromMap(map: ReadableMap): ImageSource {
            val type = map.getString("type")
                ?: throw VisionUtilsException("INVALID_SOURCE", "Missing source type")
            val value = map.getString("value")
                ?: throw VisionUtilsException("INVALID_SOURCE", "Missing source value")
            return ImageSource(ImageSourceType.fromString(type), value)
        }
    }
}

/**
 * Color format for pixel data output
 */
enum class ColorFormat {
    RGB,
    RGBA,
    BGR,
    BGRA,
    GRAYSCALE,
    HSV,
    HSL,
    LAB,
    YUV,
    YCBCR;

    companion object {
        fun fromString(value: String): ColorFormat {
            return when (value.uppercase()) {
                "RGB" -> RGB
                "RGBA" -> RGBA
                "BGR" -> BGR
                "BGRA" -> BGRA
                "GRAYSCALE" -> GRAYSCALE
                "HSV" -> HSV
                "HSL" -> HSL
                "LAB" -> LAB
                "YUV" -> YUV
                "YCBCR" -> YCBCR
                else -> RGB // Default
            }
        }
    }

    val channels: Int
        get() = when (this) {
            RGB, BGR, HSV, HSL, LAB, YUV, YCBCR -> 3
            RGBA, BGRA -> 4
            GRAYSCALE -> 1
        }
}

/**
 * Resize strategy
 */
enum class ResizeStrategy {
    COVER,
    CONTAIN,
    STRETCH,
    LETTERBOX;

    companion object {
        fun fromString(value: String): ResizeStrategy {
            return when (value.lowercase()) {
                "cover" -> COVER
                "contain" -> CONTAIN
                "stretch" -> STRETCH
                "letterbox" -> LETTERBOX
                else -> COVER // Default
            }
        }
    }
}

/**
 * Resize options
 */
data class ResizeOptions(
    val width: Int,
    val height: Int,
    val strategy: ResizeStrategy = ResizeStrategy.COVER,
    val padColor: IntArray = intArrayOf(0, 0, 0, 255),
    val letterboxColor: IntArray = intArrayOf(114, 114, 114)
) {
    companion object {
        fun fromMap(map: ReadableMap?): ResizeOptions? {
            if (map == null) return null
            val width = if (map.hasKey("width")) map.getInt("width") else return null
            val height = if (map.hasKey("height")) map.getInt("height") else return null
            val strategy = if (map.hasKey("strategy")) {
                ResizeStrategy.fromString(map.getString("strategy") ?: "cover")
            } else {
                ResizeStrategy.COVER
            }
            val padColor = if (map.hasKey("padColor")) {
                map.getArray("padColor")?.toIntArray() ?: intArrayOf(0, 0, 0, 255)
            } else {
                intArrayOf(0, 0, 0, 255)
            }
            val letterboxColor = if (map.hasKey("letterboxColor")) {
                map.getArray("letterboxColor")?.toIntArray() ?: intArrayOf(114, 114, 114)
            } else {
                intArrayOf(114, 114, 114)
            }
            return ResizeOptions(width, height, strategy, padColor, letterboxColor)
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as ResizeOptions
        return width == other.width &&
               height == other.height &&
               strategy == other.strategy &&
               padColor.contentEquals(other.padColor) &&
               letterboxColor.contentEquals(other.letterboxColor)
    }

    override fun hashCode(): Int {
        var result = width
        result = 31 * result + height
        result = 31 * result + strategy.hashCode()
        result = 31 * result + padColor.contentHashCode()
        result = 31 * result + letterboxColor.contentHashCode()
        return result
    }
}

/**
 * Region of interest for cropping
 */
data class Roi(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int
) {
    companion object {
        fun fromMap(map: ReadableMap?): Roi? {
            if (map == null) return null
            val x = if (map.hasKey("x")) map.getInt("x") else 0
            val y = if (map.hasKey("y")) map.getInt("y") else 0
            val width = if (map.hasKey("width")) map.getInt("width") else return null
            val height = if (map.hasKey("height")) map.getInt("height") else return null
            return Roi(x, y, width, height)
        }
    }
}

/**
 * Normalization presets
 */
enum class NormalizationPreset {
    IMAGENET,
    TENSORFLOW,
    SCALE,
    RAW,
    CUSTOM;

    companion object {
        fun fromString(value: String): NormalizationPreset {
            return when (value.lowercase()) {
                "imagenet" -> IMAGENET
                "tensorflow" -> TENSORFLOW
                "scale" -> SCALE
                "raw" -> RAW
                "custom" -> CUSTOM
                else -> SCALE
            }
        }
    }
}

/**
 * Normalization configuration
 */
data class Normalization(
    val preset: NormalizationPreset = NormalizationPreset.SCALE,
    val mean: FloatArray = floatArrayOf(0f, 0f, 0f),
    val std: FloatArray = floatArrayOf(1f, 1f, 1f)
) {
    companion object {
        fun fromMap(map: ReadableMap?): Normalization {
            if (map == null) return Normalization()

            val preset = if (map.hasKey("preset")) {
                NormalizationPreset.fromString(map.getString("preset") ?: "scale")
            } else {
                NormalizationPreset.SCALE
            }

            val mean = if (map.hasKey("mean")) {
                map.getArray("mean")?.toFloatArray() ?: floatArrayOf(0f, 0f, 0f)
            } else {
                floatArrayOf(0f, 0f, 0f)
            }

            val std = if (map.hasKey("std")) {
                map.getArray("std")?.toFloatArray() ?: floatArrayOf(1f, 1f, 1f)
            } else {
                floatArrayOf(1f, 1f, 1f)
            }

            return Normalization(preset, mean, std)
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as Normalization
        return preset == other.preset &&
               mean.contentEquals(other.mean) &&
               std.contentEquals(other.std)
    }

    override fun hashCode(): Int {
        var result = preset.hashCode()
        result = 31 * result + mean.contentHashCode()
        result = 31 * result + std.contentHashCode()
        return result
    }
}

/**
 * Data layout format
 */
enum class DataLayout {
    HWC,  // Height x Width x Channels (standard)
    CHW,  // Channels x Height x Width (PyTorch)
    NHWC, // Batch x Height x Width x Channels (TensorFlow)
    NCHW; // Batch x Channels x Height x Width

    companion object {
        fun fromString(value: String): DataLayout {
            return when (value.uppercase()) {
                "HWC" -> HWC
                "CHW" -> CHW
                "NHWC" -> NHWC
                "NCHW" -> NCHW
                else -> HWC
            }
        }
    }
}

/**
 * Output format type
 */
enum class OutputFormat {
    ARRAY,
    FLOAT32_ARRAY,
    UINT8_ARRAY;

    companion object {
        fun fromString(value: String): OutputFormat {
            return when (value.lowercase()) {
                "array" -> ARRAY
                "float32array", "float32_array" -> FLOAT32_ARRAY
                "uint8array", "uint8_array" -> UINT8_ARRAY
                else -> ARRAY
            }
        }
    }
}

/**
 * Complete options for getPixelData
 */
data class GetPixelDataOptions(
    val source: ImageSource,
    val colorFormat: ColorFormat = ColorFormat.RGB,
    val resize: ResizeOptions? = null,
    val roi: Roi? = null,
    val normalization: Normalization = Normalization(),
    val dataLayout: DataLayout = DataLayout.HWC,
    val outputFormat: OutputFormat = OutputFormat.ARRAY
) {
    companion object {
        fun fromMap(map: ReadableMap): GetPixelDataOptions {
            val source = if (map.hasKey("source")) {
                ImageSource.fromMap(map.getMap("source")!!)
            } else {
                throw VisionUtilsException("INVALID_SOURCE", "Missing source in options")
            }

            val colorFormat = if (map.hasKey("colorFormat")) {
                ColorFormat.fromString(map.getString("colorFormat") ?: "RGB")
            } else {
                ColorFormat.RGB
            }

            val resize = if (map.hasKey("resize")) {
                ResizeOptions.fromMap(map.getMap("resize"))
            } else {
                null
            }

            val roi = if (map.hasKey("roi")) {
                Roi.fromMap(map.getMap("roi"))
            } else {
                null
            }

            val normalization = if (map.hasKey("normalization")) {
                Normalization.fromMap(map.getMap("normalization"))
            } else {
                Normalization()
            }

            val dataLayout = if (map.hasKey("dataLayout")) {
                DataLayout.fromString(map.getString("dataLayout") ?: "HWC")
            } else {
                DataLayout.HWC
            }

            val outputFormat = if (map.hasKey("outputFormat")) {
                OutputFormat.fromString(map.getString("outputFormat") ?: "array")
            } else {
                OutputFormat.ARRAY
            }

            return GetPixelDataOptions(
                source = source,
                colorFormat = colorFormat,
                resize = resize,
                roi = roi,
                normalization = normalization,
                dataLayout = dataLayout,
                outputFormat = outputFormat
            )
        }
    }
}

/**
 * Batch processing options
 */
data class BatchOptions(
    val concurrency: Int = 4
) {
    companion object {
        fun fromMap(map: ReadableMap?): BatchOptions {
            if (map == null) return BatchOptions()
            val concurrency = if (map.hasKey("concurrency")) {
                map.getInt("concurrency")
            } else {
                4
            }
            return BatchOptions(concurrency)
        }
    }
}

/**
 * Result of pixel data extraction
 */
data class VisionUtilsResult(
    val data: FloatArray,
    val width: Int,
    val height: Int,
    val channels: Int,
    val colorFormat: ColorFormat,
    val dataLayout: DataLayout,
    val shape: IntArray,
    val processingTimeMs: Double
) {
    fun toWritableMap(): WritableMap {
        val map = Arguments.createMap()
        val dataArray = Arguments.createArray()
        data.forEach { dataArray.pushDouble(it.toDouble()) }
        map.putArray("data", dataArray)
        map.putInt("width", width)
        map.putInt("height", height)
        map.putInt("channels", channels)
        map.putString("colorFormat", colorFormat.name.lowercase())
        map.putString("dataLayout", dataLayout.name.lowercase())
        val shapeArray = Arguments.createArray()
        shape.forEach { shapeArray.pushInt(it) }
        map.putArray("shape", shapeArray)
        map.putDouble("processingTimeMs", processingTimeMs)
        return map
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as VisionUtilsResult
        return data.contentEquals(other.data) &&
               width == other.width &&
               height == other.height &&
               channels == other.channels &&
               colorFormat == other.colorFormat &&
               dataLayout == other.dataLayout &&
               shape.contentEquals(other.shape)
    }

    override fun hashCode(): Int {
        var result = data.contentHashCode()
        result = 31 * result + width
        result = 31 * result + height
        result = 31 * result + channels
        result = 31 * result + colorFormat.hashCode()
        result = 31 * result + dataLayout.hashCode()
        result = 31 * result + shape.contentHashCode()
        return result
    }
}

/**
 * Custom exception for vision utils errors
 */
class VisionUtilsException(
    val code: String,
    override val message: String
) : Exception(message)

// Extension functions

fun ReadableArray.toFloatArray(): FloatArray {
    val result = FloatArray(size())
    for (i in 0 until size()) {
        result[i] = getDouble(i).toFloat()
    }
    return result
}

fun ReadableArray.toIntArray(): IntArray {
    val result = IntArray(size())
    for (i in 0 until size()) {
        result[i] = getInt(i)
    }
    return result
}
