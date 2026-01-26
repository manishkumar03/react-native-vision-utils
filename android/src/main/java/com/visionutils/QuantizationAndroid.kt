package com.visionutils

import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Handles quantization operations for ML inference
 */
object QuantizationAndroid {

    /**
     * Quantize float data to integer format
     */
    fun quantize(data: FloatArray, options: ReadableMap): WritableMap {
        val startTime = System.nanoTime()

        val dtype = if (options.hasKey("dtype")) options.getString("dtype") ?: "int8" else "int8"
        val mode = if (options.hasKey("mode")) options.getString("mode") ?: "per-tensor" else "per-tensor"
        val dataLayout = if (options.hasKey("dataLayout")) options.getString("dataLayout") ?: "hwc" else "hwc"
        val channels = if (options.hasKey("channels")) options.getInt("channels") else 3

        val resultData: IntArray

        if (mode == "per-channel") {
            val scaleArray = options.getArray("scale")
                ?: throw VisionUtilsException("INVALID_OPTIONS", "Per-channel mode requires scale array")
            val zeroPointArray = options.getArray("zeroPoint")
                ?: throw VisionUtilsException("INVALID_OPTIONS", "Per-channel mode requires zeroPoint array")

            val scale = FloatArray(scaleArray.size()) { scaleArray.getDouble(it).toFloat() }
            val zeroPoint = FloatArray(zeroPointArray.size()) { zeroPointArray.getDouble(it).toFloat() }

            val width = if (options.hasKey("width")) options.getInt("width") else null
            val height = if (options.hasKey("height")) options.getInt("height") else null

            resultData = quantizePerChannel(data, scale, zeroPoint, dtype, dataLayout, channels)
        } else {
            // Per-tensor quantization
            val scale: Float
            val zeroPoint: Float

            if (options.hasKey("scale")) {
                val scaleVal = options.getDynamic("scale")
                scale = if (scaleVal.type.name == "Array") {
                    options.getArray("scale")!!.getDouble(0).toFloat()
                } else {
                    options.getDouble("scale").toFloat()
                }
            } else {
                throw VisionUtilsException("INVALID_OPTIONS", "Scale is required for quantization")
            }

            if (options.hasKey("zeroPoint")) {
                val zpVal = options.getDynamic("zeroPoint")
                zeroPoint = if (zpVal.type.name == "Array") {
                    options.getArray("zeroPoint")!!.getDouble(0).toFloat()
                } else {
                    options.getDouble("zeroPoint").toFloat()
                }
            } else {
                throw VisionUtilsException("INVALID_OPTIONS", "Zero point is required for quantization")
            }

            resultData = quantizePerTensor(data, scale, zeroPoint, dtype)
        }

        val processingTimeMs = (System.nanoTime() - startTime) / 1_000_000.0

        val result = Arguments.createMap()
        result.putArray("data", Arguments.fromArray(resultData.toTypedArray()))
        result.putString("dtype", dtype)
        result.putString("mode", mode)

        // Copy scale and zeroPoint back
        if (options.hasKey("scale")) {
            val scaleVal = options.getDynamic("scale")
            if (scaleVal.type.name == "Array") {
                result.putArray("scale", options.getArray("scale"))
            } else {
                result.putDouble("scale", options.getDouble("scale"))
            }
        }
        if (options.hasKey("zeroPoint")) {
            val zpVal = options.getDynamic("zeroPoint")
            if (zpVal.type.name == "Array") {
                result.putArray("zeroPoint", options.getArray("zeroPoint"))
            } else {
                result.putDouble("zeroPoint", options.getDouble("zeroPoint"))
            }
        }

        result.putDouble("processingTimeMs", processingTimeMs)
        return result
    }

    /**
     * Per-tensor quantization: quantized = round(value / scale + zeroPoint)
     */
    private fun quantizePerTensor(
        data: FloatArray,
        scale: Float,
        zeroPoint: Float,
        dtype: String
    ): IntArray {
        val (minVal, maxVal) = getRange(dtype)
        val result = IntArray(data.size)
        val scaleReciprocal = 1.0f / scale

        for (i in data.indices) {
            val quantized = (data[i] * scaleReciprocal + zeroPoint).roundToInt()
            result[i] = max(minVal, min(maxVal, quantized))
        }

        return result
    }

    /**
     * Per-channel quantization
     */
    private fun quantizePerChannel(
        data: FloatArray,
        scale: FloatArray,
        zeroPoint: FloatArray,
        dtype: String,
        dataLayout: String,
        channels: Int
    ): IntArray {
        require(scale.size == channels && zeroPoint.size == channels) {
            "Scale and zeroPoint arrays must match number of channels"
        }

        val (minVal, maxVal) = getRange(dtype)
        val result = IntArray(data.size)
        val pixelCount = data.size / channels

        if (dataLayout.lowercase() == "hwc" || dataLayout.lowercase() == "nhwc") {
            // Interleaved format: RGBRGBRGB...
            for (i in 0 until pixelCount) {
                for (c in 0 until channels) {
                    val idx = i * channels + c
                    val quantized = (data[idx] / scale[c] + zeroPoint[c]).roundToInt()
                    result[idx] = max(minVal, min(maxVal, quantized))
                }
            }
        } else {
            // Planar format: RRR...GGG...BBB...
            for (c in 0 until channels) {
                val channelOffset = c * pixelCount
                for (i in 0 until pixelCount) {
                    val idx = channelOffset + i
                    val quantized = (data[idx] / scale[c] + zeroPoint[c]).roundToInt()
                    result[idx] = max(minVal, min(maxVal, quantized))
                }
            }
        }

        return result
    }

    /**
     * Dequantize integer data back to float: value = (quantized - zeroPoint) * scale
     */
    fun dequantize(data: IntArray, options: ReadableMap): WritableMap {
        val startTime = System.nanoTime()

        val mode = if (options.hasKey("mode")) options.getString("mode") ?: "per-tensor" else "per-tensor"
        val dataLayout = if (options.hasKey("dataLayout")) options.getString("dataLayout") ?: "hwc" else "hwc"
        val channels = if (options.hasKey("channels")) options.getInt("channels") else 3

        val resultData: FloatArray

        if (mode == "per-channel") {
            val scaleArray = options.getArray("scale")
                ?: throw VisionUtilsException("INVALID_OPTIONS", "Per-channel mode requires scale array")
            val zeroPointArray = options.getArray("zeroPoint")
                ?: throw VisionUtilsException("INVALID_OPTIONS", "Per-channel mode requires zeroPoint array")

            val scale = FloatArray(scaleArray.size()) { scaleArray.getDouble(it).toFloat() }
            val zeroPoint = FloatArray(zeroPointArray.size()) { zeroPointArray.getDouble(it).toFloat() }

            resultData = dequantizePerChannel(data, scale, zeroPoint, dataLayout, channels)
        } else {
            val scale: Float
            val zeroPoint: Float

            if (options.hasKey("scale")) {
                val scaleVal = options.getDynamic("scale")
                scale = if (scaleVal.type.name == "Array") {
                    options.getArray("scale")!!.getDouble(0).toFloat()
                } else {
                    options.getDouble("scale").toFloat()
                }
            } else {
                throw VisionUtilsException("INVALID_OPTIONS", "Scale is required for dequantization")
            }

            if (options.hasKey("zeroPoint")) {
                val zpVal = options.getDynamic("zeroPoint")
                zeroPoint = if (zpVal.type.name == "Array") {
                    options.getArray("zeroPoint")!!.getDouble(0).toFloat()
                } else {
                    options.getDouble("zeroPoint").toFloat()
                }
            } else {
                throw VisionUtilsException("INVALID_OPTIONS", "Zero point is required for dequantization")
            }

            resultData = dequantizePerTensor(data, scale, zeroPoint)
        }

        val processingTimeMs = (System.nanoTime() - startTime) / 1_000_000.0

        val result = Arguments.createMap()
        result.putArray("data", Arguments.fromArray(resultData.toTypedArray()))
        result.putDouble("processingTimeMs", processingTimeMs)
        return result
    }

    /**
     * Per-tensor dequantization
     */
    private fun dequantizePerTensor(
        data: IntArray,
        scale: Float,
        zeroPoint: Float
    ): FloatArray {
        val result = FloatArray(data.size)
        for (i in data.indices) {
            result[i] = (data[i].toFloat() - zeroPoint) * scale
        }
        return result
    }

    /**
     * Per-channel dequantization
     */
    private fun dequantizePerChannel(
        data: IntArray,
        scale: FloatArray,
        zeroPoint: FloatArray,
        dataLayout: String,
        channels: Int
    ): FloatArray {
        val result = FloatArray(data.size)
        val pixelCount = data.size / channels

        if (dataLayout.lowercase() == "hwc" || dataLayout.lowercase() == "nhwc") {
            for (i in 0 until pixelCount) {
                for (c in 0 until channels) {
                    val idx = i * channels + c
                    result[idx] = (data[idx].toFloat() - zeroPoint[c]) * scale[c]
                }
            }
        } else {
            for (c in 0 until channels) {
                val channelOffset = c * pixelCount
                for (i in 0 until pixelCount) {
                    val idx = channelOffset + i
                    result[idx] = (data[idx].toFloat() - zeroPoint[c]) * scale[c]
                }
            }
        }

        return result
    }

    /**
     * Calculate optimal quantization parameters from data
     */
    fun calculateQuantizationParams(data: FloatArray, options: ReadableMap): WritableMap {
        val dtype = if (options.hasKey("dtype")) options.getString("dtype") ?: "int8" else "int8"
        val mode = if (options.hasKey("mode")) options.getString("mode") ?: "per-tensor" else "per-tensor"
        val symmetric = if (options.hasKey("symmetric")) options.getBoolean("symmetric") else false
        val dataLayout = if (options.hasKey("dataLayout")) options.getString("dataLayout") ?: "hwc" else "hwc"
        val channels = if (options.hasKey("channels")) options.getInt("channels") else 3

        val (qMin, qMax) = getRange(dtype)
        val qMinF = qMin.toFloat()
        val qMaxF = qMax.toFloat()

        return if (mode == "per-channel") {
            calculatePerChannelParams(data, symmetric, dataLayout, channels, qMinF, qMaxF)
        } else {
            calculatePerTensorParams(data, symmetric, qMinF, qMaxF)
        }
    }

    /**
     * Calculate per-tensor quantization parameters
     */
    private fun calculatePerTensorParams(
        data: FloatArray,
        symmetric: Boolean,
        qMin: Float,
        qMax: Float
    ): WritableMap {
        var minVal = Float.MAX_VALUE
        var maxVal = Float.MIN_VALUE

        for (value in data) {
            if (value < minVal) minVal = value
            if (value > maxVal) maxVal = value
        }

        val scale: Float
        val zeroPoint: Float

        if (symmetric) {
            val absMax = max(kotlin.math.abs(minVal), kotlin.math.abs(maxVal))
            scale = absMax / max(kotlin.math.abs(qMin), kotlin.math.abs(qMax))
            zeroPoint = 0f
        } else {
            scale = (maxVal - minVal) / (qMax - qMin)
            zeroPoint = qMin - minVal / scale
        }

        val result = Arguments.createMap()
        result.putDouble("scale", scale.toDouble())
        result.putDouble("zeroPoint", zeroPoint.toDouble())
        result.putDouble("min", minVal.toDouble())
        result.putDouble("max", maxVal.toDouble())
        return result
    }

    /**
     * Calculate per-channel quantization parameters
     */
    private fun calculatePerChannelParams(
        data: FloatArray,
        symmetric: Boolean,
        dataLayout: String,
        channels: Int,
        qMin: Float,
        qMax: Float
    ): WritableMap {
        val pixelCount = data.size / channels

        val scales = FloatArray(channels)
        val zeroPoints = FloatArray(channels)
        val mins = FloatArray(channels)
        val maxs = FloatArray(channels)

        for (c in 0 until channels) {
            var minVal = Float.MAX_VALUE
            var maxVal = Float.MIN_VALUE

            if (dataLayout.lowercase() == "hwc" || dataLayout.lowercase() == "nhwc") {
                for (i in 0 until pixelCount) {
                    val value = data[i * channels + c]
                    if (value < minVal) minVal = value
                    if (value > maxVal) maxVal = value
                }
            } else {
                val channelOffset = c * pixelCount
                for (i in 0 until pixelCount) {
                    val value = data[channelOffset + i]
                    if (value < minVal) minVal = value
                    if (value > maxVal) maxVal = value
                }
            }

            mins[c] = minVal
            maxs[c] = maxVal

            if (symmetric) {
                val absMax = max(kotlin.math.abs(minVal), kotlin.math.abs(maxVal))
                scales[c] = absMax / max(kotlin.math.abs(qMin), kotlin.math.abs(qMax))
                zeroPoints[c] = 0f
            } else {
                scales[c] = (maxVal - minVal) / (qMax - qMin)
                zeroPoints[c] = qMin - minVal / scales[c]
            }

            // Handle edge case where min == max
            if (scales[c] == 0f || scales[c].isNaN() || scales[c].isInfinite()) {
                scales[c] = 1.0f
                zeroPoints[c] = 0f
            }
        }

        val result = Arguments.createMap()
        result.putArray("scale", Arguments.fromArray(scales.toTypedArray()))
        result.putArray("zeroPoint", Arguments.fromArray(zeroPoints.toTypedArray()))
        result.putArray("min", Arguments.fromArray(mins.toTypedArray()))
        result.putArray("max", Arguments.fromArray(maxs.toTypedArray()))
        return result
    }

    /**
     * Get the valid range for a dtype
     */
    private fun getRange(dtype: String): Pair<Int, Int> {
        return when (dtype) {
            "int8" -> Pair(-128, 127)
            "uint8" -> Pair(0, 255)
            "int16" -> Pair(-32768, 32767)
            else -> Pair(-128, 127)
        }
    }
}
