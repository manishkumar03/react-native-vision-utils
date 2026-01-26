import Foundation
import Accelerate

/// Handles quantization operations for ML inference
class Quantization {

    // MARK: - Quantize

    /// Quantize float data to integer format
    static func quantize(
        data: [Float],
        options: [String: Any]
    ) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        let dtype = (options["dtype"] as? String) ?? "int8"
        let mode = (options["mode"] as? String) ?? "per-tensor"
        let dataLayout = (options["dataLayout"] as? String) ?? "hwc"
        let channels = (options["channels"] as? Int) ?? 3

        var resultData: [Int]

        if mode == "per-channel" {
            guard let scaleArray = options["scale"] as? [Double],
                  let zeroPointArray = options["zeroPoint"] as? [Double] else {
                throw VisionUtilsError.processingError("Per-channel mode requires scale and zeroPoint arrays")
            }

            let width = options["width"] as? Int
            let height = options["height"] as? Int

            resultData = try quantizePerChannel(
                data: data,
                scale: scaleArray.map { Float($0) },
                zeroPoint: zeroPointArray.map { Float($0) },
                dtype: dtype,
                dataLayout: dataLayout,
                channels: channels,
                width: width,
                height: height
            )
        } else {
            // Per-tensor quantization
            let scale: Float
            let zeroPoint: Float

            if let scaleVal = options["scale"] as? Double {
                scale = Float(scaleVal)
            } else if let scaleArray = options["scale"] as? [Double], !scaleArray.isEmpty {
                scale = Float(scaleArray[0])
            } else {
                throw VisionUtilsError.processingError("Scale is required for quantization")
            }

            if let zpVal = options["zeroPoint"] as? Double {
                zeroPoint = Float(zpVal)
            } else if let zpArray = options["zeroPoint"] as? [Double], !zpArray.isEmpty {
                zeroPoint = Float(zpArray[0])
            } else {
                throw VisionUtilsError.processingError("Zero point is required for quantization")
            }

            resultData = quantizePerTensor(data: data, scale: scale, zeroPoint: zeroPoint, dtype: dtype)
        }

        let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        return [
            "data": resultData,
            "dtype": dtype,
            "mode": mode,
            "scale": options["scale"] as Any,
            "zeroPoint": options["zeroPoint"] as Any,
            "processingTimeMs": processingTimeMs
        ]
    }

    /// Per-tensor quantization: quantized = round(value / scale + zeroPoint)
    private static func quantizePerTensor(
        data: [Float],
        scale: Float,
        zeroPoint: Float,
        dtype: String
    ) -> [Int] {
        let (minVal, maxVal) = getRange(for: dtype)
        var result = [Int](repeating: 0, count: data.count)

        // Use Accelerate for performance
        var scaledData = [Float](repeating: 0, count: data.count)
        var scaleReciprocal = 1.0 / scale

        // Divide by scale: scaledData = data / scale
        vDSP_vsmul(data, 1, &scaleReciprocal, &scaledData, 1, vDSP_Length(data.count))

        // Add zero point
        var zp = zeroPoint
        vDSP_vsadd(scaledData, 1, &zp, &scaledData, 1, vDSP_Length(data.count))

        // Round and clamp
        for i in 0..<data.count {
            let rounded = Int(scaledData[i].rounded())
            result[i] = max(minVal, min(maxVal, rounded))
        }

        return result
    }

    /// Per-channel quantization
    private static func quantizePerChannel(
        data: [Float],
        scale: [Float],
        zeroPoint: [Float],
        dtype: String,
        dataLayout: String,
        channels: Int,
        width: Int?,
        height: Int?
    ) throws -> [Int] {
        guard scale.count == channels && zeroPoint.count == channels else {
            throw VisionUtilsError.processingError("Scale and zeroPoint arrays must match number of channels")
        }

        let (minVal, maxVal) = getRange(for: dtype)
        var result = [Int](repeating: 0, count: data.count)

        let pixelCount = data.count / channels

        if dataLayout.lowercased() == "hwc" || dataLayout.lowercased() == "nhwc" {
            // Interleaved format: RGBRGBRGB...
            for i in 0..<pixelCount {
                for c in 0..<channels {
                    let idx = i * channels + c
                    let quantized = (data[idx] / scale[c]) + zeroPoint[c]
                    result[idx] = max(minVal, min(maxVal, Int(quantized.rounded())))
                }
            }
        } else {
            // Planar format: RRR...GGG...BBB...
            for c in 0..<channels {
                let channelOffset = c * pixelCount
                for i in 0..<pixelCount {
                    let idx = channelOffset + i
                    let quantized = (data[idx] / scale[c]) + zeroPoint[c]
                    result[idx] = max(minVal, min(maxVal, Int(quantized.rounded())))
                }
            }
        }

        return result
    }

    // MARK: - Dequantize

    /// Dequantize integer data back to float: value = (quantized - zeroPoint) * scale
    static func dequantize(
        data: [Int],
        options: [String: Any]
    ) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        let mode = (options["mode"] as? String) ?? "per-tensor"
        let dataLayout = (options["dataLayout"] as? String) ?? "hwc"
        let channels = (options["channels"] as? Int) ?? 3

        var resultData: [Float]

        if mode == "per-channel" {
            guard let scaleArray = options["scale"] as? [Double],
                  let zeroPointArray = options["zeroPoint"] as? [Double] else {
                throw VisionUtilsError.processingError("Per-channel mode requires scale and zeroPoint arrays")
            }

            resultData = dequantizePerChannel(
                data: data,
                scale: scaleArray.map { Float($0) },
                zeroPoint: zeroPointArray.map { Float($0) },
                dataLayout: dataLayout,
                channels: channels
            )
        } else {
            let scale: Float
            let zeroPoint: Float

            if let scaleVal = options["scale"] as? Double {
                scale = Float(scaleVal)
            } else if let scaleArray = options["scale"] as? [Double], !scaleArray.isEmpty {
                scale = Float(scaleArray[0])
            } else {
                throw VisionUtilsError.processingError("Scale is required for dequantization")
            }

            if let zpVal = options["zeroPoint"] as? Double {
                zeroPoint = Float(zpVal)
            } else if let zpArray = options["zeroPoint"] as? [Double], !zpArray.isEmpty {
                zeroPoint = Float(zpArray[0])
            } else {
                throw VisionUtilsError.processingError("Zero point is required for dequantization")
            }

            resultData = dequantizePerTensor(data: data, scale: scale, zeroPoint: zeroPoint)
        }

        let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        return [
            "data": resultData,
            "processingTimeMs": processingTimeMs
        ]
    }

    /// Per-tensor dequantization
    private static func dequantizePerTensor(
        data: [Int],
        scale: Float,
        zeroPoint: Float
    ) -> [Float] {
        var result = [Float](repeating: 0, count: data.count)

        for i in 0..<data.count {
            result[i] = (Float(data[i]) - zeroPoint) * scale
        }

        return result
    }

    /// Per-channel dequantization
    private static func dequantizePerChannel(
        data: [Int],
        scale: [Float],
        zeroPoint: [Float],
        dataLayout: String,
        channels: Int
    ) -> [Float] {
        var result = [Float](repeating: 0, count: data.count)
        let pixelCount = data.count / channels

        if dataLayout.lowercased() == "hwc" || dataLayout.lowercased() == "nhwc" {
            for i in 0..<pixelCount {
                for c in 0..<channels {
                    let idx = i * channels + c
                    result[idx] = (Float(data[idx]) - zeroPoint[c]) * scale[c]
                }
            }
        } else {
            for c in 0..<channels {
                let channelOffset = c * pixelCount
                for i in 0..<pixelCount {
                    let idx = channelOffset + i
                    result[idx] = (Float(data[idx]) - zeroPoint[c]) * scale[c]
                }
            }
        }

        return result
    }

    // MARK: - Calculate Quantization Parameters

    /// Calculate optimal quantization parameters from data
    static func calculateQuantizationParams(
        data: [Float],
        options: [String: Any]
    ) throws -> [String: Any] {
        let dtype = (options["dtype"] as? String) ?? "int8"
        let mode = (options["mode"] as? String) ?? "per-tensor"
        let symmetric = (options["symmetric"] as? Bool) ?? false
        let dataLayout = (options["dataLayout"] as? String) ?? "hwc"
        let channels = (options["channels"] as? Int) ?? 3

        let (qMin, qMax) = getRange(for: dtype)
        let qMinF = Float(qMin)
        let qMaxF = Float(qMax)

        if mode == "per-channel" {
            return try calculatePerChannelParams(
                data: data,
                dtype: dtype,
                symmetric: symmetric,
                dataLayout: dataLayout,
                channels: channels,
                qMin: qMinF,
                qMax: qMaxF
            )
        } else {
            return calculatePerTensorParams(
                data: data,
                symmetric: symmetric,
                qMin: qMinF,
                qMax: qMaxF
            )
        }
    }

    /// Calculate per-tensor quantization parameters
    private static func calculatePerTensorParams(
        data: [Float],
        symmetric: Bool,
        qMin: Float,
        qMax: Float
    ) -> [String: Any] {
        var minVal: Float = 0
        var maxVal: Float = 0
        vDSP_minv(data, 1, &minVal, vDSP_Length(data.count))
        vDSP_maxv(data, 1, &maxVal, vDSP_Length(data.count))

        let scale: Float
        let zeroPoint: Float

        if symmetric {
            // Symmetric quantization: zeroPoint = 0, scale based on max(|min|, |max|)
            let absMax = max(abs(minVal), abs(maxVal))
            scale = absMax / max(abs(qMin), abs(qMax))
            zeroPoint = 0
        } else {
            // Asymmetric quantization
            scale = (maxVal - minVal) / (qMax - qMin)
            zeroPoint = qMin - minVal / scale
        }

        return [
            "scale": scale,
            "zeroPoint": zeroPoint,
            "min": minVal,
            "max": maxVal
        ]
    }

    /// Calculate per-channel quantization parameters
    private static func calculatePerChannelParams(
        data: [Float],
        dtype: String,
        symmetric: Bool,
        dataLayout: String,
        channels: Int,
        qMin: Float,
        qMax: Float
    ) throws -> [String: Any] {
        let pixelCount = data.count / channels

        var scales = [Float](repeating: 0, count: channels)
        var zeroPoints = [Float](repeating: 0, count: channels)
        var mins = [Float](repeating: 0, count: channels)
        var maxs = [Float](repeating: 0, count: channels)

        for c in 0..<channels {
            // Extract channel data
            var channelData = [Float](repeating: 0, count: pixelCount)

            if dataLayout.lowercased() == "hwc" || dataLayout.lowercased() == "nhwc" {
                for i in 0..<pixelCount {
                    channelData[i] = data[i * channels + c]
                }
            } else {
                let channelOffset = c * pixelCount
                for i in 0..<pixelCount {
                    channelData[i] = data[channelOffset + i]
                }
            }

            // Find min/max for this channel
            var minVal: Float = 0
            var maxVal: Float = 0
            vDSP_minv(channelData, 1, &minVal, vDSP_Length(pixelCount))
            vDSP_maxv(channelData, 1, &maxVal, vDSP_Length(pixelCount))

            mins[c] = minVal
            maxs[c] = maxVal

            if symmetric {
                let absMax = max(abs(minVal), abs(maxVal))
                scales[c] = absMax / max(abs(qMin), abs(qMax))
                zeroPoints[c] = 0
            } else {
                scales[c] = (maxVal - minVal) / (qMax - qMin)
                zeroPoints[c] = qMin - minVal / scales[c]
            }

            // Handle edge case where min == max
            if scales[c] == 0 || scales[c].isNaN || scales[c].isInfinite {
                scales[c] = 1.0
                zeroPoints[c] = 0
            }
        }

        return [
            "scale": scales,
            "zeroPoint": zeroPoints,
            "min": mins,
            "max": maxs
        ]
    }

    // MARK: - Helpers

    /// Get the valid range for a dtype
    private static func getRange(for dtype: String) -> (Int, Int) {
        switch dtype {
        case "int8":
            return (-128, 127)
        case "uint8":
            return (0, 255)
        case "int16":
            return (-32768, 32767)
        default:
            return (-128, 127)
        }
    }
}
