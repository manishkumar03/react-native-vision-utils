import Foundation
import UIKit
import CoreGraphics

/// Handles tensor to image conversion
class TensorConverter {

    /// Convert tensor data back to an image
    static func tensorToImage(
        data: [Float],
        width: Int,
        height: Int,
        options: [String: Any]
    ) throws -> [String: Any] {
        let channels = (options["channels"] as? Int) ?? 3
        let dataLayout = (options["dataLayout"] as? String)?.lowercased() ?? "hwc"
        let denormalize = (options["denormalize"] as? Bool) ?? true
        let outputFormat = (options["outputFormat"] as? String) ?? "base64"

        let expectedSize = width * height * channels
        guard data.count >= expectedSize else {
            throw VisionUtilsError.processingError("Data size \(data.count) is less than expected \(expectedSize)")
        }

        // Convert to HWC format if necessary
        var hwcData: [Float]
        if dataLayout == "chw" || dataLayout == "nchw" {
            hwcData = convertCHWtoHWC(data: data, width: width, height: height, channels: channels)
        } else {
            hwcData = Array(data.prefix(expectedSize))
        }

        // Denormalize if needed
        if denormalize {
            // Check for normalization preset
            if let normalization = options["normalization"] as? [String: Any],
               let preset = normalization["preset"] as? String {
                hwcData = denormalizeData(hwcData, preset: preset, channels: channels, normalization: normalization)
            } else {
                // Assume scale normalization [0, 1] -> [0, 255]
                hwcData = hwcData.map { min(255, max(0, $0 * 255)) }
            }
        }

        // Convert to RGBA
        var rgbaData = [UInt8](repeating: 255, count: width * height * 4)
        let pixelCount = width * height

        for i in 0..<pixelCount {
            if channels == 1 {
                // Grayscale
                let gray = UInt8(clamping: Int(hwcData[i].rounded()))
                rgbaData[i * 4] = gray
                rgbaData[i * 4 + 1] = gray
                rgbaData[i * 4 + 2] = gray
            } else if channels >= 3 {
                rgbaData[i * 4] = UInt8(clamping: Int(hwcData[i * channels].rounded()))
                rgbaData[i * 4 + 1] = UInt8(clamping: Int(hwcData[i * channels + 1].rounded()))
                rgbaData[i * 4 + 2] = UInt8(clamping: Int(hwcData[i * channels + 2].rounded()))
                if channels == 4 {
                    rgbaData[i * 4 + 3] = UInt8(clamping: Int(hwcData[i * channels + 3].rounded()))
                }
            }
        }

        // Create UIImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let context = CGContext(
            data: &rgbaData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ),
        let cgImage = context.makeImage() else {
            throw VisionUtilsError.processingError("Failed to create image from tensor data")
        }

        let image = UIImage(cgImage: cgImage)

        // Convert to output format
        if outputFormat == "base64" {
            guard let pngData = image.pngData() else {
                throw VisionUtilsError.processingError("Failed to encode image as PNG")
            }
            let base64String = pngData.base64EncodedString()
            return [
                "base64": "data:image/png;base64,\(base64String)",
                "width": width,
                "height": height
            ]
        } else {
            throw VisionUtilsError.processingError("Unsupported output format: \(outputFormat)")
        }
    }

    private static func convertCHWtoHWC(data: [Float], width: Int, height: Int, channels: Int) -> [Float] {
        let pixelCount = width * height
        var result = [Float](repeating: 0, count: pixelCount * channels)

        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    let chwIdx = c * height * width + h * width + w
                    let hwcIdx = h * width * channels + w * channels + c
                    result[hwcIdx] = data[chwIdx]
                }
            }
        }

        return result
    }

    private static func denormalizeData(_ data: [Float], preset: String, channels: Int, normalization: [String: Any]) -> [Float] {
        var result = data
        let pixelCount = data.count / channels

        switch preset {
        case "imagenet":
            let mean: [Float] = [0.485, 0.456, 0.406]
            let std: [Float] = [0.229, 0.224, 0.225]

            for i in 0..<pixelCount {
                for c in 0..<min(channels, 3) {
                    let idx = i * channels + c
                    result[idx] = (result[idx] * std[c] + mean[c]) * 255
                }
            }

        case "tensorflow":
            // [-1, 1] -> [0, 255]
            result = result.map { ($0 + 1) * 127.5 }

        case "scale":
            // [0, 1] -> [0, 255]
            result = result.map { $0 * 255 }

        case "custom":
            if let mean = normalization["mean"] as? [Double],
               let std = normalization["std"] as? [Double] {
                let scale = (normalization["scale"] as? Double) ?? (1.0 / 255.0)

                for i in 0..<pixelCount {
                    for c in 0..<min(channels, mean.count) {
                        let idx = i * channels + c
                        result[idx] = (result[idx] * Float(std[c]) + Float(mean[c])) / Float(scale)
                    }
                }
            }

        default:
            // Raw - assume already in [0, 255]
            break
        }

        return result
    }
}
