import Foundation
import UIKit
import CoreGraphics
import Accelerate

/// Handles image statistics and metadata extraction
class ImageAnalyzer {

    // MARK: - Statistics

    /// Calculate image statistics (mean, std, min, max, histogram)
    static func getStatistics(from image: UIImage) throws -> [String: Any] {
        guard let cgImage = image.cgImage else {
            throw VisionUtilsError.processingError("Failed to get CGImage")
        }

        let width = cgImage.width
        let height = cgImage.height
        let pixelCount = width * height

        // Extract RGBA pixels
        let rgbaData = try extractRGBAPixels(from: cgImage, width: width, height: height)

        // Calculate per-channel statistics
        var channelStats: [[String: Any]] = []

        for channel in 0..<3 {
            var values = [Float](repeating: 0, count: pixelCount)

            for i in 0..<pixelCount {
                values[i] = Float(rgbaData[i * 4 + channel])
            }

            let stats = calculateStats(values)
            let histogram = calculateHistogram(values)

            var channelDict: [String: Any] = stats
            channelDict["histogram"] = histogram
            channelStats.append(channelDict)
        }

        // Calculate overall statistics
        var allValues = [Float](repeating: 0, count: pixelCount * 3)
        for i in 0..<pixelCount {
            allValues[i * 3] = Float(rgbaData[i * 4])
            allValues[i * 3 + 1] = Float(rgbaData[i * 4 + 1])
            allValues[i * 3 + 2] = Float(rgbaData[i * 4 + 2])
        }

        let overallStats = calculateStats(allValues)

        return [
            "mean": overallStats["mean"] ?? 0,
            "std": overallStats["std"] ?? 0,
            "min": overallStats["min"] ?? 0,
            "max": overallStats["max"] ?? 0,
            "perChannel": [
                "r": channelStats[0],
                "g": channelStats[1],
                "b": channelStats[2]
            ]
        ]
    }

    private static func calculateStats(_ values: [Float]) -> [String: Any] {
        guard !values.isEmpty else {
            return ["mean": 0, "std": 0, "min": 0, "max": 0]
        }

        var mean: Float = 0
        var minVal: Float = 0
        var maxVal: Float = 0

        vDSP_meanv(values, 1, &mean, vDSP_Length(values.count))
        vDSP_minv(values, 1, &minVal, vDSP_Length(values.count))
        vDSP_maxv(values, 1, &maxVal, vDSP_Length(values.count))

        // Calculate standard deviation
        var squaredDiffs = [Float](repeating: 0, count: values.count)
        var negMean = -mean
        vDSP_vsadd(values, 1, &negMean, &squaredDiffs, 1, vDSP_Length(values.count))
        vDSP_vsq(squaredDiffs, 1, &squaredDiffs, 1, vDSP_Length(values.count))

        var variance: Float = 0
        vDSP_meanv(squaredDiffs, 1, &variance, vDSP_Length(values.count))
        let std = sqrt(variance)

        return [
            "mean": mean,
            "std": std,
            "min": minVal,
            "max": maxVal
        ]
    }

    private static func calculateHistogram(_ values: [Float]) -> [Int] {
        var histogram = [Int](repeating: 0, count: 256)

        for value in values {
            let bin = min(255, max(0, Int(value)))
            histogram[bin] += 1
        }

        return histogram
    }

    // MARK: - Metadata

    /// Get image metadata (dimensions, format, color space)
    static func getMetadata(from image: UIImage, fileSize: Int? = nil, format: String? = nil) -> [String: Any] {
        let width = Int(image.size.width * image.scale)
        let height = Int(image.size.height * image.scale)

        var colorSpace = "unknown"
        var hasAlpha = false
        var bitsPerComponent = 8
        var bitsPerPixel = 32

        if let cgImage = image.cgImage {
            if let cs = cgImage.colorSpace {
                colorSpace = cs.name as String? ?? "unknown"
            }
            hasAlpha = cgImage.alphaInfo != .none && cgImage.alphaInfo != .noneSkipLast && cgImage.alphaInfo != .noneSkipFirst
            bitsPerComponent = cgImage.bitsPerComponent
            bitsPerPixel = cgImage.bitsPerPixel
        }

        var result: [String: Any] = [
            "width": width,
            "height": height,
            "format": format ?? "unknown",
            "colorSpace": colorSpace,
            "hasAlpha": hasAlpha,
            "bitsPerComponent": bitsPerComponent,
            "bitsPerPixel": bitsPerPixel,
            "orientation": image.imageOrientation.rawValue,
            "scale": image.scale
        ]

        if let size = fileSize {
            result["fileSize"] = size
        }

        return result
    }

    // MARK: - Validation

    /// Validate image against criteria
    static func validate(image: UIImage, options: [String: Any]) -> [String: Any] {
        let width = Int(image.size.width * image.scale)
        let height = Int(image.size.height * image.scale)

        var isValid = true
        var errors: [String] = []

        // Check minimum dimensions
        if let minWidth = options["minWidth"] as? Int, width < minWidth {
            isValid = false
            errors.append("Width \(width) is less than minimum \(minWidth)")
        }

        if let minHeight = options["minHeight"] as? Int, height < minHeight {
            isValid = false
            errors.append("Height \(height) is less than minimum \(minHeight)")
        }

        // Check maximum dimensions
        if let maxWidth = options["maxWidth"] as? Int, width > maxWidth {
            isValid = false
            errors.append("Width \(width) exceeds maximum \(maxWidth)")
        }

        if let maxHeight = options["maxHeight"] as? Int, height > maxHeight {
            isValid = false
            errors.append("Height \(height) exceeds maximum \(maxHeight)")
        }

        // Check aspect ratio
        if let aspectRatio = options["aspectRatio"] as? Double {
            let tolerance = (options["aspectRatioTolerance"] as? Double) ?? 0.1
            let actualRatio = Double(width) / Double(height)

            if abs(actualRatio - aspectRatio) > tolerance {
                isValid = false
                errors.append("Aspect ratio \(actualRatio) differs from expected \(aspectRatio) by more than \(tolerance)")
            }
        }

        // Check minimum file size (approximate based on pixel count)
        if let minPixels = options["minPixels"] as? Int {
            let pixelCount = width * height
            if pixelCount < minPixels {
                isValid = false
                errors.append("Pixel count \(pixelCount) is less than minimum \(minPixels)")
            }
        }

        return [
            "isValid": isValid,
            "width": width,
            "height": height,
            "errors": errors
        ]
    }

    // MARK: - Helper

    private static func extractRGBAPixels(from image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow

        var pixelData = [UInt8](repeating: 0, count: totalBytes)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw VisionUtilsError.processingError("Failed to create pixel extraction context")
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return pixelData
    }
}
