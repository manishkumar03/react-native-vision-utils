import Foundation
import UIKit
import Accelerate

/// Handles blur detection using Laplacian variance method
class BlurDetector {

    /// Result of blur detection
    struct BlurResult {
        let isBlurry: Bool
        let score: Double          // Laplacian variance (higher = sharper)
        let threshold: Double      // Threshold used for classification
        let processingTimeMs: Double

        func toDictionary() -> [String: Any] {
            return [
                "isBlurry": isBlurry,
                "score": score,
                "threshold": threshold,
                "processingTimeMs": processingTimeMs
            ]
        }
    }

    /// Detect if an image is blurry using Laplacian variance
    /// - Parameters:
    ///   - image: The input UIImage to analyze
    ///   - threshold: Variance threshold below which image is considered blurry (default: 100)
    ///   - downsampleSize: Optional size to downsample to for faster processing
    /// - Returns: BlurResult containing blur status and score
    static func detectBlur(
        image: UIImage,
        threshold: Double = 100.0,
        downsampleSize: Int? = nil
    ) throws -> BlurResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        guard let cgImage = image.cgImage else {
            throw VisionUtilsError.processingError("Failed to get CGImage from UIImage")
        }

        // Optionally downsample for faster processing
        let processImage: CGImage
        let processWidth: Int
        let processHeight: Int

        if let maxSize = downsampleSize, cgImage.width > maxSize || cgImage.height > maxSize {
            let scale = Double(maxSize) / Double(max(cgImage.width, cgImage.height))
            processWidth = Int(Double(cgImage.width) * scale)
            processHeight = Int(Double(cgImage.height) * scale)
            processImage = try downsample(cgImage, toWidth: processWidth, height: processHeight)
        } else {
            processImage = cgImage
            processWidth = cgImage.width
            processHeight = cgImage.height
        }

        // Convert to grayscale
        let grayscaleData = try convertToGrayscale(processImage, width: processWidth, height: processHeight)

        // Apply Laplacian kernel and calculate variance
        let laplacianVariance = try calculateLaplacianVariance(
            grayscaleData,
            width: processWidth,
            height: processHeight
        )

        let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        return BlurResult(
            isBlurry: laplacianVariance < threshold,
            score: laplacianVariance,
            threshold: threshold,
            processingTimeMs: processingTimeMs
        )
    }

    // MARK: - Private Helpers

    /// Downsample image for faster processing
    private static func downsample(_ image: CGImage, toWidth width: Int, height: Int) throws -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw VisionUtilsError.processingError("Failed to create downsampling context")
        }

        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let result = context.makeImage() else {
            throw VisionUtilsError.processingError("Failed to create downsampled image")
        }

        return result
    }

    /// Convert image to grayscale float array
    private static func convertToGrayscale(_ image: CGImage, width: Int, height: Int) throws -> [Float] {
        let pixelCount = width * height

        // Extract RGBA pixels
        var rgbaData = [UInt8](repeating: 0, count: pixelCount * 4)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &rgbaData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw VisionUtilsError.processingError("Failed to create pixel extraction context")
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert to grayscale using luminance formula: 0.299*R + 0.587*G + 0.114*B
        var grayscale = [Float](repeating: 0, count: pixelCount)

        for i in 0..<pixelCount {
            let r = Float(rgbaData[i * 4])
            let g = Float(rgbaData[i * 4 + 1])
            let b = Float(rgbaData[i * 4 + 2])
            grayscale[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        return grayscale
    }

    /// Apply Laplacian kernel and calculate variance
    /// Laplacian kernel:
    /// [0,  1, 0]
    /// [1, -4, 1]
    /// [0,  1, 0]
    private static func calculateLaplacianVariance(_ grayscale: [Float], width: Int, height: Int) throws -> Double {
        let pixelCount = width * height

        // Apply Laplacian convolution
        var laplacian = [Float](repeating: 0, count: pixelCount)

        // Laplacian kernel values
        // Using the standard 3x3 Laplacian kernel
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let idx = y * width + x

                // Get neighbors
                let top = grayscale[(y - 1) * width + x]
                let bottom = grayscale[(y + 1) * width + x]
                let left = grayscale[y * width + (x - 1)]
                let right = grayscale[y * width + (x + 1)]
                let center = grayscale[idx]

                // Laplacian = top + bottom + left + right - 4*center
                laplacian[idx] = top + bottom + left + right - 4.0 * center
            }
        }

        // Calculate variance of Laplacian response
        // Variance = E[X^2] - E[X]^2

        // Calculate mean
        var mean: Float = 0
        vDSP_meanv(laplacian, 1, &mean, vDSP_Length(pixelCount))

        // Calculate squared differences from mean
        var squaredDiffs = [Float](repeating: 0, count: pixelCount)
        var negMean = -mean
        vDSP_vsadd(laplacian, 1, &negMean, &squaredDiffs, 1, vDSP_Length(pixelCount))
        vDSP_vsq(squaredDiffs, 1, &squaredDiffs, 1, vDSP_Length(pixelCount))

        // Calculate variance (mean of squared differences)
        var variance: Float = 0
        vDSP_meanv(squaredDiffs, 1, &variance, vDSP_Length(pixelCount))

        return Double(variance)
    }
}
