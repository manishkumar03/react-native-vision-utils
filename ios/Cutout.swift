import Foundation
import UIKit
import CoreGraphics

/**
 * Cutout provides random erasing augmentation for data augmentation pipelines.
 *
 * Randomly masks rectangular regions of the image to improve model robustness
 * by forcing the model to rely on diverse features rather than specific regions.
 *
 * Supports:
 * - Single or multiple cutouts
 * - Configurable size and aspect ratio ranges
 * - Constant color, noise, or random color fill modes
 * - Probability-based application
 * - Reproducible results via seed
 *
 * ## Example Usage (from JS):
 * ```typescript
 * const result = await cutout(
 *   { uri: 'file://image.jpg' },
 *   {
 *     numCutouts: 2,
 *     minSize: 0.02,
 *     maxSize: 0.33,
 *     fillMode: 'noise',
 *     seed: 42
 *   }
 * );
 * ```
 */
class Cutout {

    // MARK: - Public API

    /**
     * Apply cutout augmentation to an image.
     *
     * @param image Source UIImage
     * @param options Cutout options
     * @returns Dictionary containing augmented image and cutout details
     * @throws VisionUtilsError on processing failure
     */
    static func apply(to image: UIImage, options: [String: Any]) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Parse options
        let numCutouts = (options["numCutouts"] as? Int) ?? 1
        let minSize = (options["minSize"] as? Double) ?? 0.02
        let maxSize = (options["maxSize"] as? Double) ?? 0.33
        let minAspect = (options["minAspect"] as? Double) ?? 0.3
        let maxAspect = (options["maxAspect"] as? Double) ?? 3.3
        let fillMode = (options["fillMode"] as? String) ?? "constant"
        let fillValue = parseFillValue(options["fillValue"])
        let probability = (options["probability"] as? Double) ?? 1.0

        // Setup random number generator
        let requestedSeed = options["seed"] as? Int
        let effectiveSeed = requestedSeed ?? Int.random(in: 1...Int.max)
        var rng = SeededRandomNumberGenerator(seed: UInt64(effectiveSeed))

        // Check probability
        let shouldApply = Double.random(in: 0...1, using: &rng) < probability

        let width = Int(image.size.width * image.scale)
        let height = Int(image.size.height * image.scale)
        let imageArea = Double(width * height)

        var regions: [[String: Any]] = []

        if !shouldApply || numCutouts == 0 {
            // Return original image without cutouts
            guard let pngData = image.pngData() else {
                throw VisionUtilsError.processingError("Failed to encode image")
            }
            let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

            return [
                "base64": pngData.base64EncodedString(),
                "width": width,
                "height": height,
                "applied": false,
                "numCutouts": 0,
                "regions": regions,
                "seed": effectiveSeed,
                "processingTimeMs": processingTimeMs
            ]
        }

        // Create mutable image context
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        guard let context = UIGraphicsGetCurrentContext() else {
            throw VisionUtilsError.processingError("Failed to create graphics context")
        }

        // Draw original image
        image.draw(at: .zero)

        // Apply cutouts
        for _ in 0..<numCutouts {
            // Random area within range
            let targetArea = Double.random(in: minSize...maxSize, using: &rng) * imageArea

            // Random aspect ratio
            let aspectRatio = Double.random(in: minAspect...maxAspect, using: &rng)

            // Calculate dimensions
            var cutoutWidth = Int(sqrt(targetArea * aspectRatio))
            var cutoutHeight = Int(sqrt(targetArea / aspectRatio))

            // Clamp to image bounds
            cutoutWidth = min(cutoutWidth, width)
            cutoutHeight = min(cutoutHeight, height)

            // Random position
            let x = Int.random(in: 0...(width - cutoutWidth), using: &rng)
            let y = Int.random(in: 0...(height - cutoutHeight), using: &rng)

            // Get fill color
            let (fillColor, fillDescription) = getFillColor(
                mode: fillMode,
                constantValue: fillValue,
                width: cutoutWidth,
                height: cutoutHeight,
                rng: &rng
            )

            // Draw cutout region
            let rect = CGRect(
                x: CGFloat(x) / image.scale,
                y: CGFloat(y) / image.scale,
                width: CGFloat(cutoutWidth) / image.scale,
                height: CGFloat(cutoutHeight) / image.scale
            )

            if fillMode == "noise" {
                // Fill with noise
                drawNoiseRect(context: context, rect: rect, rng: &rng)
            } else {
                context.setFillColor(fillColor.cgColor)
                context.fill(rect)
            }

            // Record region info
            var regionInfo: [String: Any] = [
                "x": x,
                "y": y,
                "width": cutoutWidth,
                "height": cutoutHeight
            ]

            if fillMode == "noise" {
                regionInfo["fill"] = "noise"
            } else {
                regionInfo["fill"] = fillDescription
            }

            regions.append(regionInfo)
        }

        guard let resultImage = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            throw VisionUtilsError.processingError("Failed to get result image")
        }
        UIGraphicsEndImageContext()

        guard let pngData = resultImage.pngData() else {
            throw VisionUtilsError.processingError("Failed to encode cutout image")
        }

        let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        return [
            "base64": pngData.base64EncodedString(),
            "width": width,
            "height": height,
            "applied": true,
            "numCutouts": regions.count,
            "regions": regions,
            "seed": effectiveSeed,
            "processingTimeMs": processingTimeMs
        ]
    }

    // MARK: - Private Helpers

    private static func parseFillValue(_ value: Any?) -> [Int] {
        if let array = value as? [NSNumber], array.count >= 3 {
            return [array[0].intValue, array[1].intValue, array[2].intValue]
        }
        return [0, 0, 0] // Default black
    }

    private static func getFillColor(
        mode: String,
        constantValue: [Int],
        width: Int,
        height: Int,
        rng: inout SeededRandomNumberGenerator
    ) -> (UIColor, [Int]) {
        switch mode {
        case "random":
            let r = Int.random(in: 0...255, using: &rng)
            let g = Int.random(in: 0...255, using: &rng)
            let b = Int.random(in: 0...255, using: &rng)
            let color = UIColor(
                red: CGFloat(r) / 255.0,
                green: CGFloat(g) / 255.0,
                blue: CGFloat(b) / 255.0,
                alpha: 1.0
            )
            return (color, [r, g, b])

        case "noise":
            // Will be handled separately
            return (UIColor.black, [0, 0, 0])

        default: // "constant"
            let color = UIColor(
                red: CGFloat(constantValue[0]) / 255.0,
                green: CGFloat(constantValue[1]) / 255.0,
                blue: CGFloat(constantValue[2]) / 255.0,
                alpha: 1.0
            )
            return (color, constantValue)
        }
    }

    private static func drawNoiseRect(
        context: CGContext,
        rect: CGRect,
        rng: inout SeededRandomNumberGenerator
    ) {
        let width = Int(rect.width)
        let height = Int(rect.height)

        guard width > 0 && height > 0 else { return }

        // Create pixel buffer for noise
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        for i in 0..<(width * height) {
            let offset = i * 4
            pixelData[offset] = UInt8.random(in: 0...255, using: &rng)     // R
            pixelData[offset + 1] = UInt8.random(in: 0...255, using: &rng) // G
            pixelData[offset + 2] = UInt8.random(in: 0...255, using: &rng) // B
            pixelData[offset + 3] = 255                                     // A
        }

        // Create CGImage from pixel data
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let dataProvider = CGDataProvider(data: Data(pixelData) as CFData),
              let noiseImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: dataProvider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            // Fallback to black fill
            context.setFillColor(UIColor.black.cgColor)
            context.fill(rect)
            return
        }

        context.draw(noiseImage, in: rect)
    }
}
