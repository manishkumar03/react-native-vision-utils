import Foundation
import UIKit
import CoreImage
import CoreGraphics

/**
 * ColorJitter provides granular color augmentation for data augmentation pipelines.
 *
 * Supports random sampling within specified ranges for:
 * - Brightness: additive adjustment
 * - Contrast: multiplicative adjustment around mean
 * - Saturation: multiplicative adjustment
 * - Hue: cyclic shift
 *
 * ## Example Usage (from JS):
 * ```typescript
 * const result = await colorJitter(
 *   { uri: 'file://image.jpg' },
 *   {
 *     brightness: 0.2,          // [-0.2, +0.2]
 *     contrast: [0.8, 1.2],     // [0.8, 1.2]
 *     saturation: 0.3,          // [0.7, 1.3]
 *     hue: 0.1,                 // [-0.1, +0.1]
 *     seed: 42                  // for reproducibility
 *   }
 * );
 * ```
 */
class ColorJitter {

    // MARK: - Public API

    /**
     * Apply color jitter augmentation to an image.
     *
     * @param image Source UIImage
     * @param options Color jitter options (brightness, contrast, saturation, hue, seed)
     * @returns Dictionary containing augmented image and applied values
     * @throws VisionUtilsError on processing failure
     */
    static func apply(to image: UIImage, options: [String: Any]) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Parse options and get ranges
        let brightnessRange = parseRange(options["brightness"], defaultMin: 0, defaultMax: 0)
        let contrastRange = parseRange(options["contrast"], defaultMin: 1, defaultMax: 1, isMultiplicative: true)
        let saturationRange = parseRange(options["saturation"], defaultMin: 1, defaultMax: 1, isMultiplicative: true)
        let hueRange = parseRange(options["hue"], defaultMin: 0, defaultMax: 0)

        // Setup random number generator
        let requestedSeed = options["seed"] as? Int
        let effectiveSeed = requestedSeed ?? Int.random(in: 1...Int.max)
        var rng = SeededRandomNumberGenerator(seed: UInt64(effectiveSeed))

        // Sample random values within ranges
        let appliedBrightness = randomInRange(brightnessRange, using: &rng)
        let appliedContrast = randomInRange(contrastRange, using: &rng)
        let appliedSaturation = randomInRange(saturationRange, using: &rng)
        let appliedHue = randomInRange(hueRange, using: &rng)

        // Apply transformations
        var currentImage = image

        // Apply brightness
        if appliedBrightness != 0 {
            currentImage = try adjustBrightness(currentImage, value: appliedBrightness)
        }

        // Apply contrast
        if appliedContrast != 1 {
            currentImage = try adjustContrast(currentImage, value: appliedContrast)
        }

        // Apply saturation
        if appliedSaturation != 1 {
            currentImage = try adjustSaturation(currentImage, value: appliedSaturation)
        }

        // Apply hue shift
        if appliedHue != 0 {
            currentImage = try adjustHue(currentImage, value: appliedHue)
        }

        // Convert to base64
        guard let pngData = currentImage.pngData() else {
            throw VisionUtilsError.processingError("Failed to encode color-jittered image")
        }

        let base64String = pngData.base64EncodedString()
        let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        return [
            "base64": base64String,
            "width": Int(currentImage.size.width * currentImage.scale),
            "height": Int(currentImage.size.height * currentImage.scale),
            "appliedBrightness": appliedBrightness,
            "appliedContrast": appliedContrast,
            "appliedSaturation": appliedSaturation,
            "appliedHue": appliedHue,
            "seed": effectiveSeed,
            "processingTimeMs": processingTimeMs
        ]
    }

    // MARK: - Range Parsing

    /**
     * Parse a range value from options.
     * - Single number: symmetric range around default
     * - Tuple [min, max]: explicit range
     */
    private static func parseRange(
        _ value: Any?,
        defaultMin: Double,
        defaultMax: Double,
        isMultiplicative: Bool = false
    ) -> (min: Double, max: Double) {
        guard let value = value else {
            return (defaultMin, defaultMax)
        }

        if let tuple = value as? [NSNumber], tuple.count == 2 {
            return (tuple[0].doubleValue, tuple[1].doubleValue)
        }

        if let number = value as? NSNumber {
            let v = number.doubleValue
            if isMultiplicative {
                // For contrast/saturation: range is [max(0, 1-v), 1+v]
                return (max(0, 1 - v), 1 + v)
            } else {
                // For brightness/hue: range is [-v, +v]
                return (-v, v)
            }
        }

        return (defaultMin, defaultMax)
    }

    /**
     * Generate a random value within a range.
     */
    private static func randomInRange(
        _ range: (min: Double, max: Double),
        using rng: inout SeededRandomNumberGenerator
    ) -> Double {
        if range.min == range.max {
            return range.min
        }
        return Double.random(in: range.min...range.max, using: &rng)
    }

    // MARK: - Color Adjustments

    private static func adjustBrightness(_ image: UIImage, value: Double) throws -> UIImage {
        guard let ciImage = CIImage(image: image) else {
            throw VisionUtilsError.processingError("Failed to create CIImage for brightness")
        }

        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(value, forKey: kCIInputBrightnessKey)

        guard let outputImage = filter.outputImage else {
            throw VisionUtilsError.processingError("Failed to apply brightness filter")
        }

        return try renderCIImage(outputImage, scale: image.scale, orientation: image.imageOrientation)
    }

    private static func adjustContrast(_ image: UIImage, value: Double) throws -> UIImage {
        guard let ciImage = CIImage(image: image) else {
            throw VisionUtilsError.processingError("Failed to create CIImage for contrast")
        }

        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(value, forKey: kCIInputContrastKey)

        guard let outputImage = filter.outputImage else {
            throw VisionUtilsError.processingError("Failed to apply contrast filter")
        }

        return try renderCIImage(outputImage, scale: image.scale, orientation: image.imageOrientation)
    }

    private static func adjustSaturation(_ image: UIImage, value: Double) throws -> UIImage {
        guard let ciImage = CIImage(image: image) else {
            throw VisionUtilsError.processingError("Failed to create CIImage for saturation")
        }

        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(value, forKey: kCIInputSaturationKey)

        guard let outputImage = filter.outputImage else {
            throw VisionUtilsError.processingError("Failed to apply saturation filter")
        }

        return try renderCIImage(outputImage, scale: image.scale, orientation: image.imageOrientation)
    }

    private static func adjustHue(_ image: UIImage, value: Double) throws -> UIImage {
        guard let ciImage = CIImage(image: image) else {
            throw VisionUtilsError.processingError("Failed to create CIImage for hue")
        }

        // CIHueAdjust expects angle in radians (value is fraction of color wheel, 0-1 = 0-360°)
        let angleInRadians = value * 2 * Double.pi

        let filter = CIFilter(name: "CIHueAdjust")!
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(angleInRadians, forKey: kCIInputAngleKey)

        guard let outputImage = filter.outputImage else {
            throw VisionUtilsError.processingError("Failed to apply hue filter")
        }

        return try renderCIImage(outputImage, scale: image.scale, orientation: image.imageOrientation)
    }

    // MARK: - Helper

    private static func renderCIImage(
        _ ciImage: CIImage,
        scale: CGFloat,
        orientation: UIImage.Orientation
    ) throws -> UIImage {
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw VisionUtilsError.processingError("Failed to render CIImage to CGImage")
        }
        return UIImage(cgImage: cgImage, scale: scale, orientation: orientation)
    }
}
