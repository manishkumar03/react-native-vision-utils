import Foundation
import UIKit
import CoreGraphics

/**
 * RandomCropper provides functionality for extracting random crops from images.
 *
 * This is useful for:
 * - Data augmentation during training
 * - Creating diverse training samples
 * - Reproducible random crops using seeds
 *
 * ## Example Usage (from JS):
 * ```typescript
 * const result = await randomCrop(
 *   { uri: 'file://image.jpg' },
 *   { width: 224, height: 224, count: 5, seed: 42 },
 *   { outputFormat: 'float32', layout: 'NCHW' }
 * );
 * ```
 */
class RandomCropper {

    // MARK: - Public API

    /**
     * Extract random crops from an image.
     *
     * @param source Source specification (uri, base64, or frame)
     * @param cropOptions Random crop options
     * @param pixelOptions Pixel data output options
     * @returns Dictionary containing crops array and metadata
     * @throws VisionUtilsError on invalid input or processing failure
     */
    static func randomCrop(
        source: [String: Any],
        cropOptions: [String: Any],
        pixelOptions: [String: Any]
    ) async throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()
        // Parse and load the source image
        let imageSource = try ImageSource(from: source)
        let image = try await ImageLoader.loadImage(from: imageSource)

        // Parse crop options with defaults
        let cropWidth = cropOptions["width"] as? Int ?? 224
        let cropHeight = cropOptions["height"] as? Int ?? 224
        let count = cropOptions["count"] as? Int ?? 1
        let requestedSeed = cropOptions["seed"] as? Int
        let effectiveSeed = requestedSeed ?? Int.random(in: 1...Int.max)

        // Validate dimensions
        guard cropWidth > 0, cropHeight > 0 else {
            throw VisionUtilsError.invalidInput("Crop dimensions must be positive")
        }
        guard count > 0 else {
            throw VisionUtilsError.invalidInput("Count must be positive")
        }

        let imageWidth = Int(image.size.width)
        let imageHeight = Int(image.size.height)

        // Check if image is large enough for the requested crop size
        guard imageWidth >= cropWidth, imageHeight >= cropHeight else {
            throw VisionUtilsError.invalidInput(
                "Image (\(imageWidth)x\(imageHeight)) is smaller than requested crop size (\(cropWidth)x\(cropHeight))"
            )
        }

        // Setup random number generator with optional seed
        var rng: RandomNumberGenerator = SeededRandomNumberGenerator(seed: UInt64(effectiveSeed))

        // Generate random crops
        var crops: [[String: Any]] = []

        for i in 0..<count {
            // Generate random position
            let maxX = imageWidth - cropWidth
            let maxY = imageHeight - cropHeight
            let x = maxX > 0 ? Int.random(in: 0...maxX, using: &rng) : 0
            let y = maxY > 0 ? Int.random(in: 0...maxY, using: &rng) : 0

            // Extract the crop
            let cropRect = CGRect(x: x, y: y, width: cropWidth, height: cropHeight)
            guard let croppedImage = cropImage(image, to: cropRect) else {
                throw VisionUtilsError.processingError("Failed to extract crop at (\(x), \(y))")
            }

            // Get pixel data for this crop
            let parsedOptions = try GetPixelDataOptions(fromPixelOptions: pixelOptions)
                let pixelResult = try PixelProcessor.process(image: croppedImage, options: parsedOptions)

            // Build crop info
            let cropInfo: [String: Any] = [
                "index": i,
                "x": x,
                "y": y,
                "width": cropWidth,
                "height": cropHeight,
                "data": pixelResult.data,
                    "seed": effectiveSeed
            ]

            crops.append(cropInfo)
        }

        let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        return [
            "crops": crops,
            "cropCount": crops.count,
            "cropWidth": cropWidth,
            "cropHeight": cropHeight,
            "seed": effectiveSeed,
            "originalWidth": imageWidth,
            "originalHeight": imageHeight,
            "processingTimeMs": processingTimeMs
        ]
    }

    // MARK: - Private Helpers

    /**
     * Crop a UIImage to the specified rectangle.
     */
    private static func cropImage(_ image: UIImage, to rect: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        // Convert to image coordinates (UIImage origin is top-left)
        let scale = image.scale
        let scaledRect = CGRect(
            x: rect.origin.x * scale,
            y: rect.origin.y * scale,
            width: rect.width * scale,
            height: rect.height * scale
        )

        guard let croppedCGImage = cgImage.cropping(to: scaledRect) else { return nil }
        return UIImage(cgImage: croppedCGImage, scale: scale, orientation: image.imageOrientation)
    }
}

// MARK: - Seeded Random Number Generator

/**
 * A simple seeded random number generator using xorshift algorithm.
 * This provides reproducible random sequences when given the same seed.
 */
struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        // Ensure non-zero state
        self.state = seed == 0 ? 1 : seed
    }

    mutating func next() -> UInt64 {
        // xorshift64* algorithm
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return state &* 0x2545F4914F6CDD1D
    }
}
