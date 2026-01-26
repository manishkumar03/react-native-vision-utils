import Foundation
import UIKit
import CoreGraphics

/// Handles multi-crop operations (five-crop, ten-crop)
class MultiCrop {

    // MARK: - Five Crop

    /// Perform five-crop operation (4 corners + center)
    static func fiveCrop(
        image: UIImage,
        options: [String: Any],
        pixelOptions: [String: Any]
    ) throws -> [String: Any] {
        guard let cgImage = image.cgImage else {
            throw VisionUtilsError.processingError("Failed to get CGImage")
        }

        let imageWidth = cgImage.width
        let imageHeight = cgImage.height

        let cropWidth = options["width"] as? Int ?? 224
        let cropHeight = options["height"] as? Int ?? 224

        guard cropWidth <= imageWidth && cropHeight <= imageHeight else {
            throw VisionUtilsError.processingError("Crop size must be smaller than image size")
        }

        // Calculate crop positions
        let crops = [
            (0, 0),                                                    // Top-left
            (imageWidth - cropWidth, 0),                              // Top-right
            (0, imageHeight - cropHeight),                            // Bottom-left
            (imageWidth - cropWidth, imageHeight - cropHeight),       // Bottom-right
            ((imageWidth - cropWidth) / 2, (imageHeight - cropHeight) / 2)  // Center
        ]

        let cropNames = ["topLeft", "topRight", "bottomLeft", "bottomRight", "center"]

        return try processCrops(
            cgImage: cgImage,
            crops: crops,
            cropNames: cropNames,
            cropWidth: cropWidth,
            cropHeight: cropHeight,
            pixelOptions: pixelOptions
        )
    }

    // MARK: - Ten Crop

    /// Perform ten-crop operation (five-crop + horizontal flips)
    static func tenCrop(
        image: UIImage,
        options: [String: Any],
        pixelOptions: [String: Any]
    ) throws -> [String: Any] {
        guard let cgImage = image.cgImage else {
            throw VisionUtilsError.processingError("Failed to get CGImage")
        }

        let imageWidth = cgImage.width
        let imageHeight = cgImage.height

        let cropWidth = options["width"] as? Int ?? 224
        let cropHeight = options["height"] as? Int ?? 224
        let includeFlips = (options["includeFlips"] as? Bool) ?? true

        guard cropWidth <= imageWidth && cropHeight <= imageHeight else {
            throw VisionUtilsError.processingError("Crop size must be smaller than image size")
        }

        // Calculate crop positions
        var crops = [
            (0, 0),                                                    // Top-left
            (imageWidth - cropWidth, 0),                              // Top-right
            (0, imageHeight - cropHeight),                            // Bottom-left
            (imageWidth - cropWidth, imageHeight - cropHeight),       // Bottom-right
            ((imageWidth - cropWidth) / 2, (imageHeight - cropHeight) / 2)  // Center
        ]

        var cropNames = ["topLeft", "topRight", "bottomLeft", "bottomRight", "center"]
        var flipFlags = [false, false, false, false, false]

        if includeFlips {
            // Add flipped versions
            crops += crops
            cropNames += ["topLeftFlipped", "topRightFlipped", "bottomLeftFlipped", "bottomRightFlipped", "centerFlipped"]
            flipFlags += [true, true, true, true, true]
        }

        return try processCrops(
            cgImage: cgImage,
            crops: crops,
            cropNames: cropNames,
            cropWidth: cropWidth,
            cropHeight: cropHeight,
            pixelOptions: pixelOptions,
            flipFlags: flipFlags
        )
    }

    // MARK: - Helper

    private static func processCrops(
        cgImage: CGImage,
        crops: [(Int, Int)],
        cropNames: [String],
        cropWidth: Int,
        cropHeight: Int,
        pixelOptions: [String: Any],
        flipFlags: [Bool]? = nil
    ) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()
        var results: [[String: Any]] = []

        for (idx, (x, y)) in crops.enumerated() {
            let cropRect = CGRect(x: x, y: y, width: cropWidth, height: cropHeight)

            guard let croppedCGImage = cgImage.cropping(to: cropRect) else {
                throw VisionUtilsError.processingError("Failed to crop image at position (\(x), \(y))")
            }

            var finalImage = croppedCGImage

            // Apply horizontal flip if needed
            if let flags = flipFlags, flags[idx] {
                finalImage = try flipImageHorizontally(croppedCGImage)
            }

            // Process the cropped image with pixel options (no source needed - image already loaded)
            let uiImage = UIImage(cgImage: finalImage)
            let parsedOptions = try GetPixelDataOptions(fromPixelOptions: pixelOptions)
            let result = try PixelProcessor.process(image: uiImage, options: parsedOptions)

            var resultDict = result.toDictionary()
            resultDict["cropName"] = cropNames[idx]
            resultDict["cropPosition"] = ["x": x, "y": y]
            results.append(resultDict)
        }

        let endTime = CFAbsoluteTimeGetCurrent()
        let totalTimeMs = (endTime - startTime) * 1000

        return [
            "crops": results,
            "count": results.count,
            "cropWidth": cropWidth,
            "cropHeight": cropHeight,
            "totalTimeMs": totalTimeMs
        ]
    }

    private static func flipImageHorizontally(_ image: CGImage) throws -> CGImage {
        let width = image.width
        let height = image.height

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw VisionUtilsError.processingError("Failed to create context for flipping")
        }

        // Flip horizontally
        context.translateBy(x: CGFloat(width), y: 0)
        context.scaleBy(x: -1, y: 1)
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let flippedImage = context.makeImage() else {
            throw VisionUtilsError.processingError("Failed to create flipped image")
        }

        return flippedImage
    }
}
