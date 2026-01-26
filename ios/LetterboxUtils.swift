import Foundation
import UIKit
import CoreGraphics

/// Utility class for letterbox padding operations
@objc(LetterboxUtils)
public class LetterboxUtils: NSObject {

    // MARK: - Letterbox Operation

    /// Apply letterbox padding to an image
    @objc
    public static func letterbox(
        image: UIImage,
        targetWidth: Int,
        targetHeight: Int,
        padColor: [Int],
        scaleUp: Bool,
        autoStride: Bool,
        stride: Int,
        center: Bool
    ) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        guard let cgImage = image.cgImage else {
            throw VisionUtilsError.invalidSource("Failed to get CGImage")
        }

        let originalWidth = cgImage.width
        let originalHeight = cgImage.height

        // Calculate scale factor
        let scaleW = Double(targetWidth) / Double(originalWidth)
        let scaleH = Double(targetHeight) / Double(originalHeight)
        var scale = min(scaleW, scaleH)

        // Don't scale up if not allowed
        if !scaleUp {
            scale = min(scale, 1.0)
        }

        // Calculate new dimensions
        var newWidth = Int(round(Double(originalWidth) * scale))
        var newHeight = Int(round(Double(originalHeight) * scale))

        // Apply stride alignment if requested
        if autoStride {
            newWidth = ((newWidth + stride - 1) / stride) * stride
            newHeight = ((newHeight + stride - 1) / stride) * stride
        }

        // Calculate padding
        let padW = targetWidth - newWidth
        let padH = targetHeight - newHeight

        var padLeft: Int, padTop: Int, padRight: Int, padBottom: Int
        if center {
            padLeft = padW / 2
            padTop = padH / 2
            padRight = padW - padLeft
            padBottom = padH - padTop
        } else {
            padLeft = 0
            padTop = 0
            padRight = padW
            padBottom = padH
        }

        // Create the letterboxed image
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw VisionUtilsError.processingFailed("Failed to create graphics context")
        }

        // Fill with pad color
        let r = CGFloat(padColor.count > 0 ? padColor[0] : 114) / 255.0
        let g = CGFloat(padColor.count > 1 ? padColor[1] : 114) / 255.0
        let b = CGFloat(padColor.count > 2 ? padColor[2] : 114) / 255.0
        context.setFillColor(red: r, green: g, blue: b, alpha: 1.0)
        context.fill(CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        // Draw the scaled image
        let drawRect = CGRect(
            x: padLeft,
            y: padTop,
            width: newWidth,
            height: newHeight
        )
        context.interpolationQuality = .high
        context.draw(cgImage, in: drawRect)

        // Create output image
        guard let outputCGImage = context.makeImage() else {
            throw VisionUtilsError.processingFailed("Failed to create output image")
        }
        let outputImage = UIImage(cgImage: outputCGImage)

        // Convert to base64
        guard let imageData = outputImage.jpegData(compressionQuality: 0.9) else {
            throw VisionUtilsError.processingFailed("Failed to encode image")
        }
        let base64String = imageData.base64EncodedString()

        // Create letterbox info for reverse transformation
        let letterboxInfo: [String: Any] = [
            "scale": scale,
            "padding": [padLeft, padTop, padRight, padBottom],
            "offset": [padLeft, padTop],
            "originalSize": [originalWidth, originalHeight],
            "letterboxedSize": [targetWidth, targetHeight]
        ]

        let endTime = CFAbsoluteTimeGetCurrent()
        let processingTimeMs = (endTime - startTime) * 1000

        return [
            "imageBase64": base64String,
            "width": targetWidth,
            "height": targetHeight,
            "scale": scale,
            "padding": [padLeft, padTop, padRight, padBottom],
            "offset": [padLeft, padTop],
            "originalSize": [originalWidth, originalHeight],
            "letterboxInfo": letterboxInfo,
            "processingTimeMs": processingTimeMs
        ]
    }

    // MARK: - Reverse Letterbox

    /// Result of reverse letterbox operation
    public struct ReverseLetterboxResult {
        public let boxes: [[Double]]
        public let format: String
        public let processingTimeMs: Double
    }

    /// Reverse letterbox transformation on bounding boxes
    public static func reverseLetterbox(
        boxes: [[Double]],
        scale: Double,
        offset: [Double],
        originalWidth: Int,
        originalHeight: Int,
        format: String,
        clip: Bool
    ) throws -> ReverseLetterboxResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        let offsetX = offset.count > 0 ? offset[0] : 0.0
        let offsetY = offset.count > 1 ? offset[1] : 0.0

        // Convert to xyxy for transformation
        var xyxyBoxes: [[Double]]
        if format != "xyxy" {
            xyxyBoxes = try BoundingBoxUtils.convertBoxFormat(
                boxes: boxes,
                sourceFormat: format,
                targetFormat: "xyxy"
            )
        } else {
            xyxyBoxes = boxes
        }

        // Apply reverse transformation
        var transformedBoxes = xyxyBoxes.map { box -> [Double] in
            guard box.count == 4 else { return box }
            return [
                (box[0] - offsetX) / scale,
                (box[1] - offsetY) / scale,
                (box[2] - offsetX) / scale,
                (box[3] - offsetY) / scale
            ]
        }

        // Clip if requested
        if clip {
            transformedBoxes = transformedBoxes.map { box -> [Double] in
                guard box.count == 4 else { return box }
                return [
                    max(0, min(Double(originalWidth), box[0])),
                    max(0, min(Double(originalHeight), box[1])),
                    max(0, min(Double(originalWidth), box[2])),
                    max(0, min(Double(originalHeight), box[3]))
                ]
            }
        }

        // Convert back to original format if needed
        var finalBoxes = transformedBoxes
        if format != "xyxy" {
            finalBoxes = try BoundingBoxUtils.convertBoxFormat(
                boxes: transformedBoxes,
                sourceFormat: "xyxy",
                targetFormat: format
            )
        }

        let endTime = CFAbsoluteTimeGetCurrent()
        return ReverseLetterboxResult(
            boxes: finalBoxes,
            format: format,
            processingTimeMs: (endTime - startTime) * 1000
        )
    }

    // MARK: - Helper to apply letterbox and get pixel data

    /// Convenience method to letterbox and return as pixel data for ML inference
    @objc
    public static func letterboxForInference(
        image: UIImage,
        targetSize: Int,
        normalize: Bool
    ) throws -> [String: Any] {
        let result = try letterbox(
            image: image,
            targetWidth: targetSize,
            targetHeight: targetSize,
            padColor: [114, 114, 114],
            scaleUp: true,
            autoStride: false,
            stride: 32,
            center: true
        )

        // Add letterbox info for reverse transformation
        var letterboxInfo: [String: Any] = [
            "scale": result["scale"] as Any,
            "padding": result["padding"] as Any,
            "originalSize": result["originalSize"] as Any,
            "letterboxedSize": [result["width"] as Any, result["height"] as Any]
        ]

        var mutableResult = result
        mutableResult["letterboxInfo"] = letterboxInfo

        return mutableResult
    }
}
