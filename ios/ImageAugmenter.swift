import Foundation
import UIKit
import CoreGraphics
import CoreImage
import Accelerate

/// Handles image augmentation operations
class ImageAugmenter {

    /// Apply augmentations to an image
    static func apply(
        to image: UIImage,
        augmentations: [String: Any]
    ) throws -> [String: Any] {
        var currentImage = image

        // Apply rotation
        if let rotation = augmentations["rotation"] as? Double, rotation != 0 {
            currentImage = try rotateImage(currentImage, degrees: rotation)
        }

        // Apply horizontal flip
        if let flipHorizontal = augmentations["flipHorizontal"] as? Bool, flipHorizontal {
            currentImage = flipImageHorizontally(currentImage)
        }

        // Apply vertical flip
        if let flipVertical = augmentations["flipVertical"] as? Bool, flipVertical {
            currentImage = flipImageVertically(currentImage)
        }

        // Apply brightness
        if let brightness = augmentations["brightness"] as? Double, brightness != 0 {
            currentImage = try adjustBrightness(currentImage, factor: brightness)
        }

        // Apply contrast
        if let contrast = augmentations["contrast"] as? Double, contrast != 1 {
            currentImage = try adjustContrast(currentImage, factor: contrast)
        }

        // Apply saturation
        if let saturation = augmentations["saturation"] as? Double, saturation != 1 {
            currentImage = try adjustSaturation(currentImage, factor: saturation)
        }

        // Apply blur
        if let blur = augmentations["blur"] as? Double, blur > 0 {
            currentImage = try applyBlur(currentImage, radius: blur)
        }

        // Convert to base64
        guard let pngData = currentImage.pngData() else {
            throw VisionUtilsError.processingError("Failed to encode augmented image")
        }

        let base64String = pngData.base64EncodedString()

        return [
            "base64": "data:image/png;base64,\(base64String)",
            "width": Int(currentImage.size.width * currentImage.scale),
            "height": Int(currentImage.size.height * currentImage.scale)
        ]
    }

    // MARK: - Rotation

    private static func rotateImage(_ image: UIImage, degrees: Double) throws -> UIImage {
        let radians = degrees * .pi / 180

        var newSize = CGRect(origin: .zero, size: image.size)
            .applying(CGAffineTransform(rotationAngle: CGFloat(radians)))
            .size
        newSize.width = floor(newSize.width)
        newSize.height = floor(newSize.height)

        UIGraphicsBeginImageContextWithOptions(newSize, false, image.scale)
        defer { UIGraphicsEndImageContext() }

        guard let context = UIGraphicsGetCurrentContext() else {
            throw VisionUtilsError.processingError("Failed to create context for rotation")
        }

        context.translateBy(x: newSize.width / 2, y: newSize.height / 2)
        context.rotate(by: CGFloat(radians))

        image.draw(in: CGRect(
            x: -image.size.width / 2,
            y: -image.size.height / 2,
            width: image.size.width,
            height: image.size.height
        ))

        guard let rotatedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            throw VisionUtilsError.processingError("Failed to create rotated image")
        }

        return rotatedImage
    }

    // MARK: - Flip

    private static func flipImageHorizontally(_ image: UIImage) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        defer { UIGraphicsEndImageContext() }

        guard let context = UIGraphicsGetCurrentContext() else { return image }

        context.translateBy(x: image.size.width, y: 0)
        context.scaleBy(x: -1, y: 1)
        image.draw(at: .zero)

        return UIGraphicsGetImageFromCurrentImageContext() ?? image
    }

    private static func flipImageVertically(_ image: UIImage) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        defer { UIGraphicsEndImageContext() }

        guard let context = UIGraphicsGetCurrentContext() else { return image }

        context.translateBy(x: 0, y: image.size.height)
        context.scaleBy(x: 1, y: -1)
        image.draw(at: .zero)

        return UIGraphicsGetImageFromCurrentImageContext() ?? image
    }

    // MARK: - Brightness

    private static func adjustBrightness(_ image: UIImage, factor: Double) throws -> UIImage {
        guard let ciImage = CIImage(image: image) else {
            throw VisionUtilsError.processingError("Failed to create CIImage for brightness")
        }

        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(factor, forKey: kCIInputBrightnessKey)

        guard let outputImage = filter.outputImage else {
            throw VisionUtilsError.processingError("Failed to apply brightness filter")
        }

        let context = CIContext()
        guard let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else {
            throw VisionUtilsError.processingError("Failed to create CGImage from brightness filter")
        }

        return UIImage(cgImage: cgImage, scale: image.scale, orientation: image.imageOrientation)
    }

    // MARK: - Contrast

    private static func adjustContrast(_ image: UIImage, factor: Double) throws -> UIImage {
        guard let ciImage = CIImage(image: image) else {
            throw VisionUtilsError.processingError("Failed to create CIImage for contrast")
        }

        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(factor, forKey: kCIInputContrastKey)

        guard let outputImage = filter.outputImage else {
            throw VisionUtilsError.processingError("Failed to apply contrast filter")
        }

        let context = CIContext()
        guard let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else {
            throw VisionUtilsError.processingError("Failed to create CGImage from contrast filter")
        }

        return UIImage(cgImage: cgImage, scale: image.scale, orientation: image.imageOrientation)
    }

    // MARK: - Saturation

    private static func adjustSaturation(_ image: UIImage, factor: Double) throws -> UIImage {
        guard let ciImage = CIImage(image: image) else {
            throw VisionUtilsError.processingError("Failed to create CIImage for saturation")
        }

        let filter = CIFilter(name: "CIColorControls")!
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(factor, forKey: kCIInputSaturationKey)

        guard let outputImage = filter.outputImage else {
            throw VisionUtilsError.processingError("Failed to apply saturation filter")
        }

        let context = CIContext()
        guard let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else {
            throw VisionUtilsError.processingError("Failed to create CGImage from saturation filter")
        }

        return UIImage(cgImage: cgImage, scale: image.scale, orientation: image.imageOrientation)
    }

    // MARK: - Blur

    private static func applyBlur(_ image: UIImage, radius: Double) throws -> UIImage {
        guard let ciImage = CIImage(image: image) else {
            throw VisionUtilsError.processingError("Failed to create CIImage for blur")
        }

        let filter = CIFilter(name: "CIGaussianBlur")!
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(radius, forKey: kCIInputRadiusKey)

        guard let outputImage = filter.outputImage else {
            throw VisionUtilsError.processingError("Failed to apply blur filter")
        }

        // Crop to original extent to remove blur edge artifacts
        let croppedImage = outputImage.cropped(to: ciImage.extent)

        let context = CIContext()
        guard let cgImage = context.createCGImage(croppedImage, from: ciImage.extent) else {
            throw VisionUtilsError.processingError("Failed to create CGImage from blur filter")
        }

        return UIImage(cgImage: cgImage, scale: image.scale, orientation: image.imageOrientation)
    }
}
