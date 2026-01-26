import Foundation
import UIKit
import CoreGraphics
import Accelerate

// MARK: - Pixel Processor

/// Handles all pixel data extraction and transformation operations
class PixelProcessor {

    // MARK: - Main Processing Function

    /// Process an image with the given options and return pixel data
    static func process(image: UIImage, options: GetPixelDataOptions) throws -> VisionUtilsResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        guard var cgImage = image.cgImage else {
            throw VisionUtilsError.processingError("Failed to get CGImage from UIImage")
        }

        // Apply ROI (crop) if specified
        if let roi = options.roi {
            cgImage = try applyRoi(to: cgImage, roi: roi)
        }

        // Apply resize if specified
        let (processedImage, finalWidth, finalHeight) = try applyResize(
            to: cgImage,
            resize: options.resize
        )

        // Extract pixel data in RGBA format
        let rgbaData = try extractRGBAPixels(from: processedImage, width: finalWidth, height: finalHeight)

        // Convert to requested color format
        let colorData = convertColorFormat(rgbaData, to: options.colorFormat)

        // Apply normalization
        let normalizedData = applyNormalization(colorData, options: options)

        // Convert to requested layout
        let layoutData = convertLayout(
            normalizedData,
            width: finalWidth,
            height: finalHeight,
            channels: options.colorFormat.channelCount,
            layout: options.layout
        )

        // Calculate shape
        let shape = calculateShape(
            width: finalWidth,
            height: finalHeight,
            channels: options.colorFormat.channelCount,
            layout: options.layout
        )

        let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        return VisionUtilsResult(
            data: layoutData,
            width: finalWidth,
            height: finalHeight,
            channels: options.colorFormat.channelCount,
            colorFormat: options.colorFormat,
            layout: options.layout,
            shape: shape,
            processingTimeMs: processingTimeMs
        )
    }

    // MARK: - ROI Processing

    private static func applyRoi(to image: CGImage, roi: Roi) throws -> CGImage {
        let imageWidth = image.width
        let imageHeight = image.height

        // Validate ROI bounds
        guard roi.x >= 0 && roi.y >= 0 &&
              roi.x + roi.width <= imageWidth &&
              roi.y + roi.height <= imageHeight else {
            throw VisionUtilsError.invalidRoi("ROI extends beyond image bounds")
        }

        let cropRect = CGRect(x: roi.x, y: roi.y, width: roi.width, height: roi.height)

        guard let croppedImage = image.cropping(to: cropRect) else {
            throw VisionUtilsError.invalidRoi("Failed to crop image")
        }

        return croppedImage
    }

    // MARK: - Resize Processing

    private static func applyResize(to image: CGImage, resize: ResizeOptions?) throws -> (CGImage, Int, Int) {
        guard let resize = resize else {
            return (image, image.width, image.height)
        }

        let targetWidth = resize.width
        let targetHeight = resize.height
        let sourceWidth = image.width
        let sourceHeight = image.height

        var drawRect: CGRect
        var canvasSize = CGSize(width: targetWidth, height: targetHeight)

        switch resize.strategy {
        case .stretch:
            drawRect = CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight)

        case .cover:
            let scaleX = CGFloat(targetWidth) / CGFloat(sourceWidth)
            let scaleY = CGFloat(targetHeight) / CGFloat(sourceHeight)
            let scale = max(scaleX, scaleY)

            let scaledWidth = CGFloat(sourceWidth) * scale
            let scaledHeight = CGFloat(sourceHeight) * scale
            let offsetX = (CGFloat(targetWidth) - scaledWidth) / 2
            let offsetY = (CGFloat(targetHeight) - scaledHeight) / 2

            drawRect = CGRect(x: offsetX, y: offsetY, width: scaledWidth, height: scaledHeight)

        case .contain, .letterbox:
            let scaleX = CGFloat(targetWidth) / CGFloat(sourceWidth)
            let scaleY = CGFloat(targetHeight) / CGFloat(sourceHeight)
            let scale = min(scaleX, scaleY)

            let scaledWidth = CGFloat(sourceWidth) * scale
            let scaledHeight = CGFloat(sourceHeight) * scale
            let offsetX = (CGFloat(targetWidth) - scaledWidth) / 2
            let offsetY = (CGFloat(targetHeight) - scaledHeight) / 2

            drawRect = CGRect(x: offsetX, y: offsetY, width: scaledWidth, height: scaledHeight)
        }

        // Create bitmap context
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw VisionUtilsError.processingError("Failed to create graphics context")
        }

        // Fill with pad color for contain/letterbox strategy
        if resize.strategy == .contain || resize.strategy == .letterbox {
            let padColor = resize.padColor
            context.setFillColor(CGColor(
                red: CGFloat(padColor[0]) / 255.0,
                green: CGFloat(padColor.count > 1 ? padColor[1] : padColor[0]) / 255.0,
                blue: CGFloat(padColor.count > 2 ? padColor[2] : padColor[0]) / 255.0,
                alpha: CGFloat(padColor.count > 3 ? padColor[3] : 255) / 255.0
            ))
            context.fill(CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        }

        // Draw image with high-quality interpolation
        context.interpolationQuality = .high
        context.draw(image, in: drawRect)

        guard let resizedImage = context.makeImage() else {
            throw VisionUtilsError.processingError("Failed to create resized image")
        }

        return (resizedImage, targetWidth, targetHeight)
    }

    // MARK: - Pixel Extraction

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

    // MARK: - Color Format Conversion

    private static func convertColorFormat(_ rgbaData: [UInt8], to format: ColorFormat) -> [UInt8] {
        let pixelCount = rgbaData.count / 4

        switch format {
        case .rgba:
            return rgbaData

        case .rgb:
            var result = [UInt8](repeating: 0, count: pixelCount * 3)
            for i in 0..<pixelCount {
                result[i * 3] = rgbaData[i * 4]       // R
                result[i * 3 + 1] = rgbaData[i * 4 + 1] // G
                result[i * 3 + 2] = rgbaData[i * 4 + 2] // B
            }
            return result

        case .bgra:
            var result = [UInt8](repeating: 0, count: pixelCount * 4)
            for i in 0..<pixelCount {
                result[i * 4] = rgbaData[i * 4 + 2]     // B
                result[i * 4 + 1] = rgbaData[i * 4 + 1] // G
                result[i * 4 + 2] = rgbaData[i * 4]     // R
                result[i * 4 + 3] = rgbaData[i * 4 + 3] // A
            }
            return result

        case .bgr:
            var result = [UInt8](repeating: 0, count: pixelCount * 3)
            for i in 0..<pixelCount {
                result[i * 3] = rgbaData[i * 4 + 2]     // B
                result[i * 3 + 1] = rgbaData[i * 4 + 1] // G
                result[i * 3 + 2] = rgbaData[i * 4]     // R
            }
            return result

        case .grayscale:
            var result = [UInt8](repeating: 0, count: pixelCount)
            for i in 0..<pixelCount {
                let r = Float(rgbaData[i * 4])
                let g = Float(rgbaData[i * 4 + 1])
                let b = Float(rgbaData[i * 4 + 2])
                // ITU-R BT.601 luma coefficients
                let gray = 0.299 * r + 0.587 * g + 0.114 * b
                result[i] = UInt8(clamping: Int(gray.rounded()))
            }
            return result

        case .hsv:
            var result = [UInt8](repeating: 0, count: pixelCount * 3)
            for i in 0..<pixelCount {
                let (h, s, v) = rgbToHsv(
                    r: Float(rgbaData[i * 4]) / 255.0,
                    g: Float(rgbaData[i * 4 + 1]) / 255.0,
                    b: Float(rgbaData[i * 4 + 2]) / 255.0
                )
                result[i * 3] = UInt8(h * 255)
                result[i * 3 + 1] = UInt8(s * 255)
                result[i * 3 + 2] = UInt8(v * 255)
            }
            return result

        case .hsl:
            var result = [UInt8](repeating: 0, count: pixelCount * 3)
            for i in 0..<pixelCount {
                let (h, s, l) = rgbToHsl(
                    r: Float(rgbaData[i * 4]) / 255.0,
                    g: Float(rgbaData[i * 4 + 1]) / 255.0,
                    b: Float(rgbaData[i * 4 + 2]) / 255.0
                )
                result[i * 3] = UInt8(h * 255)
                result[i * 3 + 1] = UInt8(s * 255)
                result[i * 3 + 2] = UInt8(l * 255)
            }
            return result

        case .lab:
            var result = [UInt8](repeating: 0, count: pixelCount * 3)
            for i in 0..<pixelCount {
                let (l, a, b) = rgbToLab(
                    r: Float(rgbaData[i * 4]) / 255.0,
                    g: Float(rgbaData[i * 4 + 1]) / 255.0,
                    b: Float(rgbaData[i * 4 + 2]) / 255.0
                )
                // L: 0-100 -> 0-255, a: -128-127 -> 0-255, b: -128-127 -> 0-255
                result[i * 3] = UInt8(clamping: Int((l / 100.0) * 255))
                result[i * 3 + 1] = UInt8(clamping: Int(a + 128))
                result[i * 3 + 2] = UInt8(clamping: Int(b + 128))
            }
            return result

        case .yuv:
            var result = [UInt8](repeating: 0, count: pixelCount * 3)
            for i in 0..<pixelCount {
                let r = Float(rgbaData[i * 4])
                let g = Float(rgbaData[i * 4 + 1])
                let b = Float(rgbaData[i * 4 + 2])
                // BT.601 conversion
                let y = 0.299 * r + 0.587 * g + 0.114 * b
                let u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128
                let v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128
                result[i * 3] = UInt8(clamping: Int(y))
                result[i * 3 + 1] = UInt8(clamping: Int(u))
                result[i * 3 + 2] = UInt8(clamping: Int(v))
            }
            return result

        case .ycbcr:
            var result = [UInt8](repeating: 0, count: pixelCount * 3)
            for i in 0..<pixelCount {
                let r = Float(rgbaData[i * 4])
                let g = Float(rgbaData[i * 4 + 1])
                let b = Float(rgbaData[i * 4 + 2])
                // ITU-R BT.601 YCbCr
                let y = 16 + 65.481 * r / 255 + 128.553 * g / 255 + 24.966 * b / 255
                let cb = 128 - 37.797 * r / 255 - 74.203 * g / 255 + 112 * b / 255
                let cr = 128 + 112 * r / 255 - 93.786 * g / 255 - 18.214 * b / 255
                result[i * 3] = UInt8(clamping: Int(y))
                result[i * 3 + 1] = UInt8(clamping: Int(cb))
                result[i * 3 + 2] = UInt8(clamping: Int(cr))
            }
            return result
        }
    }

    // MARK: - Color Space Helpers

    private static func rgbToHsv(r: Float, g: Float, b: Float) -> (Float, Float, Float) {
        let maxVal = max(r, g, b)
        let minVal = min(r, g, b)
        let delta = maxVal - minVal

        var h: Float = 0
        let s: Float = maxVal == 0 ? 0 : delta / maxVal
        let v: Float = maxVal

        if delta != 0 {
            if maxVal == r {
                h = (g - b) / delta
                if g < b { h += 6 }
            } else if maxVal == g {
                h = 2 + (b - r) / delta
            } else {
                h = 4 + (r - g) / delta
            }
            h /= 6
        }

        return (h, s, v)
    }

    private static func rgbToHsl(r: Float, g: Float, b: Float) -> (Float, Float, Float) {
        let maxVal = max(r, g, b)
        let minVal = min(r, g, b)
        let l = (maxVal + minVal) / 2

        var h: Float = 0
        var s: Float = 0

        if maxVal != minVal {
            let delta = maxVal - minVal
            s = l > 0.5 ? delta / (2 - maxVal - minVal) : delta / (maxVal + minVal)

            if maxVal == r {
                h = (g - b) / delta + (g < b ? 6 : 0)
            } else if maxVal == g {
                h = (b - r) / delta + 2
            } else {
                h = (r - g) / delta + 4
            }
            h /= 6
        }

        return (h, s, l)
    }

    private static func rgbToLab(r: Float, g: Float, b: Float) -> (Float, Float, Float) {
        // RGB to XYZ (sRGB D65)
        var rr = r > 0.04045 ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92
        var gg = g > 0.04045 ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92
        var bb = b > 0.04045 ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92

        let x = rr * 0.4124564 + gg * 0.3575761 + bb * 0.1804375
        let y = rr * 0.2126729 + gg * 0.7151522 + bb * 0.0721750
        let z = rr * 0.0193339 + gg * 0.1191920 + bb * 0.9503041

        // XYZ to Lab (D65 illuminant)
        let xn: Float = 0.95047
        let yn: Float = 1.0
        let zn: Float = 1.08883

        func f(_ t: Float) -> Float {
            return t > 0.008856 ? pow(t, 1/3) : (7.787 * t) + (16/116)
        }

        let fx = f(x / xn)
        let fy = f(y / yn)
        let fz = f(z / zn)

        let labL = (116 * fy) - 16
        let labA = 500 * (fx - fy)
        let labB = 200 * (fy - fz)

        return (labL, labA, labB)
    }

    // MARK: - Normalization

    private static func applyNormalization(_ data: [UInt8], options: GetPixelDataOptions) -> [Float] {
        let normalization = options.normalization
        let channels = options.colorFormat.channelCount

        switch normalization.preset {
        case .raw:
            // No normalization - convert to float as-is
            return data.map { Float($0) }

        case .scale:
            // Simple [0, 1] scaling
            return data.map { Float($0) / 255.0 }

        case .tensorflow:
            // TensorFlow style: [-1, 1] range
            return data.map { (Float($0) / 127.5) - 1.0 }

        case .imagenet:
            // ImageNet normalization
            let mean: [Float] = [0.485, 0.456, 0.406]
            let std: [Float] = [0.229, 0.224, 0.225]

            return applyPerChannelNormalization(data, channels: channels, mean: mean, std: std, scale: 1.0 / 255.0)

        case .custom:
            guard let mean = normalization.mean,
                  let std = normalization.std else {
                return data.map { Float($0) }
            }
            return applyPerChannelNormalization(data, channels: channels, mean: mean, std: std, scale: normalization.scale)
        }
    }

    private static func applyPerChannelNormalization(
        _ data: [UInt8],
        channels: Int,
        mean: [Float],
        std: [Float],
        scale: Float
    ) -> [Float] {
        let pixelCount = data.count / channels
        var result = [Float](repeating: 0, count: data.count)

        for i in 0..<pixelCount {
            for c in 0..<channels {
                let idx = i * channels + c
                let channelMean = c < mean.count ? mean[c] : 0.0
                let channelStd = c < std.count ? std[c] : 1.0

                let scaledValue = Float(data[idx]) * scale
                result[idx] = (scaledValue - channelMean) / channelStd
            }
        }

        return result
    }

    // MARK: - Layout Conversion

    private static func convertLayout(
        _ data: [Float],
        width: Int,
        height: Int,
        channels: Int,
        layout: DataLayout
    ) -> [Float] {
        switch layout {
        case .hwc, .nhwc:
            // Data is already in HWC format
            return data

        case .chw, .nchw:
            // Convert from HWC to CHW
            var result = [Float](repeating: 0, count: data.count)

            for c in 0..<channels {
                for h in 0..<height {
                    for w in 0..<width {
                        let hwcIndex = h * width * channels + w * channels + c
                        let chwIndex = c * height * width + h * width + w
                        result[chwIndex] = data[hwcIndex]
                    }
                }
            }

            return result
        }
    }

    // MARK: - Shape Calculation

    private static func calculateShape(width: Int, height: Int, channels: Int, layout: DataLayout) -> [Int] {
        switch layout {
        case .hwc:
            return [height, width, channels]
        case .chw:
            return [channels, height, width]
        case .nhwc:
            return [1, height, width, channels]
        case .nchw:
            return [1, channels, height, width]
        }
    }
}
