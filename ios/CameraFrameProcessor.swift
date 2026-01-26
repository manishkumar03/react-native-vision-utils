import Foundation
import Accelerate
import CoreVideo
import CoreImage

/// High-performance camera frame processor for ML preprocessing
class CameraFrameProcessor {

    // MARK: - Singleton
    static let shared = CameraFrameProcessor()
    private init() {}

    // MARK: - Types

    struct FrameSource {
        let width: Int
        let height: Int
        let pixelFormat: String
        let bytesPerRow: Int
        let timestamp: Double?
        let orientation: Int // 0=up, 1=down, 2=left, 3=right
    }

    struct ProcessedFrame {
        let tensorData: [Double]  // Normalized pixel values
        let shape: [Int]          // [height, width, channels]
        let width: Int
        let height: Int
        let processingTimeMs: Double
    }

    // MARK: - Public API

    /// Process camera frame buffer into ML-ready tensor
    func processCameraFrame(
        buffer: UnsafeRawPointer?,
        source: FrameSource,
        options: [String: Any]
    ) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Extract options
        let outputWidth = options["outputWidth"] as? Int ?? source.width
        let outputHeight = options["outputHeight"] as? Int ?? source.height
        let normalize = options["normalize"] as? Bool ?? true
        let outputFormat = options["outputFormat"] as? String ?? "rgb"
        let meanValues = options["mean"] as? [Double] ?? [0.0, 0.0, 0.0]
        let stdValues = options["std"] as? [Double] ?? [1.0, 1.0, 1.0]

        guard let buffer = buffer else {
            throw NSError(domain: "CameraFrameProcessor", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Buffer is nil"
            ])
        }

        // Convert based on pixel format
        var rgbData: [UInt8]

        switch source.pixelFormat.lowercased() {
        case "yuv420", "yuv420f", "420f":
            rgbData = try convertYUV420ToRGB(
                buffer: buffer,
                width: source.width,
                height: source.height,
                bytesPerRow: source.bytesPerRow
            )
        case "nv12":
            rgbData = try convertNV12ToRGB(
                buffer: buffer,
                width: source.width,
                height: source.height,
                bytesPerRow: source.bytesPerRow
            )
        case "nv21":
            rgbData = try convertNV21ToRGB(
                buffer: buffer,
                width: source.width,
                height: source.height,
                bytesPerRow: source.bytesPerRow
            )
        case "bgra":
            rgbData = try convertBGRAToRGB(
                buffer: buffer,
                width: source.width,
                height: source.height,
                bytesPerRow: source.bytesPerRow
            )
        case "rgba":
            rgbData = try convertRGBAToRGB(
                buffer: buffer,
                width: source.width,
                height: source.height,
                bytesPerRow: source.bytesPerRow
            )
        case "rgb":
            rgbData = try extractRGB(
                buffer: buffer,
                width: source.width,
                height: source.height,
                bytesPerRow: source.bytesPerRow
            )
        default:
            throw NSError(domain: "CameraFrameProcessor", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported pixel format: \(source.pixelFormat)"
            ])
        }

        // Apply rotation if needed
        if source.orientation != 0 {
            rgbData = rotateRGB(rgbData, width: source.width, height: source.height, orientation: source.orientation)
        }

        // Get dimensions after rotation
        var currentWidth = source.width
        var currentHeight = source.height
        if source.orientation == 2 || source.orientation == 3 {
            swap(&currentWidth, &currentHeight)
        }

        // Resize if needed
        if currentWidth != outputWidth || currentHeight != outputHeight {
            rgbData = try resizeRGB(rgbData, fromWidth: currentWidth, fromHeight: currentHeight, toWidth: outputWidth, toHeight: outputHeight)
        }

        // Normalize to float tensor
        var tensorData: [Double]
        let channelCount = outputFormat == "grayscale" ? 1 : 3

        if outputFormat == "grayscale" {
            tensorData = rgbToGrayscale(rgbData, width: outputWidth, height: outputHeight)
        } else {
            tensorData = rgbData.map { Double($0) }
        }

        // Apply normalization
        if normalize {
            tensorData = applyNormalization(tensorData, mean: meanValues, std: stdValues, channels: channelCount)
        }

        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

        return [
            "tensor": tensorData,
            "shape": [outputHeight, outputWidth, channelCount],
            "width": outputWidth,
            "height": outputHeight,
            "processingTimeMs": processingTime
        ]
    }

    /// Direct YUV to RGB conversion (for vision-camera frame processor)
    func convertYUVToRGB(options: [String: Any]) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        guard let width = options["width"] as? Int,
              let height = options["height"] as? Int else {
            throw NSError(domain: "CameraFrameProcessor", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Width and height are required"
            ])
        }

        let pixelFormat = options["pixelFormat"] as? String ?? "yuv420"
        let outputFormat = options["outputFormat"] as? String ?? "rgb"

        // Check for base64 input
        if let base64Y = options["yPlaneBase64"] as? String,
           let base64U = options["uPlaneBase64"] as? String,
           let base64V = options["vPlaneBase64"] as? String {

            guard let yData = Data(base64Encoded: base64Y),
                  let uData = Data(base64Encoded: base64U),
                  let vData = Data(base64Encoded: base64V) else {
                throw NSError(domain: "CameraFrameProcessor", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "Invalid base64 data"
                ])
            }

            let rgbData = try convertYUVPlanesToRGB(
                yPlane: Array(yData),
                uPlane: Array(uData),
                vPlane: Array(vData),
                width: width,
                height: height,
                format: pixelFormat
            )

            let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            if outputFormat == "base64" {
                let rgbBase64 = Data(rgbData).base64EncodedString()
                return [
                    "dataBase64": rgbBase64,
                    "width": width,
                    "height": height,
                    "channels": 3,
                    "processingTimeMs": processingTime
                ]
            } else {
                return [
                    "data": rgbData.map { Double($0) },
                    "width": width,
                    "height": height,
                    "channels": 3,
                    "processingTimeMs": processingTime
                ]
            }
        }

        // For pointer-based input (handled by native bridge directly)
        throw NSError(domain: "CameraFrameProcessor", code: 5, userInfo: [
            NSLocalizedDescriptionKey: "Pointer-based input must be handled by native bridge"
        ])
    }

    // MARK: - Private Conversion Methods

    private func convertYUV420ToRGB(buffer: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) throws -> [UInt8] {
        let yPlaneSize = bytesPerRow * height
        let uvPlaneSize = (bytesPerRow / 2) * (height / 2)

        let yPlane = buffer.assumingMemoryBound(to: UInt8.self)
        let uPlane = buffer.advanced(by: yPlaneSize).assumingMemoryBound(to: UInt8.self)
        let vPlane = buffer.advanced(by: yPlaneSize + uvPlaneSize).assumingMemoryBound(to: UInt8.self)

        var rgbData = [UInt8](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            for x in 0..<width {
                let yIndex = y * bytesPerRow + x
                let uvIndex = (y / 2) * (bytesPerRow / 2) + (x / 2)

                let yValue = Int(yPlane[yIndex])
                let uValue = Int(uPlane[uvIndex]) - 128
                let vValue = Int(vPlane[uvIndex]) - 128

                // YUV to RGB conversion (BT.601)
                var r = yValue + (351 * vValue) / 256
                var g = yValue - (86 * uValue + 179 * vValue) / 256
                var b = yValue + (444 * uValue) / 256

                // Clamp values
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                let rgbIndex = (y * width + x) * 3
                rgbData[rgbIndex] = UInt8(r)
                rgbData[rgbIndex + 1] = UInt8(g)
                rgbData[rgbIndex + 2] = UInt8(b)
            }
        }

        return rgbData
    }

    private func convertNV12ToRGB(buffer: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) throws -> [UInt8] {
        let yPlaneSize = bytesPerRow * height

        let yPlane = buffer.assumingMemoryBound(to: UInt8.self)
        let uvPlane = buffer.advanced(by: yPlaneSize).assumingMemoryBound(to: UInt8.self)

        var rgbData = [UInt8](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            for x in 0..<width {
                let yIndex = y * bytesPerRow + x
                let uvIndex = (y / 2) * bytesPerRow + (x / 2) * 2

                let yValue = Int(yPlane[yIndex])
                let uValue = Int(uvPlane[uvIndex]) - 128      // NV12: U comes first
                let vValue = Int(uvPlane[uvIndex + 1]) - 128  // NV12: V comes second

                var r = yValue + (351 * vValue) / 256
                var g = yValue - (86 * uValue + 179 * vValue) / 256
                var b = yValue + (444 * uValue) / 256

                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                let rgbIndex = (y * width + x) * 3
                rgbData[rgbIndex] = UInt8(r)
                rgbData[rgbIndex + 1] = UInt8(g)
                rgbData[rgbIndex + 2] = UInt8(b)
            }
        }

        return rgbData
    }

    private func convertNV21ToRGB(buffer: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) throws -> [UInt8] {
        let yPlaneSize = bytesPerRow * height

        let yPlane = buffer.assumingMemoryBound(to: UInt8.self)
        let vuPlane = buffer.advanced(by: yPlaneSize).assumingMemoryBound(to: UInt8.self)

        var rgbData = [UInt8](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            for x in 0..<width {
                let yIndex = y * bytesPerRow + x
                let vuIndex = (y / 2) * bytesPerRow + (x / 2) * 2

                let yValue = Int(yPlane[yIndex])
                let vValue = Int(vuPlane[vuIndex]) - 128      // NV21: V comes first
                let uValue = Int(vuPlane[vuIndex + 1]) - 128  // NV21: U comes second

                var r = yValue + (351 * vValue) / 256
                var g = yValue - (86 * uValue + 179 * vValue) / 256
                var b = yValue + (444 * uValue) / 256

                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                let rgbIndex = (y * width + x) * 3
                rgbData[rgbIndex] = UInt8(r)
                rgbData[rgbIndex + 1] = UInt8(g)
                rgbData[rgbIndex + 2] = UInt8(b)
            }
        }

        return rgbData
    }

    private func convertBGRAToRGB(buffer: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) throws -> [UInt8] {
        let bgraData = buffer.assumingMemoryBound(to: UInt8.self)
        var rgbData = [UInt8](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            for x in 0..<width {
                let bgraIndex = y * bytesPerRow + x * 4
                let rgbIndex = (y * width + x) * 3

                rgbData[rgbIndex] = bgraData[bgraIndex + 2]     // R
                rgbData[rgbIndex + 1] = bgraData[bgraIndex + 1] // G
                rgbData[rgbIndex + 2] = bgraData[bgraIndex]     // B
            }
        }

        return rgbData
    }

    private func convertRGBAToRGB(buffer: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) throws -> [UInt8] {
        let rgbaData = buffer.assumingMemoryBound(to: UInt8.self)
        var rgbData = [UInt8](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            for x in 0..<width {
                let rgbaIndex = y * bytesPerRow + x * 4
                let rgbIndex = (y * width + x) * 3

                rgbData[rgbIndex] = rgbaData[rgbaIndex]
                rgbData[rgbIndex + 1] = rgbaData[rgbaIndex + 1]
                rgbData[rgbIndex + 2] = rgbaData[rgbaIndex + 2]
            }
        }

        return rgbData
    }

    private func extractRGB(buffer: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) throws -> [UInt8] {
        let srcData = buffer.assumingMemoryBound(to: UInt8.self)
        var rgbData = [UInt8](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            let srcRow = y * bytesPerRow
            let dstRow = y * width * 3
            for x in 0..<(width * 3) {
                rgbData[dstRow + x] = srcData[srcRow + x]
            }
        }

        return rgbData
    }

    private func convertYUVPlanesToRGB(yPlane: [UInt8], uPlane: [UInt8], vPlane: [UInt8], width: Int, height: Int, format: String) throws -> [UInt8] {
        var rgbData = [UInt8](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            for x in 0..<width {
                let yIndex = y * width + x
                let uvIndex = (y / 2) * (width / 2) + (x / 2)

                let yValue = Int(yPlane[yIndex])
                let uValue = Int(uPlane[uvIndex]) - 128
                let vValue = Int(vPlane[uvIndex]) - 128

                var r = yValue + (351 * vValue) / 256
                var g = yValue - (86 * uValue + 179 * vValue) / 256
                var b = yValue + (444 * uValue) / 256

                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                let rgbIndex = (y * width + x) * 3
                rgbData[rgbIndex] = UInt8(r)
                rgbData[rgbIndex + 1] = UInt8(g)
                rgbData[rgbIndex + 2] = UInt8(b)
            }
        }

        return rgbData
    }

    // MARK: - Image Processing Helpers

    private func rotateRGB(_ data: [UInt8], width: Int, height: Int, orientation: Int) -> [UInt8] {
        let newWidth = (orientation == 2 || orientation == 3) ? height : width
        let newHeight = (orientation == 2 || orientation == 3) ? width : height
        var rotated = [UInt8](repeating: 0, count: newWidth * newHeight * 3)

        for y in 0..<height {
            for x in 0..<width {
                let srcIndex = (y * width + x) * 3
                var dstX: Int
                var dstY: Int

                switch orientation {
                case 1: // 180 degrees
                    dstX = width - 1 - x
                    dstY = height - 1 - y
                case 2: // 90 degrees clockwise
                    dstX = height - 1 - y
                    dstY = x
                case 3: // 90 degrees counter-clockwise
                    dstX = y
                    dstY = width - 1 - x
                default:
                    dstX = x
                    dstY = y
                }

                let dstIndex = (dstY * newWidth + dstX) * 3
                rotated[dstIndex] = data[srcIndex]
                rotated[dstIndex + 1] = data[srcIndex + 1]
                rotated[dstIndex + 2] = data[srcIndex + 2]
            }
        }

        return rotated
    }

    private func resizeRGB(_ data: [UInt8], fromWidth: Int, fromHeight: Int, toWidth: Int, toHeight: Int) throws -> [UInt8] {
        var resized = [UInt8](repeating: 0, count: toWidth * toHeight * 3)

        let xScale = Double(fromWidth) / Double(toWidth)
        let yScale = Double(fromHeight) / Double(toHeight)

        for y in 0..<toHeight {
            for x in 0..<toWidth {
                let srcX = min(Int(Double(x) * xScale), fromWidth - 1)
                let srcY = min(Int(Double(y) * yScale), fromHeight - 1)

                let srcIndex = (srcY * fromWidth + srcX) * 3
                let dstIndex = (y * toWidth + x) * 3

                resized[dstIndex] = data[srcIndex]
                resized[dstIndex + 1] = data[srcIndex + 1]
                resized[dstIndex + 2] = data[srcIndex + 2]
            }
        }

        return resized
    }

    private func rgbToGrayscale(_ rgbData: [UInt8], width: Int, height: Int) -> [Double] {
        var grayscale = [Double](repeating: 0, count: width * height)

        for i in 0..<(width * height) {
            let r = Double(rgbData[i * 3])
            let g = Double(rgbData[i * 3 + 1])
            let b = Double(rgbData[i * 3 + 2])

            // Standard grayscale conversion
            grayscale[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        return grayscale
    }

    private func applyNormalization(_ data: [Double], mean: [Double], std: [Double], channels: Int) -> [Double] {
        var normalized = data
        let pixelCount = data.count / channels

        for i in 0..<pixelCount {
            for c in 0..<channels {
                let index = i * channels + c
                let meanVal = c < mean.count ? mean[c] : 0.0
                let stdVal = c < std.count ? std[c] : 1.0

                // Normalize: (value / 255.0 - mean) / std
                normalized[index] = (data[index] / 255.0 - meanVal) / stdVal
            }
        }

        return normalized
    }
}
