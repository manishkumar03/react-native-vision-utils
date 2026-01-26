import AVFoundation
import UIKit
import Accelerate

/// Extracts frames from video files at specific timestamps or intervals.
///
/// Supports three extraction modes:
/// - `timestamps`: Array of specific timestamps in seconds
/// - `interval`: Extract every N seconds (with optional startTime/endTime/maxFrames)
/// - `count`: Extract N evenly-spaced frames
///
/// If no mode is specified, extracts a single frame at t=0.
///
/// **Supported source types:** `file`, `url`, `asset`
///
/// **Output formats:**
/// - `base64` (default): JPEG encoded, quality 0-100 (default 90)
/// - `pixelData`: Raw pixel arrays with optional colorFormat/normalization
///
/// **Note:** `colorFormat` and `normalization` only apply when `outputFormat === "pixelData"`.
///
/// Per-frame extraction errors are captured in the frame's `error` field; extraction continues for remaining frames.
class VideoFrameExtractor {

    /// Extract frames from a video file
    /// - Parameters:
    ///   - source: Video source with `type` (file/url/asset) and `value` (path or URL string)
    ///   - options: Extraction options:
    ///     - `timestamps`: [Double] - specific timestamps in seconds
    ///     - `interval`: Double - extract every N seconds
    ///     - `count`: Int - number of evenly-spaced frames
    ///     - `startTime`/`endTime`: Double - range for interval/count modes
    ///     - `maxFrames`: Int - limit for interval mode (default 100)
    ///     - `resize`: {width: Int, height: Int} - resize frames
    ///     - `outputFormat`: "base64" (default) or "pixelData"
    ///     - `quality`: Int 0-100 (default 90) - JPEG quality for base64
    ///     - `colorFormat`: String - for pixelData (rgb/rgba/bgr/grayscale)
    ///     - `normalization`: {preset: String} - for pixelData (scale/imagenet/tensorflow)
    /// - Returns: Dictionary with `frames`, `frameCount`, `videoDuration`, `videoWidth`, `videoHeight`, `frameRate`, `processingTimeMs`
    static func extractFrames(source: [String: Any], options: [String: Any]) async throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Get video URL
        guard let videoURL = try getVideoURL(from: source) else {
            throw NSError(domain: "VideoFrameExtractor", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid video source"])
        }

        // Create asset
        let asset = AVURLAsset(url: videoURL)

        // Check if asset is playable
        let isPlayable = try await asset.load(.isPlayable)
        guard isPlayable else {
            throw NSError(domain: "VideoFrameExtractor", code: 2, userInfo: [NSLocalizedDescriptionKey: "Video file is not playable"])
        }

        // Get video duration
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)

        // Determine timestamps to extract
        let timestamps = try getTimestamps(options: options, duration: durationSeconds)

        if timestamps.isEmpty {
            throw NSError(domain: "VideoFrameExtractor", code: 3, userInfo: [NSLocalizedDescriptionKey: "No timestamps to extract"])
        }

        // Create image generator
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.1, preferredTimescale: 600)

        // Set max size if resize specified
        if let resize = options["resize"] as? [String: Any],
           let width = resize["width"] as? Int,
           let height = resize["height"] as? Int {
            generator.maximumSize = CGSize(width: width, height: height)
        }

        // Extract frames
        var frames: [[String: Any]] = []
        let outputFormat = options["outputFormat"] as? String ?? "base64"
        let quality = options["quality"] as? Int ?? 90

        for timestamp in timestamps {
            let time = CMTime(seconds: timestamp, preferredTimescale: 600)

            do {
                var actualTime = CMTime.zero
                let cgImage = try generator.copyCGImage(at: time, actualTime: &actualTime)
                let uiImage = UIImage(cgImage: cgImage)

                var frameData: [String: Any] = [
                    "timestamp": CMTimeGetSeconds(actualTime),
                    "requestedTimestamp": timestamp,
                    "width": cgImage.width,
                    "height": cgImage.height
                ]

                if outputFormat == "base64" {
                    if let jpegData = uiImage.jpegData(compressionQuality: CGFloat(quality) / 100.0) {
                        frameData["base64"] = jpegData.base64EncodedString()
                    }
                } else if outputFormat == "pixelData" {
                    // Convert to pixel array if needed
                    let pixelData = try extractPixelData(from: cgImage, options: options)
                    frameData["data"] = pixelData.data
                    frameData["channels"] = pixelData.channels
                }

                frames.append(frameData)
            } catch {
                // Add error frame
                frames.append([
                    "timestamp": timestamp,
                    "requestedTimestamp": timestamp,
                    "error": error.localizedDescription
                ])
            }
        }

        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        // Get video metadata
        let tracks = try await asset.load(.tracks)
        var videoWidth: Int = 0
        var videoHeight: Int = 0
        var frameRate: Float = 0

        if let videoTrack = tracks.first(where: { $0.mediaType == .video }) {
            let size = try await videoTrack.load(.naturalSize)
            let transform = try await videoTrack.load(.preferredTransform)
            let transformedSize = size.applying(transform)
            videoWidth = Int(abs(transformedSize.width))
            videoHeight = Int(abs(transformedSize.height))
            frameRate = try await videoTrack.load(.nominalFrameRate)
        }

        return [
            "frames": frames,
            "frameCount": frames.count,
            "videoDuration": durationSeconds,
            "videoWidth": videoWidth,
            "videoHeight": videoHeight,
            "frameRate": frameRate,
            "processingTimeMs": processingTime
        ]
    }

    /// Get video URL from source
    private static func getVideoURL(from source: [String: Any]) throws -> URL? {
        guard let type = source["type"] as? String,
              let value = source["value"] as? String else {
            return nil
        }

        switch type {
        case "file":
            let url = URL(fileURLWithPath: value)
            guard FileManager.default.fileExists(atPath: value) else {
                throw NSError(domain: "VideoFrameExtractor", code: 4, userInfo: [NSLocalizedDescriptionKey: "Video file not found: \(value)"])
            }
            return url

        case "url":
            return URL(string: value)

        case "asset":
            // Handle asset library URLs
            if let url = URL(string: value) {
                return url
            }
            return nil

        default:
            throw NSError(domain: "VideoFrameExtractor", code: 5, userInfo: [NSLocalizedDescriptionKey: "Unsupported video source type: \(type)"])
        }
    }

    /// Calculate timestamps to extract based on options
    private static func getTimestamps(options: [String: Any], duration: Double) throws -> [Double] {
        var timestamps: [Double] = []

        // Option 1: Explicit timestamps
        if let explicitTimestamps = options["timestamps"] as? [Double] {
            timestamps = explicitTimestamps.filter { $0 >= 0 && $0 <= duration }
        }
        // Option 2: Interval-based extraction
        else if let interval = options["interval"] as? Double, interval > 0 {
            let startTime = options["startTime"] as? Double ?? 0
            let endTime = options["endTime"] as? Double ?? duration
            let maxFrames = options["maxFrames"] as? Int ?? 100

            var currentTime = max(0, startTime)
            let effectiveEndTime = min(endTime, duration)

            while currentTime <= effectiveEndTime && timestamps.count < maxFrames {
                timestamps.append(currentTime)
                currentTime += interval
            }
        }
        // Option 3: Count-based extraction (evenly spaced)
        else if let count = options["count"] as? Int, count > 0 {
            let startTime = options["startTime"] as? Double ?? 0
            let endTime = options["endTime"] as? Double ?? duration
            let effectiveEndTime = min(endTime, duration)
            let effectiveStartTime = max(0, startTime)

            if count == 1 {
                timestamps = [(effectiveStartTime + effectiveEndTime) / 2]
            } else {
                let interval = (effectiveEndTime - effectiveStartTime) / Double(count - 1)
                for i in 0..<count {
                    timestamps.append(effectiveStartTime + Double(i) * interval)
                }
            }
        }
        // Default: Single frame at 0
        else {
            timestamps = [0]
        }

        return timestamps
    }

    /// Extract pixel data from CGImage
    private static func extractPixelData(from cgImage: CGImage, options: [String: Any]) throws -> (data: [Float], channels: Int) {
        let width = cgImage.width
        let height = cgImage.height
        let colorFormat = options["colorFormat"] as? String ?? "rgb"
        let channels = colorFormat == "grayscale" ? 1 : (colorFormat.contains("a") ? 4 : 3)

        // Create bitmap context
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw NSError(domain: "VideoFrameExtractor", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to create bitmap context"])
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert to float array based on color format
        var result: [Float] = []
        let pixelCount = width * height

        // Get normalization
        let normalization = options["normalization"] as? [String: Any]
        let preset = normalization?["preset"] as? String ?? "scale"

        var mean: [Float] = [0, 0, 0]
        var std: [Float] = [1, 1, 1]
        var scale: Float = 1.0 / 255.0

        if preset == "imagenet" {
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        } else if preset == "tensorflow" {
            scale = 2.0 / 255.0
            mean = [1, 1, 1]
        }

        result.reserveCapacity(pixelCount * channels)

        for i in 0..<pixelCount {
            let offset = i * 4

            if colorFormat == "grayscale" {
                let r = Float(pixelData[offset]) / 255.0
                let g = Float(pixelData[offset + 1]) / 255.0
                let b = Float(pixelData[offset + 2]) / 255.0
                let gray = 0.299 * r + 0.587 * g + 0.114 * b
                result.append(gray)
            } else {
                let r = (Float(pixelData[offset]) * scale - mean[0]) / std[0]
                let g = (Float(pixelData[offset + 1]) * scale - mean[1]) / std[1]
                let b = (Float(pixelData[offset + 2]) * scale - mean[2]) / std[2]

                if colorFormat.starts(with: "bgr") {
                    result.append(b)
                    result.append(g)
                    result.append(r)
                } else {
                    result.append(r)
                    result.append(g)
                    result.append(b)
                }

                if channels == 4 {
                    result.append(Float(pixelData[offset + 3]) / 255.0)
                }
            }
        }

        return (result, channels)
    }
}
