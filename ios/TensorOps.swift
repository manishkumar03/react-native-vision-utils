import Foundation
import UIKit
import CoreGraphics

/// Handles tensor operations and conversions
class TensorOps {

    // MARK: - Extract Channel

    /// Extract a specific channel from pixel data
    static func extractChannel(
        data: [Float],
        width: Int,
        height: Int,
        channels: Int,
        channelIndex: Int,
        dataLayout: String
    ) throws -> [String: Any] {
        guard channelIndex >= 0 && channelIndex < channels else {
            throw VisionUtilsError.processingError("Invalid channel index \(channelIndex)")
        }

        let pixelCount = width * height
        var result = [Float](repeating: 0, count: pixelCount)

        let layout = dataLayout.lowercased()

        if layout == "hwc" || layout == "nhwc" {
            // Data is in HWC format
            for i in 0..<pixelCount {
                result[i] = data[i * channels + channelIndex]
            }
        } else if layout == "chw" || layout == "nchw" {
            // Data is in CHW format
            let channelOffset = channelIndex * pixelCount
            for i in 0..<pixelCount {
                result[i] = data[channelOffset + i]
            }
        }

        return [
            "data": result,
            "width": width,
            "height": height,
            "channels": 1,
            "shape": [height, width]
        ]
    }

    // MARK: - Extract Patch

    /// Extract a rectangular patch from pixel data
    static func extractPatch(
        data: [Float],
        width: Int,
        height: Int,
        channels: Int,
        options: [String: Any],
        dataLayout: String
    ) throws -> [String: Any] {
        guard let x = options["x"] as? Int,
              let y = options["y"] as? Int,
              let patchWidth = options["width"] as? Int,
              let patchHeight = options["height"] as? Int else {
            throw VisionUtilsError.processingError("Patch options must include x, y, width, height")
        }

        guard x >= 0 && y >= 0 && x + patchWidth <= width && y + patchHeight <= height else {
            throw VisionUtilsError.processingError("Patch extends beyond image bounds")
        }

        let layout = dataLayout.lowercased()
        var result = [Float](repeating: 0, count: patchWidth * patchHeight * channels)

        if layout == "hwc" || layout == "nhwc" {
            for py in 0..<patchHeight {
                for px in 0..<patchWidth {
                    for c in 0..<channels {
                        let srcIdx = ((y + py) * width + (x + px)) * channels + c
                        let dstIdx = (py * patchWidth + px) * channels + c
                        result[dstIdx] = data[srcIdx]
                    }
                }
            }
        } else if layout == "chw" || layout == "nchw" {
            for c in 0..<channels {
                let channelOffset = c * width * height
                let dstChannelOffset = c * patchWidth * patchHeight

                for py in 0..<patchHeight {
                    for px in 0..<patchWidth {
                        let srcIdx = channelOffset + (y + py) * width + (x + px)
                        let dstIdx = dstChannelOffset + py * patchWidth + px
                        result[dstIdx] = data[srcIdx]
                    }
                }
            }
        }

        let shape: [Int]
        if layout == "hwc" {
            shape = [patchHeight, patchWidth, channels]
        } else if layout == "chw" {
            shape = [channels, patchHeight, patchWidth]
        } else if layout == "nhwc" {
            shape = [1, patchHeight, patchWidth, channels]
        } else {
            shape = [1, channels, patchHeight, patchWidth]
        }

        return [
            "data": result,
            "width": patchWidth,
            "height": patchHeight,
            "channels": channels,
            "shape": shape
        ]
    }

    // MARK: - Concatenate to Batch

    /// Concatenate multiple results into a batch tensor
    static func concatenateToBatch(results: [[String: Any]]) throws -> [String: Any] {
        guard !results.isEmpty else {
            throw VisionUtilsError.processingError("Cannot concatenate empty results array")
        }

        // Get dimensions from first result - handle both [Float] and [NSNumber] types
        let firstData: [Float]
        if let floatData = results[0]["data"] as? [Float] {
            firstData = floatData
        } else if let nsNumberData = results[0]["data"] as? [NSNumber] {
            firstData = nsNumberData.map { $0.floatValue }
        } else {
            throw VisionUtilsError.processingError("Invalid result format")
        }

        guard let firstWidth = results[0]["width"] as? Int,
              let firstHeight = results[0]["height"] as? Int,
              let firstChannels = results[0]["channels"] as? Int else {
            throw VisionUtilsError.processingError("Invalid result format")
        }

        let singleImageSize = firstWidth * firstHeight * firstChannels
        let batchSize = results.count
        var batchData = [Float](repeating: 0, count: batchSize * singleImageSize)

        for (idx, result) in results.enumerated() {
            let data: [Float]
            if let floatData = result["data"] as? [Float] {
                data = floatData
            } else if let nsNumberData = result["data"] as? [NSNumber] {
                data = nsNumberData.map { $0.floatValue }
            } else {
                throw VisionUtilsError.processingError("Invalid result format at index \(idx)")
            }

            guard let width = result["width"] as? Int,
                  let height = result["height"] as? Int,
                  let channels = result["channels"] as? Int else {
                throw VisionUtilsError.processingError("Invalid result format at index \(idx)")
            }

            guard width == firstWidth && height == firstHeight && channels == firstChannels else {
                throw VisionUtilsError.processingError("All images must have same dimensions")
            }

            let offset = idx * singleImageSize
            for i in 0..<singleImageSize {
                batchData[offset + i] = data[i]
            }
        }

        // Determine output layout from first result
        let layout = (results[0]["dataLayout"] as? String)?.lowercased() ?? "hwc"
        let shape: [Int]

        if layout == "hwc" || layout == "nhwc" {
            shape = [batchSize, firstHeight, firstWidth, firstChannels]
        } else {
            shape = [batchSize, firstChannels, firstHeight, firstWidth]
        }

        return [
            "data": batchData,
            "shape": shape,
            "batchSize": batchSize,
            "width": firstWidth,
            "height": firstHeight,
            "channels": firstChannels
        ]
    }

    // MARK: - Permute

    /// Permute/transpose tensor dimensions
    static func permute(
        data: [Float],
        shape: [Int],
        order: [Int]
    ) throws -> [String: Any] {
        guard shape.count == order.count else {
            throw VisionUtilsError.processingError("Order must have same length as shape")
        }

        guard Set(order) == Set(0..<shape.count) else {
            throw VisionUtilsError.processingError("Order must be a permutation of 0..<\(shape.count)")
        }

        // Calculate strides for input
        var inputStrides = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) {
            inputStrides[i] = inputStrides[i + 1] * shape[i + 1]
        }

        // Calculate new shape and strides
        let newShape = order.map { shape[$0] }
        var outputStrides = [Int](repeating: 1, count: newShape.count)
        for i in stride(from: newShape.count - 2, through: 0, by: -1) {
            outputStrides[i] = outputStrides[i + 1] * newShape[i + 1]
        }

        // Permute data
        let totalSize = shape.reduce(1, *)
        var result = [Float](repeating: 0, count: totalSize)

        for flatIdx in 0..<totalSize {
            // Convert flat index to input indices
            var remaining = flatIdx
            var inputIndices = [Int](repeating: 0, count: shape.count)
            for i in 0..<shape.count {
                inputIndices[i] = remaining / inputStrides[i]
                remaining = remaining % inputStrides[i]
            }

            // Permute indices
            let outputIndices = order.map { inputIndices[$0] }

            // Convert output indices to flat index
            var outputFlatIdx = 0
            for i in 0..<newShape.count {
                outputFlatIdx += outputIndices[i] * outputStrides[i]
            }

            result[outputFlatIdx] = data[flatIdx]
        }

        return [
            "data": result,
            "shape": newShape
        ]
    }
}
