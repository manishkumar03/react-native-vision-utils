import Foundation
import UIKit
import Photos
import Accelerate

// MARK: - Option Structs (mirrored from TypeScript)

/// Image source types
enum ImageSourceType: String {
    case url
    case file
    case base64
    case asset
    case photoLibrary
}

/// Color format options
enum ColorFormat: String {
    case rgb
    case rgba
    case bgr
    case bgra
    case grayscale
    case hsv
    case hsl
    case lab
    case yuv
    case ycbcr

    var channelCount: Int {
        switch self {
        case .rgb, .bgr, .hsv, .hsl, .lab, .yuv, .ycbcr: return 3
        case .rgba, .bgra: return 4
        case .grayscale: return 1
        }
    }
}

/// Resize strategy options
enum ResizeStrategy: String {
    case cover
    case contain
    case stretch
    case letterbox
}

/// Data layout options
enum DataLayout: String {
    case hwc
    case chw
    case nhwc
    case nchw

    var hasBatchDimension: Bool {
        return self == .nhwc || self == .nchw
    }

    var isChannelFirst: Bool {
        return self == .chw || self == .nchw
    }
}

/// Normalization preset types
enum NormalizationPreset: String {
    case imagenet
    case tensorflow
    case scale
    case raw
    case custom
}

/// Image source configuration
struct ImageSource {
    let type: ImageSourceType
    let value: String

    init(from dict: [String: Any]) throws {
        guard let typeStr = dict["type"] as? String,
              let type = ImageSourceType(rawValue: typeStr) else {
            throw VisionUtilsError.invalidSource("Invalid or missing source type")
        }

        guard let value = dict["value"] as? String else {
            throw VisionUtilsError.invalidSource("Missing source value")
        }

        self.type = type
        self.value = value
    }
}

/// Resize options
struct ResizeOptions {
    let width: Int
    let height: Int
    let strategy: ResizeStrategy
    let padColor: [UInt8]

    init(from dict: [String: Any]) throws {
        guard let width = dict["width"] as? Int,
              let height = dict["height"] as? Int else {
            throw VisionUtilsError.invalidResize("Width and height are required")
        }

        self.width = width
        self.height = height

        if let strategyStr = dict["strategy"] as? String,
           let strategy = ResizeStrategy(rawValue: strategyStr) {
            self.strategy = strategy
        } else {
            self.strategy = .cover
        }

        if let padColorArr = dict["padColor"] as? [Int] {
            self.padColor = padColorArr.map { UInt8(clamping: $0) }
        } else {
            self.padColor = [0, 0, 0, 255]
        }
    }
}

/// Region of interest
struct Roi {
    let x: Int
    let y: Int
    let width: Int
    let height: Int

    init(from dict: [String: Any]) throws {
        guard let x = dict["x"] as? Int,
              let y = dict["y"] as? Int,
              let width = dict["width"] as? Int,
              let height = dict["height"] as? Int else {
            throw VisionUtilsError.invalidRoi("ROI requires x, y, width, and height")
        }

        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

/// Normalization configuration
struct Normalization {
    let preset: NormalizationPreset
    let mean: [Float]?
    let std: [Float]?
    let scale: Float

    init(from dict: [String: Any]) throws {
        guard let presetStr = dict["preset"] as? String,
              let preset = NormalizationPreset(rawValue: presetStr) else {
            throw VisionUtilsError.invalidNormalization("Invalid normalization preset")
        }

        self.preset = preset

        if preset == .custom {
            guard let mean = dict["mean"] as? [Double],
                  let std = dict["std"] as? [Double] else {
                throw VisionUtilsError.invalidNormalization("Custom normalization requires mean and std arrays")
            }
            self.mean = mean.map { Float($0) }
            self.std = std.map { Float($0) }
        } else {
            self.mean = nil
            self.std = nil
        }

        if let scale = dict["scale"] as? Double {
            self.scale = Float(scale)
        } else {
            self.scale = 1.0 / 255.0
        }
    }

    static let `default` = Normalization(preset: .raw, mean: nil, std: nil, scale: 1.0)

    private init(preset: NormalizationPreset, mean: [Float]?, std: [Float]?, scale: Float) {
        self.preset = preset
        self.mean = mean
        self.std = std
        self.scale = scale
    }
}

/// Complete options for getPixelData
struct GetPixelDataOptions {
    let source: ImageSource
    let colorFormat: ColorFormat
    let normalization: Normalization
    let resize: ResizeOptions?
    let roi: Roi?
    let layout: DataLayout

    init(from dict: [String: Any]) throws {
        guard let sourceDict = dict["source"] as? [String: Any] else {
            throw VisionUtilsError.invalidSource("Source is required")
        }

        self.source = try ImageSource(from: sourceDict)

        if let colorFormatStr = dict["colorFormat"] as? String,
           let colorFormat = ColorFormat(rawValue: colorFormatStr) {
            self.colorFormat = colorFormat
        } else {
            self.colorFormat = .rgb
        }

        if let normDict = dict["normalization"] as? [String: Any] {
            self.normalization = try Normalization(from: normDict)
        } else {
            self.normalization = Normalization.default
        }

        if let resizeDict = dict["resize"] as? [String: Any] {
            self.resize = try ResizeOptions(from: resizeDict)
        } else {
            self.resize = nil
        }

        if let roiDict = dict["roi"] as? [String: Any] {
            self.roi = try Roi(from: roiDict)
        } else {
            self.roi = nil
        }

        if let layoutStr = dict["dataLayout"] as? String,
           let layout = DataLayout(rawValue: layoutStr.lowercased()) {
            self.layout = layout
        } else {
            self.layout = .hwc
        }
    }
}

// MARK: - Error Types

enum VisionUtilsError: Error {
    case invalidSource(String)
    case loadError(String)
    case fileNotFound(String)
    case permissionDenied(String)
    case processingError(String)
    case invalidRoi(String)
    case invalidResize(String)
    case invalidNormalization(String)
    case unknown(String)

    var code: String {
        switch self {
        case .invalidSource: return "INVALID_SOURCE"
        case .loadError: return "LOAD_ERROR"
        case .fileNotFound: return "FILE_NOT_FOUND"
        case .permissionDenied: return "PERMISSION_DENIED"
        case .processingError: return "PROCESSING_ERROR"
        case .invalidRoi: return "INVALID_ROI"
        case .invalidResize: return "INVALID_RESIZE"
        case .invalidNormalization: return "INVALID_NORMALIZATION"
        case .unknown: return "UNKNOWN"
        }
    }

    var message: String {
        switch self {
        case .invalidSource(let msg),
             .loadError(let msg),
             .fileNotFound(let msg),
             .permissionDenied(let msg),
             .processingError(let msg),
             .invalidRoi(let msg),
             .invalidResize(let msg),
             .invalidNormalization(let msg),
             .unknown(let msg):
            return msg
        }
    }
}

// MARK: - Result Type

struct VisionUtilsResult {
    let data: [Float]
    let width: Int
    let height: Int
    let channels: Int
    let colorFormat: ColorFormat
    let layout: DataLayout
    let shape: [Int]
    let processingTimeMs: Double

    func toDictionary() -> [String: Any] {
        return [
            "data": data,
            "width": width,
            "height": height,
            "channels": channels,
            "colorFormat": colorFormat.rawValue,
            "dataLayout": layout.rawValue.uppercased(),
            "shape": shape,
            "processingTimeMs": processingTimeMs
        ]
    }
}
