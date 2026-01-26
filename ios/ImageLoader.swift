import Foundation
import UIKit
import Photos

// MARK: - Image Load Result

/// Result of loading an image, includes metadata
struct ImageLoadResult {
    let image: UIImage
    let fileSize: Int?
    let format: String?
}

// MARK: - Image Loader

/// Handles loading images from various sources
class ImageLoader {

    /// Load image from source configuration
    static func loadImage(from source: ImageSource) async throws -> UIImage {
        let result = try await loadImageWithMetadata(from: source)
        return result.image
    }

    /// Load image with metadata (file size, format)
    static func loadImageWithMetadata(from source: ImageSource) async throws -> ImageLoadResult {
        switch source.type {
        case .url:
            return try await loadFromUrlWithMetadata(source.value)
        case .file:
            return try loadFromFileWithMetadata(source.value)
        case .base64:
            return try loadFromBase64WithMetadata(source.value)
        case .asset:
            let image = try loadFromAsset(source.value)
            return ImageLoadResult(image: image, fileSize: nil, format: nil)
        case .photoLibrary:
            return try await loadFromPhotoLibraryWithMetadata(source.value)
        case .cgImage:
            // cgImage type is for internal use only - should not be used with ImageLoader
            throw VisionUtilsError.invalidSource("cgImage source type cannot be loaded via ImageLoader")
        }
    }

    // MARK: - URL Loading

    private static func loadFromUrlWithMetadata(_ uri: String) async throws -> ImageLoadResult {
        guard let url = URL(string: uri) else {
            throw VisionUtilsError.invalidSource("Invalid URL: \(uri)")
        }

        let (data, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw VisionUtilsError.loadError("HTTP request failed")
        }

        guard let image = UIImage(data: data) else {
            throw VisionUtilsError.loadError("Failed to decode image data")
        }

        let format = detectImageFormat(from: data)
        return ImageLoadResult(image: image, fileSize: data.count, format: format)
    }

    // MARK: - File Loading

    private static func loadFromFileWithMetadata(_ path: String) throws -> ImageLoadResult {
        // Handle file:// prefix
        let cleanPath = path.hasPrefix("file://")
            ? String(path.dropFirst(7))
            : path

        guard FileManager.default.fileExists(atPath: cleanPath) else {
            throw VisionUtilsError.fileNotFound("File not found: \(cleanPath)")
        }

        guard let data = FileManager.default.contents(atPath: cleanPath),
              let image = UIImage(data: data) else {
            throw VisionUtilsError.loadError("Failed to decode image file")
        }

        let format = detectImageFormat(from: data)
        return ImageLoadResult(image: image, fileSize: data.count, format: format)
    }

    // MARK: - Base64 Loading

    private static func loadFromBase64WithMetadata(_ data: String) throws -> ImageLoadResult {
        // Remove data URL prefix if present
        var format: String? = nil
        let cleanBase64: String
        if let commaIndex = data.firstIndex(of: ",") {
            let prefix = String(data[..<commaIndex])
            cleanBase64 = String(data[data.index(after: commaIndex)...])
            // Extract format from data URL (e.g., "data:image/png;base64")
            if prefix.contains("image/jpeg") || prefix.contains("image/jpg") {
                format = "jpeg"
            } else if prefix.contains("image/png") {
                format = "png"
            } else if prefix.contains("image/webp") {
                format = "webp"
            } else if prefix.contains("image/gif") {
                format = "gif"
            } else if prefix.contains("image/heic") {
                format = "heic"
            }
        } else {
            cleanBase64 = data
        }

        guard let imageData = Data(base64Encoded: cleanBase64, options: .ignoreUnknownCharacters) else {
            throw VisionUtilsError.loadError("Invalid base64 encoding")
        }

        guard let image = UIImage(data: imageData) else {
            throw VisionUtilsError.loadError("Failed to decode base64 image data")
        }

        // If format not detected from prefix, detect from data
        if format == nil {
            format = detectImageFormat(from: imageData)
        }

        return ImageLoadResult(image: image, fileSize: imageData.count, format: format)
    }

    // MARK: - Asset Loading

    private static func loadFromAsset(_ name: String) throws -> UIImage {
        if let image = UIImage(named: name) {
            return image
        }
        // Try loading from bundle path
        if let image = UIImage(contentsOfFile: name) {
            return image
        }
        throw VisionUtilsError.fileNotFound("Asset not found: \(name)")
    }

    // MARK: - Photo Library Loading

    private static func loadFromPhotoLibraryWithMetadata(_ localIdentifier: String) async throws -> ImageLoadResult {
        let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil)

        guard let asset = fetchResult.firstObject else {
            throw VisionUtilsError.fileNotFound("Photo not found with identifier: \(localIdentifier)")
        }

        // Get file size from PHAsset resource
        let resources = PHAssetResource.assetResources(for: asset)
        var fileSize: Int? = nil
        var format: String? = nil

        if let resource = resources.first {
            if let size = resource.value(forKey: "fileSize") as? Int {
                fileSize = size
            }
            // Detect format from UTI
            let uti = resource.uniformTypeIdentifier
            if uti.contains("jpeg") || uti.contains("jpg") {
                format = "jpeg"
            } else if uti.contains("png") {
                format = "png"
            } else if uti.contains("heic") || uti.contains("heif") {
                format = "heic"
            } else if uti.contains("gif") {
                format = "gif"
            } else if uti.contains("webp") {
                format = "webp"
            }
        }

        let image: UIImage = try await withCheckedThrowingContinuation { continuation in
            let options = PHImageRequestOptions()
            options.version = .current
            options.deliveryMode = .highQualityFormat
            options.isSynchronous = false
            options.isNetworkAccessAllowed = true

            PHImageManager.default().requestImage(
                for: asset,
                targetSize: PHImageManagerMaximumSize,
                contentMode: .default,
                options: options
            ) { image, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: VisionUtilsError.loadError(error.localizedDescription))
                    return
                }

                if let cancelled = info?[PHImageCancelledKey] as? Bool, cancelled {
                    continuation.resume(throwing: VisionUtilsError.loadError("Photo request was cancelled"))
                    return
                }

                guard let resultImage = image else {
                    continuation.resume(throwing: VisionUtilsError.loadError("Failed to load photo"))
                    return
                }

                continuation.resume(returning: resultImage)
            }
        }

        return ImageLoadResult(image: image, fileSize: fileSize, format: format)
    }

    // MARK: - Format Detection

    /// Detect image format from raw data bytes
    private static func detectImageFormat(from data: Data) -> String? {
        guard data.count >= 8 else { return nil }

        let bytes = [UInt8](data.prefix(8))

        // JPEG: FF D8 FF
        if bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
            return "jpeg"
        }

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if bytes[0] == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4E && bytes[3] == 0x47 {
            return "png"
        }

        // GIF: 47 49 46 38
        if bytes[0] == 0x47 && bytes[1] == 0x49 && bytes[2] == 0x46 && bytes[3] == 0x38 {
            return "gif"
        }

        // WebP: 52 49 46 46 ... 57 45 42 50
        if data.count >= 12 {
            let webpBytes = [UInt8](data.prefix(12))
            if webpBytes[0] == 0x52 && webpBytes[1] == 0x49 && webpBytes[2] == 0x46 && webpBytes[3] == 0x46 &&
               webpBytes[8] == 0x57 && webpBytes[9] == 0x45 && webpBytes[10] == 0x42 && webpBytes[11] == 0x50 {
                return "webp"
            }
        }

        // HEIC/HEIF: Check for ftyp box with heic/heif brand
        if data.count >= 12 {
            let ftypBytes = [UInt8](data[4..<12])
            let ftypString = String(bytes: ftypBytes, encoding: .ascii) ?? ""
            if ftypString.contains("heic") || ftypString.contains("heif") || ftypString.contains("mif1") {
                return "heic"
            }
        }

        return nil
    }
}
