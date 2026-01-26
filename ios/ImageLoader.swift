import Foundation
import UIKit
import Photos

// MARK: - Image Loader

/// Handles loading images from various sources
class ImageLoader {

    /// Load image from source configuration
    static func loadImage(from source: ImageSource) async throws -> UIImage {
        switch source.type {
        case .url:
            return try await loadFromUrl(source.value)
        case .file:
            return try loadFromFile(source.value)
        case .base64:
            return try loadFromBase64(source.value)
        case .asset:
            return try loadFromAsset(source.value)
        case .photoLibrary:
            return try await loadFromPhotoLibrary(source.value)
        case .cgImage:
            // cgImage type is for internal use only - should not be used with ImageLoader
            throw VisionUtilsError.invalidSource("cgImage source type cannot be loaded via ImageLoader")
        }
    }

    // MARK: - URL Loading

    private static func loadFromUrl(_ uri: String) async throws -> UIImage {
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

        return image
    }

    // MARK: - File Loading

    private static func loadFromFile(_ path: String) throws -> UIImage {
        // Handle file:// prefix
        let cleanPath = path.hasPrefix("file://")
            ? String(path.dropFirst(7))
            : path

        guard FileManager.default.fileExists(atPath: cleanPath) else {
            throw VisionUtilsError.fileNotFound("File not found: \(cleanPath)")
        }

        guard let image = UIImage(contentsOfFile: cleanPath) else {
            throw VisionUtilsError.loadError("Failed to decode image file")
        }

        return image
    }

    // MARK: - Base64 Loading

    private static func loadFromBase64(_ data: String) throws -> UIImage {
        // Remove data URL prefix if present
        let cleanBase64: String
        if let commaIndex = data.firstIndex(of: ",") {
            cleanBase64 = String(data[data.index(after: commaIndex)...])
        } else {
            cleanBase64 = data
        }

        guard let imageData = Data(base64Encoded: cleanBase64, options: .ignoreUnknownCharacters) else {
            throw VisionUtilsError.loadError("Invalid base64 encoding")
        }

        guard let image = UIImage(data: imageData) else {
            throw VisionUtilsError.loadError("Failed to decode base64 image data")
        }

        return image
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

    private static func loadFromPhotoLibrary(_ localIdentifier: String) async throws -> UIImage {
        let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil)

        guard let asset = fetchResult.firstObject else {
            throw VisionUtilsError.fileNotFound("Photo not found with identifier: \(localIdentifier)")
        }

        return try await withCheckedThrowingContinuation { continuation in
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
    }
}
