import Foundation
import UIKit
import CoreGraphics

/**
 * GridExtractor provides functionality for extracting image patches in a grid pattern.
 *
 * This is useful for:
 * - Sliding window inference on large images
 * - Creating training patches from images
 * - Tiled processing of high-resolution images
 *
 * ## Example Usage (from JS):
 * ```typescript
 * const result = await extractGrid(
 *   { uri: 'file://image.jpg' },
 *   { rows: 4, columns: 4, overlap: 32, includePartial: false },
 *   { outputFormat: 'float32', layout: 'NCHW' }
 * );
 * ```
 */
class GridExtractor {

    // MARK: - Public API

    /**
     * Extract patches from an image in a grid pattern.
     *
     * @param source Source specification (uri, base64, or frame)
     * @param gridOptions Grid extraction options (rows, columns, overlap, includePartial)
     * @param pixelOptions Pixel data output options
     * @returns Dictionary containing patches array and metadata
     * @throws VisionUtilsError on invalid input or processing failure
     */
    static func extractGrid(
        source: [String: Any],
        gridOptions: [String: Any],
        pixelOptions: [String: Any]
    ) async throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()
        // Parse and load the source image
        let imageSource = try ImageSource(from: source)
        let image = try await ImageLoader.loadImage(from: imageSource)

        // Parse grid options
        guard let columns = gridOptions["columns"] as? Int, columns >= 1 else {
            throw VisionUtilsError.invalidInput("columns must be at least 1")
        }
        guard let rows = gridOptions["rows"] as? Int, rows >= 1 else {
            throw VisionUtilsError.invalidInput("rows must be at least 1")
        }

        let overlap = gridOptions["overlap"] as? Int ?? 0
        let overlapPercent = gridOptions["overlapPercent"] as? Double
        let includePartial = gridOptions["includePartial"] as? Bool ?? false

        let imageWidth = Int(image.size.width)
        let imageHeight = Int(image.size.height)

        // Calculate patch size based on grid dimensions and overlap
        let effectiveOverlap: Int
        if let overlapPct = overlapPercent {
            // Calculate overlap based on percent (will apply after calculating base patch size)
            let basePatchWidth = imageWidth / columns
            let basePatchHeight = imageHeight / rows
            effectiveOverlap = Int(Double(min(basePatchWidth, basePatchHeight)) * overlapPct)
        } else {
            effectiveOverlap = overlap
        }

        // Calculate patch dimensions:
        // Total coverage = patchSize * count - overlap * (count - 1)
        // imageSize = patchSize * count - overlap * (count - 1)
        // patchSize = (imageSize + overlap * (count - 1)) / count
        let patchWidth = (imageWidth + effectiveOverlap * (columns - 1)) / columns
        let patchHeight = (imageHeight + effectiveOverlap * (rows - 1)) / rows

        // Validate dimensions
        guard patchWidth > 0, patchHeight > 0 else {
            throw VisionUtilsError.invalidInput("Calculated patch dimensions are invalid")
        }

        // Calculate stride (patch size minus overlap)
        let strideX = max(1, patchWidth - effectiveOverlap)
        let strideY = max(1, patchHeight - effectiveOverlap)

        // Extract patches
        var patches: [[String: Any]] = []

        for row in 0..<rows {
            for col in 0..<columns {
                let x = col * strideX
                let y = row * strideY

                // Check if this patch would be partial
                let isPartialX = (x + patchWidth) > imageWidth
                let isPartialY = (y + patchHeight) > imageHeight

                if (isPartialX || isPartialY) && !includePartial {
                    continue
                }

                // Calculate actual crop dimensions
                let actualWidth = min(patchWidth, imageWidth - x)
                let actualHeight = min(patchHeight, imageHeight - y)

                // Extract the patch
                let cropRect = CGRect(x: x, y: y, width: actualWidth, height: actualHeight)
                if let patchImage = cropImage(image, to: cropRect) {
                    // Handle partial patches by padding if needed
                    let finalImage: UIImage
                    if actualWidth < patchWidth || actualHeight < patchHeight {
                        finalImage = padImage(patchImage, to: CGSize(width: patchWidth, height: patchHeight))
                    } else {
                        finalImage = patchImage
                    }

                    // Get pixel data for this patch
                    let parsedOptions = try GetPixelDataOptions(fromPixelOptions: pixelOptions)
                    let pixelResult = try PixelProcessor.process(image: finalImage, options: parsedOptions)

                    // Build patch info matching GridPatch interface
                    let patchInfo: [String: Any] = [
                        "row": row,
                        "column": col,
                        "x": x,
                        "y": y,
                        "width": actualWidth,
                        "height": actualHeight,
                        "data": pixelResult.data,
                    ]

                    patches.append(patchInfo)
                }
            }
        }

        return [
            "patches": patches,
            "patchCount": patches.count,
            "columns": columns,
            "rows": rows,
            "originalWidth": imageWidth,
            "originalHeight": imageHeight,
            "patchWidth": patchWidth,
            "patchHeight": patchHeight,
            "processingTimeMs": (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        ]
    }

    // MARK: - Private Helpers

    /**
     * Crop a UIImage to the specified rectangle.
     */
    private static func cropImage(_ image: UIImage, to rect: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        // Convert to image coordinates (UIImage origin is top-left)
        let scale = image.scale
        let scaledRect = CGRect(
            x: rect.origin.x * scale,
            y: rect.origin.y * scale,
            width: rect.width * scale,
            height: rect.height * scale
        )

        guard let croppedCGImage = cgImage.cropping(to: scaledRect) else { return nil }
        return UIImage(cgImage: croppedCGImage, scale: scale, orientation: image.imageOrientation)
    }

    /**
     * Pad an image to the target size (adds black padding to right/bottom).
     */
    private static func padImage(_ image: UIImage, to targetSize: CGSize) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { context in
            // Fill with black background
            UIColor.black.setFill()
            context.fill(CGRect(origin: .zero, size: targetSize))

            // Draw the image at top-left
            image.draw(at: .zero)
        }
    }
}
