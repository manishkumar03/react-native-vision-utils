import Foundation
import UIKit
import CoreGraphics
import CoreText

/// Utility class for drawing visualizations on images
@objc(DrawingUtils)
public class DrawingUtils: NSObject {

    // MARK: - Color Palette for Classes

    /// Default color palette for up to 80 classes (COCO-style)
    private static let defaultColors: [[Int]] = [
        [255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29],
        [207, 210, 49], [72, 249, 10], [146, 204, 23], [61, 219, 134],
        [26, 147, 52], [0, 212, 187], [44, 153, 168], [0, 194, 255],
        [52, 69, 147], [100, 115, 255], [0, 24, 236], [132, 56, 255],
        [82, 0, 133], [203, 56, 255], [255, 149, 200], [255, 55, 199],
        [255, 99, 99], [255, 173, 173], [255, 155, 85], [255, 198, 94],
        [224, 226, 117], [135, 252, 82], [175, 221, 98], [121, 232, 168],
        [91, 174, 115], [64, 225, 208], [102, 179, 192], [64, 210, 255],
        [103, 121, 176], [144, 156, 255], [64, 84, 244], [166, 114, 255],
        [132, 64, 170], [219, 114, 255], [255, 177, 217], [255, 113, 214],
        [255, 128, 128], [255, 189, 189], [255, 177, 127], [255, 212, 138],
        [235, 237, 156], [168, 253, 125], [196, 232, 146], [158, 239, 191],
        [136, 197, 159], [114, 235, 223], [145, 201, 210], [114, 223, 255],
        [146, 162, 197], [175, 187, 255], [114, 133, 247], [189, 157, 255],
        [166, 114, 192], [231, 157, 255], [255, 199, 230], [255, 156, 227],
        [255, 153, 153], [255, 203, 203], [255, 194, 163], [255, 223, 173],
        [242, 243, 185], [190, 254, 162], [212, 240, 180], [185, 244, 211],
        [170, 214, 189], [153, 241, 233], [178, 217, 224], [153, 232, 255],
        [178, 190, 213], [199, 207, 255], [153, 169, 249], [208, 187, 255],
        [189, 153, 206], [239, 187, 255], [255, 215, 238], [255, 187, 235]
    ]

    /// Get color for a class index
    private static func getColor(forClass classIndex: Int) -> [Int] {
        return defaultColors[classIndex % defaultColors.count]
    }

    // MARK: - Draw Bounding Boxes

    /// Draw bounding boxes on an image
    @objc
    public static func drawBoxes(
        image: UIImage,
        boxes: [[String: Any]],
        lineWidth: CGFloat,
        fontSize: CGFloat,
        drawLabels: Bool,
        labelBackgroundAlpha: CGFloat,
        labelColor: [Int],
        defaultColor: [Int]?,
        quality: Int
    ) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        let size = image.size
        let width = Int(size.width)
        let height = Int(size.height)

        // Use UIGraphicsImageRenderer for proper coordinate handling
        let renderer = UIGraphicsImageRenderer(size: size)
        var boxesDrawn = 0

        let outputImage = renderer.image { ctx in
            // Draw original image
            image.draw(at: .zero)

            let context = ctx.cgContext

            for boxInfo in boxes {
                guard let boxArray = boxInfo["box"] as? [Double], boxArray.count >= 4 else {
                    continue
                }

                let x1 = CGFloat(boxArray[0])
                let y1 = CGFloat(boxArray[1])
                let x2 = CGFloat(boxArray[2])
                let y2 = CGFloat(boxArray[3])

                // Get color
                var color: [Int]
                if let customColor = boxInfo["color"] as? [Int], customColor.count >= 3 {
                    color = customColor
                } else if let classIndex = boxInfo["classIndex"] as? Int {
                    color = getColor(forClass: classIndex)
                } else if let defaultCol = defaultColor, defaultCol.count >= 3 {
                    color = defaultCol
                } else {
                    color = [255, 0, 0] // Default red
                }

                let r = CGFloat(color[0]) / 255.0
                let g = CGFloat(color[1]) / 255.0
                let b = CGFloat(color[2]) / 255.0

                // Draw box
                context.setStrokeColor(red: r, green: g, blue: b, alpha: 1.0)
                context.setLineWidth(lineWidth)
                let rect = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
                context.stroke(rect)

                // Draw label if provided
                if drawLabels {
                    var labelText = ""
                    if let label = boxInfo["label"] as? String {
                        labelText = label
                    }
                    if let score = boxInfo["score"] as? Double {
                        let scoreStr = String(format: "%.2f", score)
                        labelText = labelText.isEmpty ? scoreStr : "\(labelText) \(scoreStr)"
                    }

                    if !labelText.isEmpty {
                        // Draw label background
                        let font = UIFont.systemFont(ofSize: fontSize, weight: .semibold)
                        let textAttributes: [NSAttributedString.Key: Any] = [
                            .font: font,
                            .foregroundColor: UIColor(
                                red: CGFloat(labelColor.count > 0 ? labelColor[0] : 255) / 255.0,
                                green: CGFloat(labelColor.count > 1 ? labelColor[1] : 255) / 255.0,
                                blue: CGFloat(labelColor.count > 2 ? labelColor[2] : 255) / 255.0,
                                alpha: 1.0
                            )
                        ]

                        let textSize = (labelText as NSString).size(withAttributes: textAttributes)
                        let labelRect = CGRect(
                            x: x1,
                            y: max(0, y1 - textSize.height - 4),
                            width: textSize.width + 8,
                            height: textSize.height + 4
                        )

                        // Background
                        context.setFillColor(red: r, green: g, blue: b, alpha: labelBackgroundAlpha)
                        context.fill(labelRect)

                        // Text
                        let textPoint = CGPoint(x: x1 + 4, y: max(0, y1 - textSize.height - 2))
                        (labelText as NSString).draw(at: textPoint, withAttributes: textAttributes)
                    }
                }

                boxesDrawn += 1
            }
        }

        // Convert to base64
        let compressionQuality = CGFloat(quality) / 100.0
        guard let imageData = outputImage.jpegData(compressionQuality: compressionQuality) else {
            throw VisionUtilsError.processingFailed("Failed to encode image")
        }
        let base64String = imageData.base64EncodedString()

        let endTime = CFAbsoluteTimeGetCurrent()
        let processingTimeMs = (endTime - startTime) * 1000

        return [
            "imageBase64": base64String,
            "width": width,
            "height": height,
            "boxesDrawn": boxesDrawn,
            "processingTimeMs": processingTimeMs
        ]
    }

    // MARK: - Draw Keypoints

    /// Draw keypoints and skeleton on an image
    @objc
    public static func drawKeypoints(
        image: UIImage,
        keypoints: [[String: Any]],
        pointRadius: CGFloat,
        pointColors: [[Int]]?,
        skeleton: [[String: Any]]?,
        lineWidth: CGFloat,
        minConfidence: CGFloat,
        quality: Int
    ) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        let size = image.size
        let width = Int(size.width)
        let height = Int(size.height)

        // Use UIGraphicsImageRenderer for proper coordinate handling
        let renderer = UIGraphicsImageRenderer(size: size)
        var pointsDrawn = 0
        var connectionsDrawn = 0

        let outputImage = renderer.image { ctx in
            // Draw original image
            image.draw(at: .zero)

            let context = ctx.cgContext

            // Draw skeleton connections first (behind points)
            if let skeletonConnections = skeleton {
                for connection in skeletonConnections {
                    guard let fromIdx = connection["from"] as? Int,
                          let toIdx = connection["to"] as? Int,
                          fromIdx < keypoints.count,
                          toIdx < keypoints.count else {
                        continue
                    }

                    let fromPoint = keypoints[fromIdx]
                    let toPoint = keypoints[toIdx]

                    guard let fromX = fromPoint["x"] as? Double,
                          let fromY = fromPoint["y"] as? Double,
                          let toX = toPoint["x"] as? Double,
                          let toY = toPoint["y"] as? Double else {
                        continue
                    }

                    let fromConf = fromPoint["confidence"] as? Double ?? 1.0
                    let toConf = toPoint["confidence"] as? Double ?? 1.0

                    if CGFloat(fromConf) < minConfidence || CGFloat(toConf) < minConfidence {
                        continue
                    }

                    // Get connection color
                    let color: [Int]
                    if let connColor = connection["color"] as? [Int], connColor.count >= 3 {
                        color = connColor
                    } else {
                        color = [0, 255, 0] // Default green
                    }

                    context.setStrokeColor(
                        red: CGFloat(color[0]) / 255.0,
                        green: CGFloat(color[1]) / 255.0,
                        blue: CGFloat(color[2]) / 255.0,
                        alpha: 1.0
                    )
                    context.setLineWidth(lineWidth)
                    context.move(to: CGPoint(x: fromX, y: fromY))
                    context.addLine(to: CGPoint(x: toX, y: toY))
                    context.strokePath()

                    connectionsDrawn += 1
                }
            }

            // Draw keypoints
            for (idx, point) in keypoints.enumerated() {
                guard let x = point["x"] as? Double,
                      let y = point["y"] as? Double else {
                    continue
                }

                let confidence = point["confidence"] as? Double ?? 1.0
                if CGFloat(confidence) < minConfidence {
                    continue
                }

                // Get point color
                let color: [Int]
                if let colors = pointColors, idx < colors.count, colors[idx].count >= 3 {
                    color = colors[idx]
                } else {
                    color = getColor(forClass: idx)
                }

                context.setFillColor(
                    red: CGFloat(color[0]) / 255.0,
                    green: CGFloat(color[1]) / 255.0,
                    blue: CGFloat(color[2]) / 255.0,
                    alpha: 1.0
                )

                let pointRect = CGRect(
                    x: CGFloat(x) - pointRadius,
                    y: CGFloat(y) - pointRadius,
                    width: pointRadius * 2,
                    height: pointRadius * 2
                )
                context.fillEllipse(in: pointRect)

                pointsDrawn += 1
            }
        }

        // Convert to base64
        let compressionQuality = CGFloat(quality) / 100.0
        guard let imageData = outputImage.jpegData(compressionQuality: compressionQuality) else {
            throw VisionUtilsError.processingFailed("Failed to encode image")
        }
        let base64String = imageData.base64EncodedString()

        let endTime = CFAbsoluteTimeGetCurrent()

        return [
            "imageBase64": base64String,
            "width": width,
            "height": height,
            "pointsDrawn": pointsDrawn,
            "connectionsDrawn": connectionsDrawn,
            "processingTimeMs": (endTime - startTime) * 1000
        ]
    }

    // MARK: - Overlay Mask

    /// Overlay a segmentation mask on an image
    @objc
    public static func overlayMask(
        image: UIImage,
        mask: [Int],
        maskWidth: Int,
        maskHeight: Int,
        alpha: CGFloat,
        colorMap: [[Int]]?,
        singleColor: [Int]?,
        isClassMask: Bool,
        quality: Int
    ) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        let size = image.size
        let width = Int(size.width)
        let height = Int(size.height)

        guard mask.count == maskWidth * maskHeight else {
            throw VisionUtilsError.invalidInput("Mask size doesn't match dimensions")
        }

        // Create mask image
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var maskPixels = [UInt8](repeating: 0, count: maskWidth * maskHeight * 4)

        for y in 0..<maskHeight {
            for x in 0..<maskWidth {
                let idx = y * maskWidth + x
                let pixelIdx = idx * 4
                let value = mask[idx]

                if isClassMask {
                    // Class index - use color map
                    if value > 0 { // Skip background (0)
                        let color: [Int]
                        if let map = colorMap, value < map.count {
                            color = map[value]
                        } else if let single = singleColor, single.count >= 3 {
                            color = single
                        } else {
                            color = getColor(forClass: value)
                        }

                        maskPixels[pixelIdx] = UInt8(color[0])
                        maskPixels[pixelIdx + 1] = UInt8(color[1])
                        maskPixels[pixelIdx + 2] = UInt8(color[2])
                        maskPixels[pixelIdx + 3] = UInt8(alpha * 255)
                    }
                } else {
                    // Binary mask or probability
                    if value > 0 {
                        let color = singleColor ?? [0, 255, 0]
                        maskPixels[pixelIdx] = UInt8(color[0])
                        maskPixels[pixelIdx + 1] = UInt8(color[1])
                        maskPixels[pixelIdx + 2] = UInt8(color[2])
                        maskPixels[pixelIdx + 3] = UInt8(alpha * 255 * CGFloat(min(value, 255)) / 255.0)
                    }
                }
            }
        }

        // Create mask CGImage
        guard let maskDataProvider = CGDataProvider(data: Data(maskPixels) as CFData),
              let maskCGImage = CGImage(
                width: maskWidth,
                height: maskHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: maskWidth * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: maskDataProvider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            throw VisionUtilsError.processingFailed("Failed to create mask image")
        }

        // Use UIGraphicsImageRenderer for proper coordinate handling
        let renderer = UIGraphicsImageRenderer(size: size)
        let outputImage = renderer.image { ctx in
            // Draw original image
            image.draw(at: .zero)

            // Draw mask scaled to image size
            let maskUIImage = UIImage(cgImage: maskCGImage)
            maskUIImage.draw(in: CGRect(x: 0, y: 0, width: width, height: height), blendMode: .normal, alpha: 1.0)
        }

        // Convert to base64
        guard let imageData = outputImage.jpegData(compressionQuality: CGFloat(quality) / 100.0) else {
            throw VisionUtilsError.processingFailed("Failed to encode image")
        }

        let endTime = CFAbsoluteTimeGetCurrent()

        return [
            "imageBase64": imageData.base64EncodedString(),
            "width": width,
            "height": height,
            "processingTimeMs": (endTime - startTime) * 1000
        ]
    }

    // MARK: - Overlay Heatmap

    /// Overlay a heatmap on an image (for attention/CAM visualization)
    public static func overlayHeatmap(
        image: UIImage,
        heatmap: [Double],
        heatmapWidth: Int,
        heatmapHeight: Int,
        alpha: CGFloat,
        colorScheme: String,
        minValue: Double?,
        maxValue: Double?,
        quality: Int
    ) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()

        let size = image.size
        let width = Int(size.width)
        let height = Int(size.height)

        guard heatmap.count == heatmapWidth * heatmapHeight else {
            throw VisionUtilsError.invalidInput("Heatmap size doesn't match dimensions")
        }

        // Normalize heatmap
        let minVal = minValue ?? heatmap.min() ?? 0
        let maxVal = maxValue ?? heatmap.max() ?? 1
        let range = maxVal - minVal

        // Create heatmap image
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var heatmapPixels = [UInt8](repeating: 0, count: heatmapWidth * heatmapHeight * 4)

        for y in 0..<heatmapHeight {
            for x in 0..<heatmapWidth {
                let idx = y * heatmapWidth + x
                let pixelIdx = idx * 4

                let normalizedValue = range > 0 ? (heatmap[idx] - minVal) / range : 0
                let clampedValue = max(0, min(1, normalizedValue))

                let (r, g, b) = colorForValue(clampedValue, scheme: colorScheme)

                heatmapPixels[pixelIdx] = r
                heatmapPixels[pixelIdx + 1] = g
                heatmapPixels[pixelIdx + 2] = b
                heatmapPixels[pixelIdx + 3] = UInt8(alpha * 255 * clampedValue)
            }
        }

        // Create heatmap CGImage
        guard let heatmapDataProvider = CGDataProvider(data: Data(heatmapPixels) as CFData),
              let heatmapCGImage = CGImage(
                width: heatmapWidth,
                height: heatmapHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: heatmapWidth * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: heatmapDataProvider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            throw VisionUtilsError.processingFailed("Failed to create heatmap image")
        }

        // Use UIGraphicsImageRenderer for proper coordinate handling
        let renderer = UIGraphicsImageRenderer(size: size)
        let outputImage = renderer.image { ctx in
            // Draw original image
            image.draw(at: .zero)

            // Draw heatmap scaled to image size
            let heatmapUIImage = UIImage(cgImage: heatmapCGImage)
            heatmapUIImage.draw(in: CGRect(x: 0, y: 0, width: width, height: height), blendMode: .normal, alpha: 1.0)
        }

        // Convert to base64
        guard let imageData = outputImage.jpegData(compressionQuality: CGFloat(quality) / 100.0) else {
            throw VisionUtilsError.processingFailed("Failed to encode image")
        }

        let endTime = CFAbsoluteTimeGetCurrent()

        return [
            "imageBase64": imageData.base64EncodedString(),
            "width": width,
            "height": height,
            "processingTimeMs": (endTime - startTime) * 1000
        ]
    }

    // MARK: - Color Map Helpers

    /// Get color for a normalized value using a color scheme
    private static func colorForValue(_ value: Double, scheme: String) -> (UInt8, UInt8, UInt8) {
        let v = max(0, min(1, value))

        switch scheme {
        case "hot":
            // Black -> Red -> Yellow -> White
            if v < 0.33 {
                let t = v / 0.33
                return (UInt8(t * 255), 0, 0)
            } else if v < 0.66 {
                let t = (v - 0.33) / 0.33
                return (255, UInt8(t * 255), 0)
            } else {
                let t = (v - 0.66) / 0.34
                return (255, 255, UInt8(t * 255))
            }

        case "viridis":
            // Purple -> Blue -> Green -> Yellow
            if v < 0.25 {
                let t = v / 0.25
                return (UInt8(68 + t * (59 - 68)), UInt8(1 + t * (82 - 1)), UInt8(84 + t * (139 - 84)))
            } else if v < 0.5 {
                let t = (v - 0.25) / 0.25
                return (UInt8(59 + t * (33 - 59)), UInt8(82 + t * (145 - 82)), UInt8(139 + t * (140 - 139)))
            } else if v < 0.75 {
                let t = (v - 0.5) / 0.25
                return (UInt8(33 + t * (94 - 33)), UInt8(145 + t * (201 - 145)), UInt8(140 + t * (98 - 140)))
            } else {
                let t = (v - 0.75) / 0.25
                return (UInt8(94 + t * (253 - 94)), UInt8(201 + t * (231 - 201)), UInt8(98 + t * (37 - 98)))
            }

        default: // "jet"
            // Blue -> Cyan -> Green -> Yellow -> Red
            if v < 0.125 {
                let t = v / 0.125
                return (0, 0, UInt8(128 + t * 127))
            } else if v < 0.375 {
                let t = (v - 0.125) / 0.25
                return (0, UInt8(t * 255), 255)
            } else if v < 0.625 {
                let t = (v - 0.375) / 0.25
                return (UInt8(t * 255), 255, UInt8(255 - t * 255))
            } else if v < 0.875 {
                let t = (v - 0.625) / 0.25
                return (255, UInt8(255 - t * 255), 0)
            } else {
                let t = (v - 0.875) / 0.125
                return (UInt8(255 - t * 127), 0, 0)
            }
        }
    }
}
