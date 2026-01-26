import Foundation
import Accelerate

/// Utility class for bounding box operations
@objc(BoundingBoxUtils)
public class BoundingBoxUtils: NSObject {

    // MARK: - Box Format Conversion

    /// Convert boxes between formats (xyxy, xywh, cxcywh)
    @objc
    public static func convertBoxFormat(
        boxes: [[Double]],
        sourceFormat: String,
        targetFormat: String
    ) throws -> [[Double]] {
        guard sourceFormat != targetFormat else {
            return boxes
        }

        return boxes.map { box -> [Double] in
            guard box.count == 4 else { return box }

            // First convert to xyxy
            let xyxy: [Double]
            switch sourceFormat {
            case "xywh":
                // [x, y, w, h] -> [x1, y1, x2, y2]
                xyxy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            case "cxcywh":
                // [cx, cy, w, h] -> [x1, y1, x2, y2]
                let halfW = box[2] / 2.0
                let halfH = box[3] / 2.0
                xyxy = [box[0] - halfW, box[1] - halfH, box[0] + halfW, box[1] + halfH]
            default: // xyxy
                xyxy = box
            }

            // Then convert from xyxy to target
            switch targetFormat {
            case "xywh":
                // [x1, y1, x2, y2] -> [x, y, w, h]
                return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
            case "cxcywh":
                // [x1, y1, x2, y2] -> [cx, cy, w, h]
                let w = xyxy[2] - xyxy[0]
                let h = xyxy[3] - xyxy[1]
                return [xyxy[0] + w / 2.0, xyxy[1] + h / 2.0, w, h]
            default: // xyxy
                return xyxy
            }
        }
    }

    // MARK: - Scale Boxes

    /// Scale boxes from one image size to another
    @objc
    public static func scaleBoxes(
        boxes: [[Double]],
        sourceWidth: Double,
        sourceHeight: Double,
        targetWidth: Double,
        targetHeight: Double,
        format: String,
        clip: Bool
    ) throws -> [[Double]] {
        let scaleX = targetWidth / sourceWidth
        let scaleY = targetHeight / sourceHeight

        // Convert to xyxy for scaling
        let xyxyBoxes: [[Double]]
        if format != "xyxy" {
            xyxyBoxes = try convertBoxFormat(boxes: boxes, sourceFormat: format, targetFormat: "xyxy")
        } else {
            xyxyBoxes = boxes
        }

        var scaledBoxes = xyxyBoxes.map { box -> [Double] in
            guard box.count == 4 else { return box }
            return [box[0] * scaleX, box[1] * scaleY, box[2] * scaleX, box[3] * scaleY]
        }

        // Clip if requested
        if clip {
            scaledBoxes = scaledBoxes.map { box -> [Double] in
                guard box.count == 4 else { return box }
                return [
                    max(0, min(targetWidth, box[0])),
                    max(0, min(targetHeight, box[1])),
                    max(0, min(targetWidth, box[2])),
                    max(0, min(targetHeight, box[3]))
                ]
            }
        }

        // Convert back to original format if needed
        if format != "xyxy" {
            return try convertBoxFormat(boxes: scaledBoxes, sourceFormat: "xyxy", targetFormat: format)
        }

        return scaledBoxes
    }

    // MARK: - Clip Boxes

    /// Clip boxes to image boundaries
    @objc
    public static func clipBoxes(
        boxes: [[Double]],
        width: Double,
        height: Double,
        format: String
    ) throws -> [[Double]] {
        // Convert to xyxy for clipping
        let xyxyBoxes: [[Double]]
        if format != "xyxy" {
            xyxyBoxes = try convertBoxFormat(boxes: boxes, sourceFormat: format, targetFormat: "xyxy")
        } else {
            xyxyBoxes = boxes
        }

        let clippedBoxes = xyxyBoxes.map { box -> [Double] in
            guard box.count == 4 else { return box }
            return [
                max(0, min(width, box[0])),
                max(0, min(height, box[1])),
                max(0, min(width, box[2])),
                max(0, min(height, box[3]))
            ]
        }

        // Convert back to original format if needed
        if format != "xyxy" {
            return try convertBoxFormat(boxes: clippedBoxes, sourceFormat: "xyxy", targetFormat: format)
        }

        return clippedBoxes
    }

    // MARK: - IoU Calculation

    /// Calculate Intersection over Union between two boxes
    @objc
    public static func calculateIoU(
        box1: [Double],
        box2: [Double],
        format: String
    ) throws -> [String: Double] {
        let startTime = CFAbsoluteTimeGetCurrent()

        guard box1.count == 4 && box2.count == 4 else {
            throw VisionUtilsError.invalidInput("Boxes must have 4 elements")
        }

        // Convert to xyxy
        let xyxy1: [Double]
        let xyxy2: [Double]

        if format != "xyxy" {
            let converted1 = try convertBoxFormat(boxes: [box1], sourceFormat: format, targetFormat: "xyxy")
            let converted2 = try convertBoxFormat(boxes: [box2], sourceFormat: format, targetFormat: "xyxy")
            xyxy1 = converted1[0]
            xyxy2 = converted2[0]
        } else {
            xyxy1 = box1
            xyxy2 = box2
        }

        // Calculate intersection
        let x1 = max(xyxy1[0], xyxy2[0])
        let y1 = max(xyxy1[1], xyxy2[1])
        let x2 = min(xyxy1[2], xyxy2[2])
        let y2 = min(xyxy1[3], xyxy2[3])

        let intersectionWidth = max(0, x2 - x1)
        let intersectionHeight = max(0, y2 - y1)
        let intersectionArea = intersectionWidth * intersectionHeight

        // Calculate areas
        let area1 = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
        let area2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
        let unionArea = area1 + area2 - intersectionArea

        let iou = unionArea > 0 ? intersectionArea / unionArea : 0.0

        let endTime = CFAbsoluteTimeGetCurrent()
        let processingTimeMs = (endTime - startTime) * 1000

        return [
            "iou": iou,
            "intersection": intersectionArea,
            "union": unionArea,
            "processingTimeMs": processingTimeMs
        ]
    }

    // MARK: - Non-Maximum Suppression

    /// Apply NMS to filter overlapping detections
    @objc
    public static func nonMaxSuppression(
        boxes: [[Double]],
        scores: [Double],
        iouThreshold: Double,
        scoreThreshold: Double,
        maxDetections: Int,
        format: String
    ) throws -> [String: Any] {
        guard boxes.count == scores.count else {
            throw VisionUtilsError.invalidInput("Boxes and scores must have same length")
        }

        // Convert to xyxy for IoU calculation
        let xyxyBoxes: [[Double]]
        if format != "xyxy" {
            xyxyBoxes = try convertBoxFormat(boxes: boxes, sourceFormat: format, targetFormat: "xyxy")
        } else {
            xyxyBoxes = boxes
        }

        // Filter by score threshold and sort by score descending
        var candidates: [(index: Int, box: [Double], score: Double)] = []
        for i in 0..<boxes.count {
            if scores[i] >= scoreThreshold {
                candidates.append((i, xyxyBoxes[i], scores[i]))
            }
        }
        candidates.sort { $0.score > $1.score }

        // Apply NMS
        var keepIndices: [Int] = []
        var keepDetections: [[String: Any]] = []
        var suppressed = Set<Int>()

        for candidate in candidates {
            if suppressed.contains(candidate.index) {
                continue
            }
            if keepIndices.count >= maxDetections {
                break
            }

            keepIndices.append(candidate.index)
            keepDetections.append([
                "box": boxes[candidate.index],
                "score": candidate.score,
                "index": candidate.index
            ])

            // Suppress overlapping boxes
            for other in candidates {
                if !suppressed.contains(other.index) && other.index != candidate.index {
                    let iouResult = try calculateIoU(
                        box1: candidate.box,
                        box2: other.box,
                        format: "xyxy"
                    )
                    if let iou = iouResult["iou"], iou > iouThreshold {
                        suppressed.insert(other.index)
                    }
                }
            }
        }

        return [
            "indices": keepIndices,
            "detections": keepDetections,
            "totalBefore": boxes.count,
            "totalAfter": keepIndices.count
        ]
    }
}
