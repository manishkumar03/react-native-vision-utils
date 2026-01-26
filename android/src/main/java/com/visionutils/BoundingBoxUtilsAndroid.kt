package com.visionutils

import kotlin.math.max
import kotlin.math.min

/**
 * Result classes for bounding box operations
 */
data class ScaleBoxesResult(
    val boxes: List<List<Double>>,
    val format: String,
    val processingTimeMs: Double
)

data class ClipBoxesResult(
    val boxes: List<List<Double>>,
    val format: String,
    val removedCount: Int,
    val processingTimeMs: Double
)

data class NMSResult(
    val indices: List<Int>,
    val detections: List<Map<String, Any>>,
    val suppressedCount: Int,
    val processingTimeMs: Double
)

/**
 * Utility class for bounding box operations
 */
object BoundingBoxUtilsAndroid {

    /**
     * Convert boxes between formats (xyxy, xywh, cxcywh)
     */
    fun convertBoxFormat(
        boxes: List<List<Double>>,
        sourceFormat: String,
        targetFormat: String
    ): List<List<Double>> {
        if (sourceFormat == targetFormat) {
            return boxes
        }

        return boxes.map { box ->
            if (box.size != 4) return@map box

            // First convert to xyxy
            val xyxy = when (sourceFormat) {
                "xywh" -> listOf(box[0], box[1], box[0] + box[2], box[1] + box[3])
                "cxcywh" -> {
                    val halfW = box[2] / 2.0
                    val halfH = box[3] / 2.0
                    listOf(box[0] - halfW, box[1] - halfH, box[0] + halfW, box[1] + halfH)
                }
                else -> box // xyxy
            }

            // Then convert from xyxy to target
            when (targetFormat) {
                "xywh" -> listOf(xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
                "cxcywh" -> {
                    val w = xyxy[2] - xyxy[0]
                    val h = xyxy[3] - xyxy[1]
                    listOf(xyxy[0] + w / 2.0, xyxy[1] + h / 2.0, w, h)
                }
                else -> xyxy // xyxy
            }
        }
    }

    /**
     * Scale boxes from one image size to another
     */
    fun scaleBoxes(
        boxes: List<List<Double>>,
        sourceWidth: Double,
        sourceHeight: Double,
        targetWidth: Double,
        targetHeight: Double,
        format: String,
        clip: Boolean = true
    ): ScaleBoxesResult {
        val startTime = System.nanoTime()

        val scaleX = targetWidth / sourceWidth
        val scaleY = targetHeight / sourceHeight

        // Convert to xyxy for scaling
        val xyxyBoxes = if (format != "xyxy") {
            convertBoxFormat(boxes, format, "xyxy")
        } else {
            boxes
        }

        var scaledBoxes = xyxyBoxes.map { box ->
            if (box.size != 4) return@map box
            listOf(box[0] * scaleX, box[1] * scaleY, box[2] * scaleX, box[3] * scaleY)
        }

        // Clip if requested
        if (clip) {
            scaledBoxes = scaledBoxes.map { box ->
                if (box.size != 4) return@map box
                listOf(
                    max(0.0, min(targetWidth, box[0])),
                    max(0.0, min(targetHeight, box[1])),
                    max(0.0, min(targetWidth, box[2])),
                    max(0.0, min(targetHeight, box[3]))
                )
            }
        }

        // Convert back to original format if needed
        val finalBoxes = if (format != "xyxy") {
            convertBoxFormat(scaledBoxes, "xyxy", format)
        } else {
            scaledBoxes
        }

        val endTime = System.nanoTime()
        val processingTimeMs = (endTime - startTime) / 1_000_000.0

        return ScaleBoxesResult(
            boxes = finalBoxes,
            format = format,
            processingTimeMs = processingTimeMs
        )
    }

    /**
     * Clip boxes to image boundaries
     */
    fun clipBoxes(
        boxes: List<List<Double>>,
        width: Double,
        height: Double,
        format: String,
        removeInvalid: Boolean = false
    ): ClipBoxesResult {
        val startTime = System.nanoTime()

        // Convert to xyxy for clipping
        val xyxyBoxes = if (format != "xyxy") {
            convertBoxFormat(boxes, format, "xyxy")
        } else {
            boxes
        }

        var clippedBoxes = xyxyBoxes.map { box ->
            if (box.size != 4) return@map box
            listOf(
                max(0.0, min(width, box[0])),
                max(0.0, min(height, box[1])),
                max(0.0, min(width, box[2])),
                max(0.0, min(height, box[3]))
            )
        }

        var removedCount = 0
        if (removeInvalid) {
            val validBoxes = clippedBoxes.filter { box ->
                if (box.size != 4) return@filter false
                val boxWidth = box[2] - box[0]
                val boxHeight = box[3] - box[1]
                boxWidth > 0 && boxHeight > 0
            }
            removedCount = clippedBoxes.size - validBoxes.size
            clippedBoxes = validBoxes
        }

        // Convert back to original format if needed
        val finalBoxes = if (format != "xyxy") {
            convertBoxFormat(clippedBoxes, "xyxy", format)
        } else {
            clippedBoxes
        }

        val endTime = System.nanoTime()
        val processingTimeMs = (endTime - startTime) / 1_000_000.0

        return ClipBoxesResult(
            boxes = finalBoxes,
            format = format,
            removedCount = removedCount,
            processingTimeMs = processingTimeMs
        )
    }

    /**
     * Calculate IoU between two boxes
     */
    fun calculateIoU(
        box1: List<Double>,
        box2: List<Double>,
        format: String
    ): Map<String, Double> {
        val startTime = System.nanoTime()

        require(box1.size == 4 && box2.size == 4) { "Boxes must have 4 elements" }

        // Convert to xyxy
        val xyxy1 = if (format != "xyxy") {
            convertBoxFormat(listOf(box1), format, "xyxy")[0]
        } else {
            box1
        }
        val xyxy2 = if (format != "xyxy") {
            convertBoxFormat(listOf(box2), format, "xyxy")[0]
        } else {
            box2
        }

        // Calculate intersection
        val x1 = max(xyxy1[0], xyxy2[0])
        val y1 = max(xyxy1[1], xyxy2[1])
        val x2 = min(xyxy1[2], xyxy2[2])
        val y2 = min(xyxy1[3], xyxy2[3])

        val intersectionWidth = max(0.0, x2 - x1)
        val intersectionHeight = max(0.0, y2 - y1)
        val intersectionArea = intersectionWidth * intersectionHeight

        // Calculate areas
        val area1 = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
        val area2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
        val unionArea = area1 + area2 - intersectionArea

        val iou = if (unionArea > 0) intersectionArea / unionArea else 0.0

        val endTime = System.nanoTime()
        val processingTimeMs = (endTime - startTime) / 1_000_000.0

        return mapOf(
            "iou" to iou,
            "intersection" to intersectionArea,
            "union" to unionArea,
            "processingTimeMs" to processingTimeMs
        )
    }

    /**
     * Apply Non-Maximum Suppression
     */
    fun nonMaxSuppression(
        detections: List<Map<String, Any>>,
        iouThreshold: Double,
        scoreThreshold: Double,
        maxDetections: Int?,
        format: String
    ): NMSResult {
        val startTime = System.nanoTime()

        // Extract boxes and scores from detections
        val boxes = detections.mapNotNull { det ->
            @Suppress("UNCHECKED_CAST")
            det["box"] as? List<Double>
        }
        val scores = detections.mapNotNull { det ->
            (det["score"] as? Number)?.toDouble()
        }

        require(boxes.size == scores.size) { "Boxes and scores must have same length" }

        val maxDets = maxDetections ?: 100

        // Convert to xyxy for IoU calculation
        val xyxyBoxes = if (format != "xyxy") {
            convertBoxFormat(boxes, format, "xyxy")
        } else {
            boxes
        }

        // Filter by score threshold and sort by score descending
        data class Candidate(val index: Int, val box: List<Double>, val score: Double, val detection: Map<String, Any>)

        val candidates = detections.indices
            .filter { scores[it] >= scoreThreshold }
            .map { Candidate(it, xyxyBoxes[it], scores[it], detections[it]) }
            .sortedByDescending { it.score }
            .toMutableList()

        val keepIndices = mutableListOf<Int>()
        val keepDetections = mutableListOf<Map<String, Any>>()
        val suppressed = mutableSetOf<Int>()

        for (candidate in candidates) {
            if (candidate.index in suppressed) continue
            if (keepIndices.size >= maxDets) break

            keepIndices.add(candidate.index)
            // Preserve original detection with all fields (box, score, classIndex, label)
            keepDetections.add(candidate.detection)

            // Suppress overlapping boxes
            for (other in candidates) {
                if (other.index !in suppressed && other.index != candidate.index) {
                    val iouResult = calculateIoU(candidate.box, other.box, "xyxy")
                    if ((iouResult["iou"] ?: 0.0) > iouThreshold) {
                        suppressed.add(other.index)
                    }
                }
            }
        }

        val endTime = System.nanoTime()
        val processingTimeMs = (endTime - startTime) / 1_000_000.0

        return NMSResult(
            indices = keepIndices,
            detections = keepDetections,
            suppressedCount = detections.size - keepIndices.size,
            processingTimeMs = processingTimeMs
        )
    }
}
