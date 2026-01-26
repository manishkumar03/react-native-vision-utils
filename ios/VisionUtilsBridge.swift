import Foundation

/// Bridge class to expose Swift functionality to Objective-C/C++
@objc(VisionUtilsBridge)
public class VisionUtilsBridge: NSObject {

    // MARK: - Image Cache

    private static var imageCache: [String: (data: [Float], timestamp: Date)] = [:]
    private static let cacheLock = NSLock()
    private static var cacheHitCount: Int = 0
    private static var cacheMissCount: Int = 0
    private static let maxCacheSize: Int = 50

    // MARK: - Existing Methods

    @objc
    public static func getPixelData(
        _ options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let optionsDict = options as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid options format")
                    return
                }

                let parsedOptions = try GetPixelDataOptions(from: optionsDict)
                guard let source = parsedOptions.source else {
                    reject("INVALID_SOURCE", "Source is required")
                    return
                }
                let image = try await ImageLoader.loadImage(from: source)
                let result = try PixelProcessor.process(image: image, options: parsedOptions)

                resolve(result.toDictionary() as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    @objc
    public static func batchGetPixelData(
        _ optionsArray: NSArray,
        batchOptions: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            let startTime = CFAbsoluteTimeGetCurrent()

            guard let optionsList = optionsArray as? [[String: Any]] else {
                reject("INVALID_SOURCE", "Invalid options array format")
                return
            }

            // Parse batch options
            let batchDict = batchOptions as? [String: Any] ?? [:]
            let concurrency = (batchDict["concurrency"] as? Int) ?? 4

            // Process images with concurrency limit
            var results: [[String: Any]] = Array(repeating: [:], count: optionsList.count)

            await withTaskGroup(of: (Int, [String: Any]).self) { group in
                var activeCount = 0

                for (idx, optionsDict) in optionsList.enumerated() {
                    // Wait if we've reached concurrency limit
                    while activeCount >= concurrency {
                        if let completed = await group.next() {
                            results[completed.0] = completed.1
                            activeCount -= 1
                        }
                    }

                    let currentIndex = idx
                    activeCount += 1

                    group.addTask {
                        do {
                            let parsedOptions = try GetPixelDataOptions(from: optionsDict)
                            guard let source = parsedOptions.source else {
                                throw VisionUtilsError.invalidSource("Source is required")
                            }
                            let image = try await ImageLoader.loadImage(from: source)
                            let result = try PixelProcessor.process(image: image, options: parsedOptions)
                            return (currentIndex, result.toDictionary())
                        } catch let error as VisionUtilsError {
                            return (currentIndex, [
                                "error": true,
                                "message": error.message,
                                "code": error.code,
                                "index": currentIndex
                            ])
                        } catch {
                            return (currentIndex, [
                                "error": true,
                                "message": error.localizedDescription,
                                "code": "UNKNOWN",
                                "index": currentIndex
                            ])
                        }
                    }
                }

                // Collect remaining results
                for await completed in group {
                    results[completed.0] = completed.1
                }
            }

            let endTime = CFAbsoluteTimeGetCurrent()
            let totalTimeMs = (endTime - startTime) * 1000

            resolve([
                "results": results,
                "totalTimeMs": totalTimeMs
            ] as NSDictionary)
        }
    }

    // MARK: - Image Statistics

    @objc
    public static func getImageStatistics(
        _ source: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)
                let stats = try ImageAnalyzer.getStatistics(from: image)

                resolve(stats as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    // MARK: - Image Metadata

    @objc
    public static func getImageMetadata(
        _ source: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)
                let metadata = ImageAnalyzer.getMetadata(from: image)

                resolve(metadata as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    // MARK: - Image Validation

    @objc
    public static func validateImage(
        _ source: NSDictionary,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let optionsDict = options as? [String: Any] ?? [:]
                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)
                let result = ImageAnalyzer.validate(image: image, options: optionsDict)

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    // MARK: - Tensor to Image

    @objc
    public static func tensorToImage(
        _ data: NSArray,
        width: Int,
        height: Int,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let dataArray = data as? [NSNumber] else {
                    reject("INVALID_DATA", "Invalid data format")
                    return
                }

                let floatData = dataArray.map { $0.floatValue }
                let optionsDict = options as? [String: Any] ?? [:]
                let result = try TensorConverter.tensorToImage(
                    data: floatData,
                    width: width,
                    height: height,
                    options: optionsDict
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    // MARK: - Multi-crop Operations

    @objc
    public static func fiveCrop(
        _ source: NSDictionary,
        options: NSDictionary,
        pixelOptions: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let optionsDict = options as? [String: Any] ?? [:]
                let pixelOptionsDict = pixelOptions as? [String: Any] ?? [:]

                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)
                let result = try MultiCrop.fiveCrop(
                    image: image,
                    options: optionsDict,
                    pixelOptions: pixelOptionsDict
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    @objc
    public static func tenCrop(
        _ source: NSDictionary,
        options: NSDictionary,
        pixelOptions: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let optionsDict = options as? [String: Any] ?? [:]
                let pixelOptionsDict = pixelOptions as? [String: Any] ?? [:]

                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)
                let result = try MultiCrop.tenCrop(
                    image: image,
                    options: optionsDict,
                    pixelOptions: pixelOptionsDict
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    // MARK: - Channel and Patch Extraction

    @objc
    public static func extractChannel(
        _ data: NSArray,
        width: Int,
        height: Int,
        channels: Int,
        channelIndex: Int,
        dataLayout: String,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let dataArray = data as? [NSNumber] else {
                    reject("INVALID_DATA", "Invalid data format")
                    return
                }

                let floatData = dataArray.map { $0.floatValue }
                let result = try TensorOps.extractChannel(
                    data: floatData,
                    width: width,
                    height: height,
                    channels: channels,
                    channelIndex: channelIndex,
                    dataLayout: dataLayout
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    @objc
    public static func extractPatch(
        _ data: NSArray,
        width: Int,
        height: Int,
        channels: Int,
        patchOptions: NSDictionary,
        dataLayout: String,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let dataArray = data as? [NSNumber] else {
                    reject("INVALID_DATA", "Invalid data format")
                    return
                }

                let floatData = dataArray.map { $0.floatValue }
                let optionsDict = patchOptions as? [String: Any] ?? [:]
                let result = try TensorOps.extractPatch(
                    data: floatData,
                    width: width,
                    height: height,
                    channels: channels,
                    options: optionsDict,
                    dataLayout: dataLayout
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    // MARK: - Tensor Manipulation

    @objc
    public static func concatenateToBatch(
        _ results: NSArray,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let resultsArray = results as? [[String: Any]] else {
                    reject("INVALID_DATA", "Invalid results format")
                    return
                }

                let result = try TensorOps.concatenateToBatch(results: resultsArray)
                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    @objc
    public static func permute(
        _ data: NSArray,
        shape: NSArray,
        order: NSArray,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let dataArray = data as? [NSNumber],
                      let shapeArray = shape as? [Int],
                      let orderArray = order as? [Int] else {
                    reject("INVALID_DATA", "Invalid data, shape, or order format")
                    return
                }

                let floatData = dataArray.map { $0.floatValue }
                let result = try TensorOps.permute(
                    data: floatData,
                    shape: shapeArray,
                    order: orderArray
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    // MARK: - Augmentation

    @objc
    public static func applyAugmentations(
        _ source: NSDictionary,
        augmentations: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let augmentationsDict = augmentations as? [String: Any] ?? [:]
                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)
                let result = try ImageAugmenter.apply(
                    to: image,
                    augmentations: augmentationsDict
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    // MARK: - Quantization

    @objc
    public static func quantize(
        _ data: NSArray,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let dataArray = data as? [NSNumber] else {
                    reject("INVALID_DATA", "Invalid data format")
                    return
                }

                let floatData = dataArray.map { $0.floatValue }
                let optionsDict = options as? [String: Any] ?? [:]
                let result = try Quantization.quantize(data: floatData, options: optionsDict)

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    @objc
    public static func dequantize(
        _ data: NSArray,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let dataArray = data as? [NSNumber] else {
                    reject("INVALID_DATA", "Invalid data format")
                    return
                }

                let intData = dataArray.map { $0.intValue }
                let optionsDict = options as? [String: Any] ?? [:]
                let result = try Quantization.dequantize(data: intData, options: optionsDict)

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    @objc
    public static func calculateQuantizationParams(
        _ data: NSArray,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let dataArray = data as? [NSNumber] else {
                    reject("INVALID_DATA", "Invalid data format")
                    return
                }

                let floatData = dataArray.map { $0.floatValue }
                let optionsDict = options as? [String: Any] ?? [:]
                let result = try Quantization.calculateQuantizationParams(data: floatData, options: optionsDict)

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("UNKNOWN", error.localizedDescription)
            }
        }
    }

    // MARK: - Cache Management

    @objc
    public static func clearCache(
        resolve: @escaping () -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        cacheLock.lock()
        imageCache.removeAll()
        cacheHitCount = 0
        cacheMissCount = 0
        cacheLock.unlock()
        resolve()
    }

    @objc
    public static func getCacheStats(
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        cacheLock.lock()
        let stats: [String: Any] = [
            "hitCount": cacheHitCount,
            "missCount": cacheMissCount,
            "size": imageCache.count,
            "maxSize": maxCacheSize
        ]
        cacheLock.unlock()
        resolve(stats as NSDictionary)
    }

    // MARK: - Label Database

    @objc
    public static func getLabel(
        _ index: Int,
        dataset: String,
        includeMetadata: Bool,
        resolve: @escaping (Any) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            let result = try LabelDatabase.shared.getLabel(
                index: index,
                dataset: dataset,
                includeMetadata: includeMetadata
            )

            resolve(result)
        } catch {
            reject("LABEL_ERROR", error.localizedDescription)
        }
    }

    @objc
    public static func getTopLabels(
        _ scores: NSArray,
        options: NSDictionary,
        resolve: @escaping (NSArray) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            guard let scoresArray = scores as? [NSNumber] else {
                reject("INVALID_DATA", "Invalid scores format")
                return
            }

            let doubleScores = scoresArray.map { $0.doubleValue }
            let optionsDict = options as? [String: Any] ?? [:]
            let result = try LabelDatabase.shared.getTopLabels(scores: doubleScores, options: optionsDict)

            resolve(result as NSArray)
        } catch {
            reject("LABEL_ERROR", error.localizedDescription)
        }
    }

    @objc
    public static func getAllLabels(
        _ dataset: String,
        resolve: @escaping (NSArray) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            let result = try LabelDatabase.shared.getAllLabels(dataset: dataset)
            resolve(result as NSArray)
        } catch {
            reject("LABEL_ERROR", error.localizedDescription)
        }
    }

    @objc
    public static func getDatasetInfo(
        _ dataset: String,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            let result = try LabelDatabase.shared.getDatasetInfo(dataset: dataset)
            resolve(result as NSDictionary)
        } catch {
            reject("LABEL_ERROR", error.localizedDescription)
        }
    }

    @objc
    public static func getAvailableDatasets(
        resolve: @escaping (NSArray) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        let result = LabelDatabase.shared.getAvailableDatasets()
        resolve(result as NSArray)
    }

    // MARK: - Camera Frame Processing

    @objc
    public static func processCameraFrame(
        _ source: NSDictionary,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            guard let sourceDict = source as? [String: Any] else {
                reject("INVALID_SOURCE", "Invalid source format")
                return
            }

            let width = sourceDict["width"] as? Int ?? 0
            let height = sourceDict["height"] as? Int ?? 0
            let pixelFormat = sourceDict["pixelFormat"] as? String ?? "yuv420"
            let bytesPerRow = sourceDict["bytesPerRow"] as? Int ?? (width * 4)
            let timestamp = sourceDict["timestamp"] as? Double
            let orientation = sourceDict["orientation"] as? Int ?? 0

            let frameSource = CameraFrameProcessor.FrameSource(
                width: width,
                height: height,
                pixelFormat: pixelFormat,
                bytesPerRow: bytesPerRow,
                timestamp: timestamp,
                orientation: orientation
            )

            let optionsDict = options as? [String: Any] ?? [:]

            // Check for base64 data input
            if let base64Data = sourceDict["dataBase64"] as? String,
               let data = Data(base64Encoded: base64Data) {
                let result = try data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> [String: Any] in
                    return try CameraFrameProcessor.shared.processCameraFrame(
                        buffer: buffer.baseAddress,
                        source: frameSource,
                        options: optionsDict
                    )
                }
                resolve(result as NSDictionary)
            } else {
                // For pointer-based input, we need different handling
                reject("INVALID_SOURCE", "Base64 data required for processCameraFrame. Use frame processor for direct buffer access.")
            }
        } catch {
            reject("CAMERA_FRAME_ERROR", error.localizedDescription)
        }
    }

    @objc
    public static func convertYUVToRGB(
        _ yBuffer: String,
        uBuffer: String,
        vBuffer: String,
        width: Int,
        height: Int,
        pixelFormat: String,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            let options: [String: Any] = [
                "yPlaneBase64": yBuffer,
                "uPlaneBase64": uBuffer,
                "vPlaneBase64": vBuffer,
                "width": width,
                "height": height,
                "pixelFormat": pixelFormat
            ]
            let result = try CameraFrameProcessor.shared.convertYUVToRGB(options: options)
            resolve(result as NSDictionary)
        } catch {
            reject("YUV_CONVERT_ERROR", error.localizedDescription)
        }
    }

    // MARK: - Bounding Box Utilities

    @objc
    public static func convertBoxFormat(
        _ boxes: NSArray,
        sourceFormat: String,
        targetFormat: String,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        let startTime = CFAbsoluteTimeGetCurrent()
        do {
            guard let boxesArray = boxes as? [[NSNumber]] else {
                reject("INVALID_INPUT", "Boxes must be array of arrays")
                return
            }

            let doubleBoxes = boxesArray.map { $0.map { $0.doubleValue } }
            let result = try BoundingBoxUtils.convertBoxFormat(
                boxes: doubleBoxes,
                sourceFormat: sourceFormat,
                targetFormat: targetFormat
            )

            let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            resolve([
                "boxes": result,
                "processingTimeMs": processingTimeMs
            ] as NSDictionary)
        } catch {
            reject("BOX_ERROR", error.localizedDescription)
        }
    }

    @objc
    public static func scaleBoxes(
        _ boxes: NSArray,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        let startTime = CFAbsoluteTimeGetCurrent()
        do {
            guard let boxesArray = boxes as? [[NSNumber]] else {
                reject("INVALID_INPUT", "Boxes must be array of arrays")
                return
            }

            let optionsDict = options as? [String: Any] ?? [:]
            let sourceWidth = optionsDict["sourceWidth"] as? Double ?? 1.0
            let sourceHeight = optionsDict["sourceHeight"] as? Double ?? 1.0
            let targetWidth = optionsDict["targetWidth"] as? Double ?? 1.0
            let targetHeight = optionsDict["targetHeight"] as? Double ?? 1.0
            let format = optionsDict["format"] as? String ?? "xyxy"
            let clip = optionsDict["clip"] as? Bool ?? true

            let doubleBoxes = boxesArray.map { $0.map { $0.doubleValue } }
            let result = try BoundingBoxUtils.scaleBoxes(
                boxes: doubleBoxes,
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                targetWidth: targetWidth,
                targetHeight: targetHeight,
                format: format,
                clip: clip
            )

            let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            resolve([
                "boxes": result,
                "processingTimeMs": processingTimeMs
            ] as NSDictionary)
        } catch {
            reject("BOX_ERROR", error.localizedDescription)
        }
    }

    @objc
    public static func clipBoxes(
        _ boxes: NSArray,
        width: Int,
        height: Int,
        format: String,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        let startTime = CFAbsoluteTimeGetCurrent()
        do {
            guard let boxesArray = boxes as? [[NSNumber]] else {
                reject("INVALID_INPUT", "Boxes must be array of arrays")
                return
            }

            let doubleBoxes = boxesArray.map { $0.map { $0.doubleValue } }
            let result = try BoundingBoxUtils.clipBoxes(
                boxes: doubleBoxes,
                width: Double(width),
                height: Double(height),
                format: format
            )

            let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            resolve([
                "boxes": result,
                "processingTimeMs": processingTimeMs
            ] as NSDictionary)
        } catch {
            reject("BOX_ERROR", error.localizedDescription)
        }
    }

    @objc
    public static func calculateIoU(
        _ box1: NSArray,
        box2: NSArray,
        format: String,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            guard let box1Array = box1 as? [NSNumber],
                  let box2Array = box2 as? [NSNumber] else {
                reject("INVALID_INPUT", "Boxes must be arrays of numbers")
                return
            }

            let result = try BoundingBoxUtils.calculateIoU(
                box1: box1Array.map { $0.doubleValue },
                box2: box2Array.map { $0.doubleValue },
                format: format
            )

            resolve(result as NSDictionary)
        } catch {
            reject("BOX_ERROR", error.localizedDescription)
        }
    }

    @objc
    public static func nonMaxSuppression(
        _ boxes: NSArray,
        scores: NSArray,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        let startTime = CFAbsoluteTimeGetCurrent()
        do {
            guard let boxesArray = boxes as? [[NSNumber]],
                  let scoresArray = scores as? [NSNumber] else {
                reject("INVALID_INPUT", "Invalid boxes or scores format")
                return
            }

            let optionsDict = options as? [String: Any] ?? [:]
            let iouThreshold = optionsDict["iouThreshold"] as? Double ?? 0.5
            let scoreThreshold = optionsDict["scoreThreshold"] as? Double ?? 0.25
            let maxDetections = optionsDict["maxDetections"] as? Int ?? 100
            let format = optionsDict["format"] as? String ?? "xyxy"

            let doubleBoxes = boxesArray.map { $0.map { $0.doubleValue } }
            let doubleScores = scoresArray.map { $0.doubleValue }

            var result = try BoundingBoxUtils.nonMaxSuppression(
                boxes: doubleBoxes,
                scores: doubleScores,
                iouThreshold: iouThreshold,
                scoreThreshold: scoreThreshold,
                maxDetections: maxDetections,
                format: format
            )

            let processingTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            result["processingTimeMs"] = processingTimeMs
            resolve(result as NSDictionary)
        } catch {
            reject("NMS_ERROR", error.localizedDescription)
        }
    }

    // MARK: - Letterbox Utilities

    @objc
    public static func letterbox(
        _ source: NSDictionary,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)

                let optionsDict = options as? [String: Any] ?? [:]
                let targetWidth = optionsDict["targetWidth"] as? Int ?? 640
                let targetHeight = optionsDict["targetHeight"] as? Int ?? 640
                let padColor = (optionsDict["padColor"] as? [NSNumber])?.map { $0.intValue } ?? [114, 114, 114]
                let scaleUp = optionsDict["scaleUp"] as? Bool ?? true
                let autoStride = optionsDict["autoStride"] as? Bool ?? false
                let stride = optionsDict["stride"] as? Int ?? 32
                let center = optionsDict["center"] as? Bool ?? true

                let result = try LetterboxUtils.letterbox(
                    image: image,
                    targetWidth: targetWidth,
                    targetHeight: targetHeight,
                    padColor: padColor,
                    scaleUp: scaleUp,
                    autoStride: autoStride,
                    stride: stride,
                    center: center
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("LETTERBOX_ERROR", error.localizedDescription)
            }
        }
    }

    @objc
    public static func reverseLetterbox(
        _ boxes: NSArray,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            guard let boxesArray = boxes as? [[NSNumber]] else {
                reject("INVALID_INPUT", "Boxes must be array of arrays")
                return
            }

            let opts = options as? [String: Any] ?? [:]
            let scale = opts["scale"] as? Double ?? 1.0
            let offset = (opts["offset"] as? [NSNumber])?.map { $0.doubleValue } ?? [0.0, 0.0]
            let originalSize = (opts["originalSize"] as? [NSNumber])?.map { $0.intValue } ?? [0, 0]
            let format = opts["format"] as? String ?? "xyxy"
            let clip = opts["clip"] as? Bool ?? true

            let doubleBoxes = boxesArray.map { $0.map { $0.doubleValue } }
            let result = try LetterboxUtils.reverseLetterbox(
                boxes: doubleBoxes,
                scale: scale,
                offset: offset,
                originalWidth: originalSize.count > 0 ? originalSize[0] : 0,
                originalHeight: originalSize.count > 1 ? originalSize[1] : 0,
                format: format,
                clip: clip
            )

            resolve([
                "boxes": result.boxes,
                "format": result.format,
                "processingTimeMs": result.processingTimeMs
            ] as NSDictionary)
        } catch {
            reject("LETTERBOX_ERROR", error.localizedDescription)
        }
    }

    // MARK: - Drawing Utilities

    @objc
    public static func drawBoxes(
        _ source: NSDictionary,
        boxes: NSArray,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)

                guard let boxesArray = boxes as? [[String: Any]] else {
                    reject("INVALID_INPUT", "Boxes must be array of objects")
                    return
                }

                let optionsDict = options as? [String: Any] ?? [:]
                let lineWidth = CGFloat(optionsDict["lineWidth"] as? Double ?? 2.0)
                let fontSize = CGFloat(optionsDict["fontSize"] as? Double ?? 14.0)
                let drawLabels = optionsDict["drawLabels"] as? Bool ?? true
                let labelBackgroundAlpha = CGFloat(optionsDict["labelBackgroundAlpha"] as? Double ?? 0.7)
                let labelColor = (optionsDict["labelColor"] as? [NSNumber])?.map { $0.intValue } ?? [255, 255, 255]
                let defaultColor = (optionsDict["color"] as? [NSNumber])?.map { $0.intValue }
                let quality = optionsDict["quality"] as? Int ?? 90

                let result = try DrawingUtils.drawBoxes(
                    image: image,
                    boxes: boxesArray,
                    lineWidth: lineWidth,
                    fontSize: fontSize,
                    drawLabels: drawLabels,
                    labelBackgroundAlpha: labelBackgroundAlpha,
                    labelColor: labelColor,
                    defaultColor: defaultColor,
                    quality: quality
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("DRAW_ERROR", error.localizedDescription)
            }
        }
    }

    @objc
    public static func drawKeypoints(
        _ source: NSDictionary,
        keypoints: NSArray,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)

                guard let keypointsArray = keypoints as? [[String: Any]] else {
                    reject("INVALID_INPUT", "Keypoints must be array of objects")
                    return
                }

                let optionsDict = options as? [String: Any] ?? [:]
                let pointRadius = CGFloat(optionsDict["pointRadius"] as? Double ?? 5.0)
                let pointColors = (optionsDict["pointColor"] as? [[NSNumber]])?.map { $0.map { $0.intValue } }
                let skeleton = optionsDict["skeleton"] as? [[String: Any]]
                let lineWidth = CGFloat(optionsDict["lineWidth"] as? Double ?? 2.0)
                let minConfidence = CGFloat(optionsDict["minConfidence"] as? Double ?? 0.3)
                let quality = optionsDict["quality"] as? Int ?? 90

                let result = try DrawingUtils.drawKeypoints(
                    image: image,
                    keypoints: keypointsArray,
                    pointRadius: pointRadius,
                    pointColors: pointColors,
                    skeleton: skeleton,
                    lineWidth: lineWidth,
                    minConfidence: minConfidence,
                    quality: quality
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("DRAW_ERROR", error.localizedDescription)
            }
        }
    }

    @objc
    public static func overlayMask(
        _ source: NSDictionary,
        mask: NSArray,
        width: Double,
        height: Double,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)

                guard let maskArray = mask as? [NSNumber] else {
                    reject("INVALID_INPUT", "Mask must be array of numbers")
                    return
                }

                let optionsDict = options as? [String: Any] ?? [:]
                let alpha = CGFloat(optionsDict["alpha"] as? Double ?? 0.5)
                let colorMap = (optionsDict["colorMap"] as? [[NSNumber]])?.map { $0.map { $0.intValue } }
                let singleColor = (optionsDict["color"] as? [NSNumber])?.map { $0.intValue }
                let isClassMask = optionsDict["isClassMask"] as? Bool ?? true
                let quality = optionsDict["quality"] as? Int ?? 90

                let result = try DrawingUtils.overlayMask(
                    image: image,
                    mask: maskArray.map { $0.intValue },
                    maskWidth: Int(width),
                    maskHeight: Int(height),
                    alpha: alpha,
                    colorMap: colorMap,
                    singleColor: singleColor,
                    isClassMask: isClassMask,
                    quality: quality
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("MASK_ERROR", error.localizedDescription)
            }
        }
    }

    @objc
    public static func overlayHeatmap(
        _ source: NSDictionary,
        heatmap: NSArray,
        width: Double,
        height: Double,
        options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        Task {
            do {
                guard let sourceDict = source as? [String: Any] else {
                    reject("INVALID_SOURCE", "Invalid source format")
                    return
                }

                let imageSource = try ImageSource(from: sourceDict)
                let image = try await ImageLoader.loadImage(from: imageSource)

                guard let heatmapArray = heatmap as? [NSNumber] else {
                    reject("INVALID_INPUT", "Heatmap must be array of numbers")
                    return
                }

                let optionsDict = options as? [String: Any] ?? [:]
                let alpha = CGFloat(optionsDict["alpha"] as? Double ?? 0.6)
                let colorScheme = optionsDict["colorScheme"] as? String ?? "jet"
                let minValue = optionsDict["minValue"] as? Double
                let maxValue = optionsDict["maxValue"] as? Double
                let quality = optionsDict["quality"] as? Int ?? 90

                let result = try DrawingUtils.overlayHeatmap(
                    image: image,
                    heatmap: heatmapArray.map { $0.doubleValue },
                    heatmapWidth: Int(width),
                    heatmapHeight: Int(height),
                    alpha: alpha,
                    colorScheme: colorScheme,
                    minValue: minValue,
                    maxValue: maxValue,
                    quality: quality
                )

                resolve(result as NSDictionary)
            } catch let error as VisionUtilsError {
                reject(error.code, error.message)
            } catch {
                reject("HEATMAP_ERROR", error.localizedDescription)
            }
        }
    }
}
