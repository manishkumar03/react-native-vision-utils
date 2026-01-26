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
        options: NSDictionary,
        resolve: @escaping (Any) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            let optionsDict = options as? [String: Any] ?? [:]
            let dataset = optionsDict["dataset"] as? String ?? "coco"
            let includeMetadata = optionsDict["includeMetadata"] as? Bool ?? false

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
        _ options: NSDictionary,
        resolve: @escaping (NSDictionary) -> Void,
        reject: @escaping (String, String) -> Void
    ) {
        do {
            let optionsDict = options as? [String: Any] ?? [:]
            let result = try CameraFrameProcessor.shared.convertYUVToRGB(options: optionsDict)
            resolve(result as NSDictionary)
        } catch {
            reject("YUV_CONVERT_ERROR", error.localizedDescription)
        }
    }
}
