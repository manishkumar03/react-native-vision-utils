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
                let image = try await ImageLoader.loadImage(from: parsedOptions.source)
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
                            let image = try await ImageLoader.loadImage(from: parsedOptions.source)
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
}
