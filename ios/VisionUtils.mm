#import "VisionUtils.h"
#import <React/RCTBridge.h>

#if __has_include(<VisionUtils/VisionUtils-Swift.h>)
#import <VisionUtils/VisionUtils-Swift.h>
#else
#import "VisionUtils-Swift.h"
#endif

@implementation VisionUtils

// MARK: - Existing Methods

- (void)getPixelData:(NSDictionary *)options
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge getPixelData:options resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)batchGetPixelData:(NSArray *)optionsArray
             batchOptions:(NSDictionary *)batchOptions
                  resolve:(RCTPromiseResolveBlock)resolve
                   reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge batchGetPixelData:optionsArray
                            batchOptions:batchOptions
                                 resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Image Statistics and Metadata

- (void)getImageStatistics:(NSDictionary *)source
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge getImageStatistics:source resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)getImageMetadata:(NSDictionary *)source
                 resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge getImageMetadata:source resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)validateImage:(NSDictionary *)source
              options:(NSDictionary *)options
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge validateImage:source options:options resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Tensor Operations

- (void)tensorToImage:(NSArray *)data
                width:(double)width
               height:(double)height
              options:(NSDictionary *)options
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge tensorToImage:data
                               width:(NSInteger)width
                              height:(NSInteger)height
                             options:options
                             resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Multi-crop Operations

- (void)fiveCrop:(NSDictionary *)source
         options:(NSDictionary *)options
    pixelOptions:(NSDictionary *)pixelOptions
         resolve:(RCTPromiseResolveBlock)resolve
          reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge fiveCrop:source
                        options:options
                   pixelOptions:pixelOptions
                        resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)tenCrop:(NSDictionary *)source
        options:(NSDictionary *)options
   pixelOptions:(NSDictionary *)pixelOptions
        resolve:(RCTPromiseResolveBlock)resolve
         reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge tenCrop:source
                       options:options
                  pixelOptions:pixelOptions
                       resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Channel and Patch Extraction

- (void)extractChannel:(NSArray *)data
                 width:(double)width
                height:(double)height
              channels:(double)channels
          channelIndex:(double)channelIndex
            dataLayout:(NSString *)dataLayout
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge extractChannel:data
                                width:(NSInteger)width
                               height:(NSInteger)height
                             channels:(NSInteger)channels
                         channelIndex:(NSInteger)channelIndex
                           dataLayout:dataLayout
                              resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)extractPatch:(NSArray *)data
               width:(double)width
              height:(double)height
            channels:(double)channels
        patchOptions:(NSDictionary *)patchOptions
          dataLayout:(NSString *)dataLayout
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge extractPatch:data
                              width:(NSInteger)width
                             height:(NSInteger)height
                           channels:(NSInteger)channels
                       patchOptions:patchOptions
                         dataLayout:dataLayout
                            resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Tensor Manipulation

- (void)concatenateToBatch:(NSArray *)results
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge concatenateToBatch:results resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)permute:(NSArray *)data
          shape:(NSArray *)shape
          order:(NSArray *)order
        resolve:(RCTPromiseResolveBlock)resolve
         reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge permute:data shape:shape order:order resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Augmentation

- (void)applyAugmentations:(NSDictionary *)source
             augmentations:(NSDictionary *)augmentations
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge applyAugmentations:source
                            augmentations:augmentations
                                  resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)colorJitter:(NSDictionary *)source
            options:(NSDictionary *)options
            resolve:(RCTPromiseResolveBlock)resolve
             reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge colorJitter:source
                           options:options
                           resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)cutout:(NSDictionary *)source
       options:(NSDictionary *)options
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge cutout:source
                      options:options
                      resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Quantization

- (void)quantize:(NSArray *)data
         options:(NSDictionary *)options
         resolve:(RCTPromiseResolveBlock)resolve
          reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge quantize:data
                        options:options
                        resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)dequantize:(NSArray *)data
           options:(NSDictionary *)options
           resolve:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge dequantize:data
                          options:options
                          resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)calculateQuantizationParams:(NSArray *)data
                            options:(NSDictionary *)options
                            resolve:(RCTPromiseResolveBlock)resolve
                             reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge calculateQuantizationParams:data
                                           options:options
                                           resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Cache Management

- (void)clearCache:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge clearCacheWithResolve:^{
        resolve(nil);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)getCacheStats:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge getCacheStatsWithResolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Label Database

- (void)getLabel:(double)index
         dataset:(NSString *)dataset
 includeMetadata:(BOOL)includeMetadata
         resolve:(RCTPromiseResolveBlock)resolve
          reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge getLabel:(NSInteger)index
                        dataset:dataset
                includeMetadata:includeMetadata
                        resolve:^(id result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)getTopLabels:(NSArray *)scores
             options:(NSDictionary *)options
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge getTopLabels:scores
                            options:options
                            resolve:^(NSArray *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)getAllLabels:(NSString *)dataset
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge getAllLabels:dataset
                            resolve:^(NSArray *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)getDatasetInfo:(NSString *)dataset
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge getDatasetInfo:dataset
                              resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)getAvailableDatasets:(RCTPromiseResolveBlock)resolve
                      reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge getAvailableDatasetsWithResolve:^(NSArray *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Camera Frame Processing

- (void)processCameraFrame:(NSDictionary *)source
                   options:(NSDictionary *)options
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge processCameraFrame:source
                                  options:options
                                  resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)convertYUVToRGB:(NSString *)yBuffer
                uBuffer:(NSString *)uBuffer
                vBuffer:(NSString *)vBuffer
                  width:(double)width
                 height:(double)height
            pixelFormat:(NSString *)pixelFormat
                resolve:(RCTPromiseResolveBlock)resolve
                 reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge convertYUVToRGB:yBuffer
                               uBuffer:uBuffer
                               vBuffer:vBuffer
                                 width:(NSInteger)width
                                height:(NSInteger)height
                           pixelFormat:pixelFormat
                               resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Bounding Box Utilities

- (void)convertBoxFormat:(NSArray *)boxes
            sourceFormat:(NSString *)sourceFormat
            targetFormat:(NSString *)targetFormat
                 resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge convertBoxFormat:boxes
                           sourceFormat:sourceFormat
                           targetFormat:targetFormat
                                resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)scaleBoxes:(NSArray *)boxes
           options:(NSDictionary *)options
           resolve:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge scaleBoxes:boxes
                          options:options
                          resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)clipBoxes:(NSArray *)boxes
            width:(double)width
           height:(double)height
           format:(NSString *)format
          resolve:(RCTPromiseResolveBlock)resolve
           reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge clipBoxes:boxes
                           width:(NSInteger)width
                          height:(NSInteger)height
                          format:format
                         resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)calculateIoU:(NSArray *)box1
                box2:(NSArray *)box2
              format:(NSString *)format
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge calculateIoU:box1
                               box2:box2
                             format:format
                            resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)nonMaxSuppression:(NSArray *)detections
                  options:(NSDictionary *)options
                  resolve:(RCTPromiseResolveBlock)resolve
                   reject:(RCTPromiseRejectBlock)reject {
    // Extract boxes and scores from detections array
    NSMutableArray *boxes = [NSMutableArray array];
    NSMutableArray *scores = [NSMutableArray array];

    for (NSDictionary *detection in detections) {
        NSArray *box = detection[@"box"];
        NSNumber *score = detection[@"score"];
        if (box && score) {
            [boxes addObject:box];
            [scores addObject:score];
        }
    }

    [VisionUtilsBridge nonMaxSuppression:boxes
                                  scores:scores
                                 options:options
                                 resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Letterbox Utilities

- (void)letterbox:(NSDictionary *)source
          options:(NSDictionary *)options
          resolve:(RCTPromiseResolveBlock)resolve
           reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge letterbox:source
                         options:options
                         resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)reverseLetterbox:(NSArray *)boxes
                 options:(NSDictionary *)options
                 resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge reverseLetterbox:boxes
                                options:options
                                resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Drawing Utilities

- (void)drawBoxes:(NSDictionary *)source
            boxes:(NSArray *)boxes
          options:(NSDictionary *)options
          resolve:(RCTPromiseResolveBlock)resolve
           reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge drawBoxes:source
                           boxes:boxes
                         options:options
                         resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)drawKeypoints:(NSDictionary *)source
            keypoints:(NSArray *)keypoints
              options:(NSDictionary *)options
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge drawKeypoints:source
                           keypoints:keypoints
                             options:options
                             resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)overlayMask:(NSDictionary *)source
               mask:(NSArray *)mask
            options:(NSDictionary *)options
            resolve:(RCTPromiseResolveBlock)resolve
             reject:(RCTPromiseRejectBlock)reject {
    double width = [[options objectForKey:@"maskWidth"] doubleValue];
    double height = [[options objectForKey:@"maskHeight"] doubleValue];
    [VisionUtilsBridge overlayMask:source
                              mask:mask
                             width:width
                            height:height
                           options:options
                           resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

- (void)overlayHeatmap:(NSDictionary *)source
               heatmap:(NSArray *)heatmap
               options:(NSDictionary *)options
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject {
    double width = [[options objectForKey:@"heatmapWidth"] doubleValue];
    double height = [[options objectForKey:@"heatmapHeight"] doubleValue];
    [VisionUtilsBridge overlayHeatmap:source
                              heatmap:heatmap
                                width:width
                               height:height
                              options:options
                              resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Blur Detection

- (void)detectBlur:(NSDictionary *)source
           options:(NSDictionary *)options
           resolve:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge detectBlur:source
                          options:options
                          resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Video Frame Extraction

- (void)extractVideoFrames:(NSDictionary *)source
                   options:(NSDictionary *)options
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge extractVideoFrames:source
                                  options:options
                                  resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Grid Extraction

- (void)extractGrid:(NSDictionary *)source
        gridOptions:(NSDictionary *)gridOptions
       pixelOptions:(NSDictionary *)pixelOptions
            resolve:(RCTPromiseResolveBlock)resolve
             reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge extractGrid:source
                       gridOptions:gridOptions
                      pixelOptions:pixelOptions
                           resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - Random Crop

- (void)randomCrop:(NSDictionary *)source
       cropOptions:(NSDictionary *)cropOptions
      pixelOptions:(NSDictionary *)pixelOptions
           resolve:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject {
    [VisionUtilsBridge randomCrop:source
                      cropOptions:cropOptions
                     pixelOptions:pixelOptions
                          resolve:^(NSDictionary *result) {
        resolve(result);
    } reject:^(NSString *code, NSString *message) {
        reject(code, message, nil);
    }];
}

// MARK: - TurboModule

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeVisionUtilsSpecJSI>(params);
}

+ (NSString *)moduleName
{
  return @"VisionUtils";
}

@end
