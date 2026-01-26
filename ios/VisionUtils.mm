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
