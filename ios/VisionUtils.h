#import <VisionUtilsSpec/VisionUtilsSpec.h>

@interface VisionUtils : NSObject <NativeVisionUtilsSpec>

// Existing methods
- (void)getPixelData:(NSDictionary *)options
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject;

- (void)batchGetPixelData:(NSArray *)optionsArray
             batchOptions:(NSDictionary *)batchOptions
                  resolve:(RCTPromiseResolveBlock)resolve
                   reject:(RCTPromiseRejectBlock)reject;

// Image statistics and metadata
- (void)getImageStatistics:(NSDictionary *)source
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject;

- (void)getImageMetadata:(NSDictionary *)source
                 resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject;

- (void)validateImage:(NSDictionary *)source
              options:(NSDictionary *)options
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject;

// Tensor operations
- (void)tensorToImage:(NSArray *)data
                width:(double)width
               height:(double)height
              options:(NSDictionary *)options
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject;

// Multi-crop operations
- (void)fiveCrop:(NSDictionary *)source
         options:(NSDictionary *)options
    pixelOptions:(NSDictionary *)pixelOptions
         resolve:(RCTPromiseResolveBlock)resolve
          reject:(RCTPromiseRejectBlock)reject;

- (void)tenCrop:(NSDictionary *)source
        options:(NSDictionary *)options
   pixelOptions:(NSDictionary *)pixelOptions
        resolve:(RCTPromiseResolveBlock)resolve
         reject:(RCTPromiseRejectBlock)reject;

// Channel and patch extraction
- (void)extractChannel:(NSArray *)data
                 width:(double)width
                height:(double)height
              channels:(double)channels
          channelIndex:(double)channelIndex
            dataLayout:(NSString *)dataLayout
               resolve:(RCTPromiseResolveBlock)resolve
                reject:(RCTPromiseRejectBlock)reject;

- (void)extractPatch:(NSArray *)data
               width:(double)width
              height:(double)height
            channels:(double)channels
        patchOptions:(NSDictionary *)patchOptions
          dataLayout:(NSString *)dataLayout
             resolve:(RCTPromiseResolveBlock)resolve
              reject:(RCTPromiseRejectBlock)reject;

// Tensor manipulation
- (void)concatenateToBatch:(NSArray *)results
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject;

- (void)permute:(NSArray *)data
          shape:(NSArray *)shape
          order:(NSArray *)order
        resolve:(RCTPromiseResolveBlock)resolve
         reject:(RCTPromiseRejectBlock)reject;

// Augmentation
- (void)applyAugmentations:(NSDictionary *)source
             augmentations:(NSDictionary *)augmentations
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject;

// Quantization
- (void)quantize:(NSArray *)data
         options:(NSDictionary *)options
         resolve:(RCTPromiseResolveBlock)resolve
          reject:(RCTPromiseRejectBlock)reject;

- (void)dequantize:(NSArray *)data
           options:(NSDictionary *)options
           resolve:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject;

- (void)calculateQuantizationParams:(NSArray *)data
                            options:(NSDictionary *)options
                            resolve:(RCTPromiseResolveBlock)resolve
                             reject:(RCTPromiseRejectBlock)reject;

// Cache management
- (void)clearCache:(RCTPromiseResolveBlock)resolve
            reject:(RCTPromiseRejectBlock)reject;

- (void)getCacheStats:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject;

@end
