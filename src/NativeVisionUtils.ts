import { TurboModuleRegistry, type TurboModule } from 'react-native';

/**
 * Native module spec for VisionUtils
 *
 * Note: TurboModules require simple types - complex types are passed as Object
 * and validated/parsed on both TypeScript and native sides.
 */
export interface Spec extends TurboModule {
  /**
   * Get pixel data from a single image
   * @param options - GetPixelDataOptions serialized as Object
   * @returns Promise resolving to PixelDataResult as Object
   */
  getPixelData(options: Object): Promise<Object>;

  /**
   * Get pixel data from multiple images with batch processing
   * @param optionsArray - Array of GetPixelDataOptions serialized as Object[]
   * @param batchOptions - BatchOptions serialized as Object
   * @returns Promise resolving to BatchResult as Object
   */
  batchGetPixelData(
    optionsArray: Object[],
    batchOptions: Object
  ): Promise<Object>;

  /**
   * Get image statistics (mean, std, min, max, histogram)
   * @param source - ImageSource serialized as Object
   * @returns Promise resolving to ImageStatistics as Object
   */
  getImageStatistics(source: Object): Promise<Object>;

  /**
   * Get image metadata (dimensions, format, color space, EXIF)
   * @param source - ImageSource serialized as Object
   * @returns Promise resolving to ImageMetadata as Object
   */
  getImageMetadata(source: Object): Promise<Object>;

  /**
   * Validate an image against specified criteria
   * @param source - ImageSource serialized as Object
   * @param options - ImageValidationOptions serialized as Object
   * @returns Promise resolving to ImageValidationResult as Object
   */
  validateImage(source: Object, options: Object): Promise<Object>;

  /**
   * Convert tensor data back to an image
   * @param data - Pixel data as number array
   * @param width - Image width
   * @param height - Image height
   * @param options - TensorToImageOptions serialized as Object
   * @returns Promise resolving to TensorToImageResult as Object
   */
  tensorToImage(
    data: number[],
    width: number,
    height: number,
    options: Object
  ): Promise<Object>;

  /**
   * Perform five-crop operation (4 corners + center)
   * @param source - ImageSource serialized as Object
   * @param options - FiveCropOptions serialized as Object
   * @param pixelOptions - GetPixelDataOptions serialized as Object
   * @returns Promise resolving to MultiCropResult as Object
   */
  fiveCrop(
    source: Object,
    options: Object,
    pixelOptions: Object
  ): Promise<Object>;

  /**
   * Perform ten-crop operation (five-crop + flips)
   * @param source - ImageSource serialized as Object
   * @param options - TenCropOptions serialized as Object
   * @param pixelOptions - GetPixelDataOptions serialized as Object
   * @returns Promise resolving to MultiCropResult as Object
   */
  tenCrop(
    source: Object,
    options: Object,
    pixelOptions: Object
  ): Promise<Object>;

  /**
   * Extract a specific channel from pixel data
   * @param data - Pixel data as number array
   * @param width - Image width
   * @param height - Image height
   * @param channels - Number of channels
   * @param channelIndex - Index of channel to extract
   * @param dataLayout - Data layout format
   * @returns Promise resolving to extracted channel data
   */
  extractChannel(
    data: number[],
    width: number,
    height: number,
    channels: number,
    channelIndex: number,
    dataLayout: string
  ): Promise<Object>;

  /**
   * Extract a patch/region from pixel data
   * @param data - Pixel data as number array
   * @param width - Image width
   * @param height - Image height
   * @param channels - Number of channels
   * @param patchOptions - ExtractPatchOptions serialized as Object
   * @param dataLayout - Data layout format
   * @returns Promise resolving to extracted patch data
   */
  extractPatch(
    data: number[],
    width: number,
    height: number,
    channels: number,
    patchOptions: Object,
    dataLayout: string
  ): Promise<Object>;

  /**
   * Concatenate multiple results into a batch tensor
   * @param results - Array of pixel data results
   * @returns Promise resolving to concatenated batch tensor
   */
  concatenateToBatch(results: Object[]): Promise<Object>;

  /**
   * Permute/transpose tensor dimensions
   * @param data - Pixel data as number array
   * @param shape - Current shape
   * @param order - New dimension order
   * @returns Promise resolving to permuted data
   */
  permute(data: number[], shape: number[], order: number[]): Promise<Object>;

  /**
   * Apply augmentations to an image
   * @param source - ImageSource serialized as Object
   * @param augmentations - AugmentationOptions serialized as Object
   * @returns Promise resolving to augmented image as base64
   */
  applyAugmentations(source: Object, augmentations: Object): Promise<Object>;

  /**
   * Clear the pixel data cache
   */
  clearCache(): Promise<void>;

  /**
   * Get cache statistics
   * @returns Promise resolving to cache stats
   */
  getCacheStats(): Promise<Object>;
}

export default TurboModuleRegistry.getEnforcing<Spec>('VisionUtils');
