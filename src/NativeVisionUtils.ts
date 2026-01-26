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
   * Quantize float data to int8/uint8/int16
   * @param data - Float pixel data as number array
   * @param options - QuantizeOptions serialized as Object
   * @returns Promise resolving to QuantizeResult as Object
   */
  quantize(data: number[], options: Object): Promise<Object>;

  /**
   * Dequantize int8/uint8/int16 data back to float
   * @param data - Quantized data as number array
   * @param options - DequantizeOptions serialized as Object
   * @returns Promise resolving to DequantizeResult as Object
   */
  dequantize(data: number[], options: Object): Promise<Object>;

  /**
   * Calculate optimal quantization parameters from data
   * @param data - Float pixel data as number array
   * @param options - CalculateQuantizationParamsOptions serialized as Object
   * @returns Promise resolving to QuantizationParams as Object
   */
  calculateQuantizationParams(data: number[], options: Object): Promise<Object>;

  // ==========================================================================
  // Label Database Methods
  // ==========================================================================

  /**
   * Get a label by index from a dataset
   * @param index - The class index
   * @param dataset - The dataset name
   * @param includeMetadata - Whether to include extra metadata
   * @returns Promise resolving to label string or LabelInfo object
   */
  getLabel(
    index: number,
    dataset: string,
    includeMetadata: boolean
  ): Promise<Object>;

  /**
   * Get labels for top-K indices with confidence scores
   * @param scores - Array of confidence scores
   * @param options - GetTopLabelsOptions serialized as Object
   * @returns Promise resolving to TopLabelResult array
   */
  getTopLabels(scores: number[], options: Object): Promise<Object>;

  /**
   * Get all labels for a dataset
   * @param dataset - The dataset name
   * @returns Promise resolving to array of labels
   */
  getAllLabels(dataset: string): Promise<string[]>;

  /**
   * Get dataset information
   * @param dataset - The dataset name
   * @returns Promise resolving to DatasetInfo
   */
  getDatasetInfo(dataset: string): Promise<Object>;

  /**
   * Get list of available datasets
   * @returns Promise resolving to array of dataset names
   */
  getAvailableDatasets(): Promise<string[]>;

  // ==========================================================================
  // Camera Frame Methods
  // ==========================================================================

  /**
   * Process a camera frame directly to tensor data
   * @param frameSource - CameraFrameSource serialized as Object
   * @param options - CameraFrameOptions serialized as Object
   * @returns Promise resolving to CameraFrameResult
   */
  processCameraFrame(frameSource: Object, options: Object): Promise<Object>;

  /**
   * Convert YUV camera frame to RGB
   * @param yBuffer - Y plane data or pointer
   * @param uBuffer - U plane data or pointer
   * @param vBuffer - V plane data or pointer
   * @param width - Frame width
   * @param height - Frame height
   * @param pixelFormat - YUV format type
   * @returns Promise resolving to RGB data
   */
  convertYUVToRGB(
    yBuffer: string,
    uBuffer: string,
    vBuffer: string,
    width: number,
    height: number,
    pixelFormat: string
  ): Promise<Object>;

  // ==========================================================================
  // Bounding Box Methods
  // ==========================================================================

  /**
   * Convert bounding boxes between formats (xyxy, xywh, cxcywh)
   * @param boxes - Array of boxes (each box is [a, b, c, d])
   * @param sourceFormat - Input format: 'xyxy' | 'xywh' | 'cxcywh'
   * @param targetFormat - Output format: 'xyxy' | 'xywh' | 'cxcywh'
   * @returns Promise resolving to converted boxes
   */
  convertBoxFormat(
    boxes: number[][],
    sourceFormat: string,
    targetFormat: string
  ): Promise<Object>;

  /**
   * Scale bounding boxes from one image size to another
   * @param boxes - Array of boxes
   * @param options - Scale options with source/target dimensions
   * @returns Promise resolving to scaled boxes
   */
  scaleBoxes(boxes: number[][], options: Object): Promise<Object>;

  /**
   * Clip bounding boxes to image boundaries
   * @param boxes - Array of boxes
   * @param width - Image width
   * @param height - Image height
   * @param format - Box format (default: 'xyxy')
   * @returns Promise resolving to clipped boxes
   */
  clipBoxes(
    boxes: number[][],
    width: number,
    height: number,
    format: string
  ): Promise<Object>;

  /**
   * Calculate IoU between two bounding boxes
   * @param box1 - First box
   * @param box2 - Second box
   * @param format - Box format (default: 'xyxy')
   * @returns Promise resolving to IoU value
   */
  calculateIoU(box1: number[], box2: number[], format: string): Promise<Object>;

  /**
   * Apply Non-Maximum Suppression to filter overlapping detections
   * @param detections - Array of detections with box and score
   * @param options - NMS options (iouThreshold, scoreThreshold, maxDetections, format)
   * @returns Promise resolving to NMS result with indices and filtered detections
   */
  nonMaxSuppression(detections: Object[], options: Object): Promise<Object>;

  // ==========================================================================
  // Letterbox Methods
  // ==========================================================================

  /**
   * Apply letterbox padding to an image (for YOLO-style models)
   * @param source - Image source
   * @param options - Letterbox options (targetWidth, targetHeight, padColor, etc.)
   * @returns Promise resolving to letterboxed image with transform info
   */
  letterbox(source: Object, options: Object): Promise<Object>;

  /**
   * Reverse letterbox transformation on bounding boxes
   * @param boxes - Array of boxes in letterboxed coordinates
   * @param options - Options with scale, offset, originalSize, format
   * @returns Promise resolving to boxes in original image coordinates
   */
  reverseLetterbox(boxes: number[][], options: Object): Promise<Object>;

  // ==========================================================================
  // Drawing/Visualization Methods
  // ==========================================================================

  /**
   * Draw bounding boxes on an image
   * @param source - Image source
   * @param boxes - Array of drawable boxes with labels and colors
   * @param options - Drawing options (lineWidth, fontSize, etc.)
   * @returns Promise resolving to annotated image
   */
  drawBoxes(source: Object, boxes: Object[], options: Object): Promise<Object>;

  /**
   * Draw keypoints on an image (for pose estimation)
   * @param source - Image source
   * @param keypoints - Array of keypoints with x, y, confidence
   * @param options - Drawing options (pointRadius, skeleton, etc.)
   * @returns Promise resolving to annotated image
   */
  drawKeypoints(
    source: Object,
    keypoints: Object[],
    options: Object
  ): Promise<Object>;

  /**
   * Overlay a segmentation mask on an image
   * @param source - Image source
   * @param mask - Mask data as flat array
   * @param options - Overlay options (maskWidth, maskHeight, alpha, colorMap, etc.)
   * @returns Promise resolving to composite image
   */
  overlayMask(source: Object, mask: number[], options: Object): Promise<Object>;

  /**
   * Overlay a heatmap on an image (for attention/CAM visualization)
   * @param source - Image source
   * @param heatmap - Heatmap data as flat array
   * @param options - Overlay options (heatmapWidth, heatmapHeight, alpha, colorScheme, etc.)
   * @returns Promise resolving to composite image
   */
  overlayHeatmap(
    source: Object,
    heatmap: number[],
    options: Object
  ): Promise<Object>;

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
