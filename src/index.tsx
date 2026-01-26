/**
 * VisionUtils - React Native library for extracting pixel data from images
 *
 * Designed for ML/AI inference pipelines, providing efficient image preprocessing
 * with support for various color formats, normalizations, and data layouts.
 */

import VisionUtils from './NativeVisionUtils';
import {
  VisionUtilsException,
  type GetPixelDataOptions,
  type PixelDataResult,
  type BatchOptions,
  type BatchResult,
  type BatchResultItem,
  type PreparedOptions,
  type ColorFormat,
  type DataLayout,
  type OutputFormat,
  type Normalization,
  type ResizeOptions,
  type ImageSource,
  type ImageStatistics,
  type ImageMetadata,
  type ImageValidationOptions,
  type ImageValidationResult,
  type TensorToImageOptions,
  type TensorToImageResult,
  type FiveCropOptions,
  type TenCropOptions,
  type MultiCropResult,
  type ExtractPatchOptions,
  type MemoryLayout,
  type AccelerationType,
  type OutputTarget,
  type ModelPreset,
  type AugmentationOptions,
  type QuantizeOptions,
  type QuantizeResult,
  type DequantizeOptions,
  type DequantizeResult,
  type CalculateQuantizationParamsOptions,
  type QuantizationParams,
  type QuantizationDtype,
  type QuantizationMode,
  type LabelDataset,
  type LabelInfo,
  type GetTopLabelsOptions,
  type TopLabelResult,
  type DatasetInfo,
  type CameraFrameSource,
  type CameraFrameOptions,
  type CameraFrameResult,
  type CameraPixelFormat,
} from './types';

// Re-export all types
export * from './types';

// =============================================================================
// Default Values
// =============================================================================

const DEFAULT_COLOR_FORMAT: ColorFormat = 'rgb';
const DEFAULT_DATA_LAYOUT: DataLayout = 'hwc';
const DEFAULT_MEMORY_LAYOUT: MemoryLayout = 'interleaved';
const DEFAULT_OUTPUT_FORMAT: OutputFormat = 'array';
const DEFAULT_NORMALIZATION: Normalization = { preset: 'scale' };
const DEFAULT_RESIZE_STRATEGY = 'cover' as const;
const DEFAULT_PAD_COLOR: [number, number, number, number] = [0, 0, 0, 255];
const DEFAULT_LETTERBOX_COLOR: [number, number, number] = [114, 114, 114];
const DEFAULT_BATCH_CONCURRENCY = 4;
const DEFAULT_ACCELERATION: AccelerationType = 'auto';
const DEFAULT_OUTPUT_TARGET: OutputTarget = 'default';

// =============================================================================
// Model Preset Configurations
// =============================================================================

interface ModelPresetConfig {
  resize: {
    width: number;
    height: number;
    strategy: 'cover' | 'contain' | 'stretch' | 'letterbox';
  };
  normalization: Normalization;
  colorFormat: ColorFormat;
  dataLayout: DataLayout;
}

const MODEL_PRESETS: Record<ModelPreset, ModelPresetConfig> = {
  yolo: {
    resize: { width: 640, height: 640, strategy: 'letterbox' },
    normalization: { preset: 'scale' },
    colorFormat: 'rgb',
    dataLayout: 'nchw',
  },
  yolov8: {
    resize: { width: 640, height: 640, strategy: 'letterbox' },
    normalization: { preset: 'scale' },
    colorFormat: 'rgb',
    dataLayout: 'nchw',
  },
  mobilenet: {
    resize: { width: 224, height: 224, strategy: 'cover' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nhwc',
  },
  mobilenet_v2: {
    resize: { width: 224, height: 224, strategy: 'cover' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nhwc',
  },
  mobilenet_v3: {
    resize: { width: 224, height: 224, strategy: 'cover' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nhwc',
  },
  efficientnet: {
    resize: { width: 224, height: 224, strategy: 'cover' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nhwc',
  },
  resnet: {
    resize: { width: 224, height: 224, strategy: 'cover' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nchw',
  },
  resnet50: {
    resize: { width: 224, height: 224, strategy: 'cover' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nchw',
  },
  vit: {
    resize: { width: 224, height: 224, strategy: 'cover' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nchw',
  },
  clip: {
    resize: { width: 224, height: 224, strategy: 'cover' },
    normalization: {
      preset: 'custom',
      mean: [0.48145466, 0.4578275, 0.40821073],
      std: [0.26862954, 0.26130258, 0.27577711],
      scale: 1 / 255,
    },
    colorFormat: 'rgb',
    dataLayout: 'nchw',
  },
  sam: {
    resize: { width: 1024, height: 1024, strategy: 'contain' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nchw',
  },
  dino: {
    resize: { width: 224, height: 224, strategy: 'cover' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nchw',
  },
  detr: {
    resize: { width: 800, height: 800, strategy: 'contain' },
    normalization: { preset: 'imagenet' },
    colorFormat: 'rgb',
    dataLayout: 'nchw',
  },
};

// =============================================================================
// Validation
// =============================================================================

/**
 * Validates the source configuration
 */
function validateSource(source: GetPixelDataOptions['source']): void {
  if (!source) {
    throw new VisionUtilsException('INVALID_SOURCE', 'Source is required');
  }

  const validTypes = ['url', 'file', 'base64', 'asset', 'photoLibrary'];
  if (!validTypes.includes(source.type)) {
    throw new VisionUtilsException(
      'INVALID_SOURCE',
      `Invalid source type: ${source.type}. Must be one of: ${validTypes.join(
        ', '
      )}`
    );
  }

  if (source.value === undefined || source.value === null) {
    throw new VisionUtilsException(
      'INVALID_SOURCE',
      'Source value is required'
    );
  }

  // Type-specific validation
  if (source.type === 'url') {
    const urlPattern = /^https?:\/\/.+/i;
    if (typeof source.value !== 'string' || !urlPattern.test(source.value)) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'URL source must be a valid http:// or https:// URL'
      );
    }
  }
}

/**
 * Validates resize options if provided
 */
function validateResize(resize?: ResizeOptions): void {
  if (!resize) return;

  if (typeof resize.width !== 'number' || resize.width <= 0) {
    throw new VisionUtilsException(
      'INVALID_RESIZE',
      'Resize width must be a positive number'
    );
  }

  if (typeof resize.height !== 'number' || resize.height <= 0) {
    throw new VisionUtilsException(
      'INVALID_RESIZE',
      'Resize height must be a positive number'
    );
  }

  if (resize.strategy) {
    const validStrategies = ['cover', 'contain', 'stretch', 'letterbox'];
    if (!validStrategies.includes(resize.strategy)) {
      throw new VisionUtilsException(
        'INVALID_RESIZE',
        `Invalid resize strategy: ${
          resize.strategy
        }. Must be one of: ${validStrategies.join(', ')}`
      );
    }
  }
}

/**
 * Validates normalization options if provided
 */
function validateNormalization(normalization?: Normalization): void {
  if (!normalization) return;

  const validPresets = ['imagenet', 'tensorflow', 'scale', 'raw', 'custom'];
  if (!validPresets.includes(normalization.preset)) {
    throw new VisionUtilsException(
      'INVALID_NORMALIZATION',
      `Invalid normalization preset: ${
        normalization.preset
      }. Must be one of: ${validPresets.join(', ')}`
    );
  }

  if (normalization.preset === 'custom') {
    if (!normalization.mean || !Array.isArray(normalization.mean)) {
      throw new VisionUtilsException(
        'INVALID_NORMALIZATION',
        'Custom normalization requires mean array'
      );
    }
    if (!normalization.std || !Array.isArray(normalization.std)) {
      throw new VisionUtilsException(
        'INVALID_NORMALIZATION',
        'Custom normalization requires std array'
      );
    }
  }
}

/**
 * Validates all options
 */
function validateOptions(options: GetPixelDataOptions): void {
  validateSource(options.source);
  validateResize(options.resize);
  validateNormalization(options.normalization);

  if (options.colorFormat) {
    const validFormats = [
      'rgb',
      'rgba',
      'bgr',
      'bgra',
      'grayscale',
      'hsv',
      'hsl',
      'lab',
      'yuv',
      'ycbcr',
    ];
    if (!validFormats.includes(options.colorFormat)) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        `Invalid color format: ${
          options.colorFormat
        }. Must be one of: ${validFormats.join(', ')}`
      );
    }
  }

  if (options.dataLayout) {
    const validLayouts = ['hwc', 'chw', 'nhwc', 'nchw'];
    if (!validLayouts.includes(options.dataLayout)) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        `Invalid data layout: ${
          options.dataLayout
        }. Must be one of: ${validLayouts.join(', ')}`
      );
    }
  }

  if (options.memoryLayout) {
    const validMemoryLayouts = ['interleaved', 'planar'];
    if (!validMemoryLayouts.includes(options.memoryLayout)) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        `Invalid memory layout: ${
          options.memoryLayout
        }. Must be one of: ${validMemoryLayouts.join(', ')}`
      );
    }
  }

  if (options.outputFormat) {
    const validOutputFormats = [
      'array',
      'float32Array',
      'uint8Array',
      'int8Array',
      'int16Array',
    ];
    if (!validOutputFormats.includes(options.outputFormat)) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        `Invalid output format: ${
          options.outputFormat
        }. Must be one of: ${validOutputFormats.join(', ')}`
      );
    }
  }

  if (options.augmentation) {
    validateAugmentation(options.augmentation);
  }

  if (options.edgeDetection) {
    const validEdgeTypes = ['sobel', 'canny', 'laplacian'];
    if (!validEdgeTypes.includes(options.edgeDetection.type)) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        `Invalid edge detection type: ${
          options.edgeDetection.type
        }. Must be one of: ${validEdgeTypes.join(', ')}`
      );
    }
  }

  if (options.filters?.medianFilter !== undefined) {
    if (options.filters.medianFilter % 2 === 0) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'Median filter kernel size must be an odd number'
      );
    }
  }

  if (options.roi) {
    const { x, y, width, height } = options.roi;
    if (x < 0 || y < 0 || width <= 0 || height <= 0) {
      throw new VisionUtilsException(
        'INVALID_ROI',
        'ROI coordinates must be non-negative and dimensions must be positive'
      );
    }
  }
}

/**
 * Validates augmentation options
 */
function validateAugmentation(augmentation: AugmentationOptions): void {
  if (augmentation.rotation !== undefined) {
    if (augmentation.rotation < 0 || augmentation.rotation > 360) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'Rotation must be between 0 and 360 degrees'
      );
    }
  }

  if (augmentation.brightness !== undefined) {
    if (augmentation.brightness < -1 || augmentation.brightness > 1) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'Brightness must be between -1 and 1'
      );
    }
  }

  if (augmentation.contrast !== undefined) {
    if (augmentation.contrast < 0 || augmentation.contrast > 2) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'Contrast must be between 0 and 2'
      );
    }
  }

  if (augmentation.saturation !== undefined) {
    if (augmentation.saturation < 0 || augmentation.saturation > 2) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'Saturation must be between 0 and 2'
      );
    }
  }

  if (augmentation.hue !== undefined) {
    if (augmentation.hue < -180 || augmentation.hue > 180) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'Hue must be between -180 and 180 degrees'
      );
    }
  }

  if (augmentation.noise) {
    const validNoiseTypes = ['gaussian', 'salt-pepper'];
    if (!validNoiseTypes.includes(augmentation.noise.type)) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        `Invalid noise type: ${
          augmentation.noise.type
        }. Must be one of: ${validNoiseTypes.join(', ')}`
      );
    }
    if (augmentation.noise.intensity < 0 || augmentation.noise.intensity > 1) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'Noise intensity must be between 0 and 1'
      );
    }
  }

  if (augmentation.blur) {
    const validBlurTypes = ['gaussian', 'box'];
    if (!validBlurTypes.includes(augmentation.blur.type)) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        `Invalid blur type: ${
          augmentation.blur.type
        }. Must be one of: ${validBlurTypes.join(', ')}`
      );
    }
    if (augmentation.blur.radius <= 0) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'Blur radius must be positive'
      );
    }
  }
}

// =============================================================================
// Option Preparation
// =============================================================================

/**
 * Applies model preset configuration to options
 */
function applyModelPreset(options: GetPixelDataOptions): GetPixelDataOptions {
  if (!options.modelPreset) return options;

  const preset = MODEL_PRESETS[options.modelPreset];
  if (!preset) return options;

  return {
    ...options,
    resize: options.resize || preset.resize,
    normalization: options.normalization || preset.normalization,
    colorFormat: options.colorFormat || preset.colorFormat,
    dataLayout: options.dataLayout || preset.dataLayout,
  };
}

/**
 * Prepares options with defaults applied
 */
function prepareOptions(options: GetPixelDataOptions): PreparedOptions {
  // Apply model preset first
  const presetApplied = applyModelPreset(options);

  const prepared: PreparedOptions = {
    source: presetApplied.source,
    colorFormat: presetApplied.colorFormat || DEFAULT_COLOR_FORMAT,
    normalization: presetApplied.normalization || DEFAULT_NORMALIZATION,
    dataLayout: presetApplied.dataLayout || DEFAULT_DATA_LAYOUT,
    memoryLayout: presetApplied.memoryLayout || DEFAULT_MEMORY_LAYOUT,
    outputFormat: presetApplied.outputFormat || DEFAULT_OUTPUT_FORMAT,
    acceleration: presetApplied.acceleration || DEFAULT_ACCELERATION,
    outputTarget: presetApplied.outputTarget || DEFAULT_OUTPUT_TARGET,
    modelPreset: presetApplied.modelPreset,
  };

  if (presetApplied.resize) {
    prepared.resize = {
      width: presetApplied.resize.width,
      height: presetApplied.resize.height,
      strategy: presetApplied.resize.strategy || DEFAULT_RESIZE_STRATEGY,
      padColor: presetApplied.resize.padColor || DEFAULT_PAD_COLOR,
      letterboxColor:
        presetApplied.resize.letterboxColor || DEFAULT_LETTERBOX_COLOR,
    };
  }

  if (presetApplied.roi) {
    prepared.roi = presetApplied.roi;
  }

  if (presetApplied.centerCrop) {
    prepared.centerCrop = presetApplied.centerCrop;
  }

  if (presetApplied.augmentation) {
    prepared.augmentation = presetApplied.augmentation;
  }

  if (presetApplied.edgeDetection) {
    prepared.edgeDetection = presetApplied.edgeDetection;
  }

  if (presetApplied.padding) {
    prepared.padding = presetApplied.padding;
  }

  if (presetApplied.preprocessing) {
    prepared.preprocessing = presetApplied.preprocessing;
  }

  if (presetApplied.filters) {
    prepared.filters = presetApplied.filters;
  }

  if (presetApplied.quantization) {
    prepared.quantization = presetApplied.quantization;
  }

  if (presetApplied.cache) {
    prepared.cache = presetApplied.cache;
  }

  return prepared;
}

// =============================================================================
// Output Conversion
// =============================================================================

/**
 * Converts the native result to the requested output format
 */
function convertOutputFormat(
  result: Record<string, unknown>,
  outputFormat: OutputFormat,
  quantization?: { scale: number; zeroPoint: number }
): PixelDataResult {
  const data = result.data as number[];

  switch (outputFormat) {
    case 'float32Array':
      return {
        data: new Float32Array(data),
        width: result.width as number,
        height: result.height as number,
        channels: result.channels as number,
        colorFormat:
          (result.colorFormat as ColorFormat) || DEFAULT_COLOR_FORMAT,
        dataLayout: (result.dataLayout as DataLayout) || DEFAULT_DATA_LAYOUT,
        shape: result.shape as number[],
        processingTimeMs: result.processingTimeMs as number,
      };
    case 'uint8Array':
      // Scale back to 0-255 range if normalized
      const uint8Data = new Uint8Array(data.length);
      for (let i = 0; i < data.length; i++) {
        uint8Data[i] = Math.max(0, Math.min(255, Math.round(data[i]! * 255)));
      }
      return {
        data: uint8Data,
        width: result.width as number,
        height: result.height as number,
        channels: result.channels as number,
        colorFormat:
          (result.colorFormat as ColorFormat) || DEFAULT_COLOR_FORMAT,
        dataLayout: (result.dataLayout as DataLayout) || DEFAULT_DATA_LAYOUT,
        shape: result.shape as number[],
        processingTimeMs: result.processingTimeMs as number,
      };
    case 'int8Array': {
      // Quantize to int8 range [-128, 127]
      const scale = quantization?.scale ?? 1 / 255;
      const zeroPoint = quantization?.zeroPoint ?? 0;
      const int8Data = new Int8Array(data.length);
      for (let i = 0; i < data.length; i++) {
        const quantized = Math.round(data[i]! / scale + zeroPoint);
        int8Data[i] = Math.max(-128, Math.min(127, quantized));
      }
      return {
        data: int8Data,
        width: result.width as number,
        height: result.height as number,
        channels: result.channels as number,
        colorFormat:
          (result.colorFormat as ColorFormat) || DEFAULT_COLOR_FORMAT,
        dataLayout: (result.dataLayout as DataLayout) || DEFAULT_DATA_LAYOUT,
        shape: result.shape as number[],
        processingTimeMs: result.processingTimeMs as number,
      };
    }
    case 'int16Array': {
      // Quantize to int16 range [-32768, 32767]
      const scale = quantization?.scale ?? 1 / 255;
      const zeroPoint = quantization?.zeroPoint ?? 0;
      const int16Data = new Int16Array(data.length);
      for (let i = 0; i < data.length; i++) {
        const quantized = Math.round(data[i]! / scale + zeroPoint);
        int16Data[i] = Math.max(-32768, Math.min(32767, quantized));
      }
      return {
        data: int16Data,
        width: result.width as number,
        height: result.height as number,
        channels: result.channels as number,
        colorFormat:
          (result.colorFormat as ColorFormat) || DEFAULT_COLOR_FORMAT,
        dataLayout: (result.dataLayout as DataLayout) || DEFAULT_DATA_LAYOUT,
        shape: result.shape as number[],
        processingTimeMs: result.processingTimeMs as number,
      };
    }
    default:
      return {
        data: data,
        width: result.width as number,
        height: result.height as number,
        channels: result.channels as number,
        colorFormat:
          (result.colorFormat as ColorFormat) || DEFAULT_COLOR_FORMAT,
        dataLayout: (result.dataLayout as DataLayout) || DEFAULT_DATA_LAYOUT,
        shape: result.shape as number[],
        processingTimeMs: result.processingTimeMs as number,
      };
  }
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Extract pixel data from a single image
 *
 * @param options - Image source and processing options
 * @returns Promise resolving to pixel data result
 *
 * @example
 * // Basic usage
 * const result = await getPixelData({
 *   source: { type: 'url', value: 'https://example.com/image.jpg' }
 * });
 *
 * @example
 * // Full ML preprocessing pipeline
 * const result = await getPixelData({
 *   source: { type: 'file', value: '/path/to/image.jpg' },
 *   resize: { width: 224, height: 224, strategy: 'cover' },
 *   colorFormat: 'rgb',
 *   normalization: { preset: 'imagenet' },
 *   dataLayout: 'nchw',
 *   outputFormat: 'float32Array'
 * });
 */
export async function getPixelData(
  options: GetPixelDataOptions
): Promise<PixelDataResult> {
  try {
    // Validate options
    validateOptions(options);

    // Prepare options with defaults
    const preparedOptions = prepareOptions(options);

    // Call native module
    const result = (await VisionUtils.getPixelData(
      preparedOptions as unknown as Object
    )) as Record<string, unknown>;

    // Convert output format
    return convertOutputFormat(
      result,
      preparedOptions.outputFormat,
      preparedOptions.quantization
    );
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Extract pixel data from multiple images with batch processing
 *
 * @param optionsArray - Array of image options
 * @param batchOptions - Batch processing options
 * @returns Promise resolving to batch results
 *
 * @example
 * const results = await batchGetPixelData(
 *   [
 *     { source: { type: 'url', value: 'https://example.com/1.jpg' } },
 *     { source: { type: 'url', value: 'https://example.com/2.jpg' } },
 *   ],
 *   { concurrency: 4 }
 * );
 */
export async function batchGetPixelData(
  optionsArray: GetPixelDataOptions[],
  batchOptions: BatchOptions = {}
): Promise<BatchResult> {
  try {
    // Validate all options
    optionsArray.forEach((options, index) => {
      try {
        validateOptions(options);
      } catch (error) {
        if (error instanceof VisionUtilsException) {
          throw new VisionUtilsException(
            error.code,
            `[Index ${index}] ${error.originalMessage}`
          );
        }
        throw error;
      }
    });

    // Prepare all options
    const preparedOptionsArray = optionsArray.map(prepareOptions);

    // Prepare batch options
    const preparedBatchOptions = {
      concurrency: batchOptions.concurrency || DEFAULT_BATCH_CONCURRENCY,
    };

    // Call native module
    const result = (await VisionUtils.batchGetPixelData(
      preparedOptionsArray as unknown as Object[],
      preparedBatchOptions as unknown as Object
    )) as Record<string, unknown>;

    // Convert results
    const nativeResults = result.results as Record<string, unknown>[];
    const convertedResults: BatchResultItem[] = nativeResults.map(
      (itemResult, index) => {
        if ((itemResult as { error?: boolean }).error) {
          return itemResult as {
            error: true;
            message: string;
            code: string;
            index: number;
          };
        }

        const outputFormat =
          preparedOptionsArray[index]?.outputFormat || DEFAULT_OUTPUT_FORMAT;
        const quantization = preparedOptionsArray[index]?.quantization;
        return convertOutputFormat(itemResult, outputFormat, quantization);
      }
    );

    return {
      results: convertedResults,
      totalTimeMs: result.totalTimeMs as number,
    };
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Image Statistics API
// =============================================================================

/**
 * Get statistics for an image (mean, std, min, max, histogram)
 *
 * @param source - Image source specification
 * @returns Promise resolving to image statistics
 *
 * @example
 * const stats = await getImageStatistics({ type: 'url', value: 'https://example.com/image.jpg' });
 * console.log(stats.mean);  // [r, g, b] mean values
 * console.log(stats.std);   // [r, g, b] standard deviation
 */
export async function getImageStatistics(
  source: ImageSource
): Promise<ImageStatistics> {
  try {
    validateSource(source);
    const result = (await VisionUtils.getImageStatistics(
      source as unknown as Object
    )) as ImageStatistics;
    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Image Metadata API
// =============================================================================

/**
 * Get metadata for an image (dimensions, format, color space, EXIF)
 *
 * @param source - Image source specification
 * @returns Promise resolving to image metadata
 *
 * @example
 * const metadata = await getImageMetadata({ type: 'file', value: '/path/to/image.jpg' });
 * console.log(metadata.width, metadata.height);
 * console.log(metadata.format);  // 'jpeg', 'png', etc.
 */
export async function getImageMetadata(
  source: ImageSource
): Promise<ImageMetadata> {
  try {
    validateSource(source);
    const result = (await VisionUtils.getImageMetadata(
      source as unknown as Object
    )) as ImageMetadata;
    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Image Validation API
// =============================================================================

/**
 * Validate an image against specified criteria
 *
 * @param source - Image source specification
 * @param options - Validation options
 * @returns Promise resolving to validation result
 *
 * @example
 * const result = await validateImage(
 *   { type: 'url', value: 'https://example.com/image.jpg' },
 *   { minWidth: 224, minHeight: 224, allowedFormats: ['jpeg', 'png'] }
 * );
 * if (!result.isValid) {
 *   console.log(result.errors);
 * }
 */
export async function validateImage(
  source: ImageSource,
  options: ImageValidationOptions = {}
): Promise<ImageValidationResult> {
  try {
    validateSource(source);
    const result = (await VisionUtils.validateImage(
      source as unknown as Object,
      options as unknown as Object
    )) as ImageValidationResult;
    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Tensor to Image API
// =============================================================================

/**
 * Convert tensor data back to an image
 *
 * @param pixelData - Pixel data result to convert
 * @param options - Conversion options
 * @returns Promise resolving to base64 encoded image
 *
 * @example
 * const imageResult = await tensorToImage(pixelDataResult, {
 *   denormalization: { preset: 'imagenet' },
 *   format: 'png'
 * });
 * // Use imageResult.base64 to display or save the image
 */
export async function tensorToImage(
  pixelData: PixelDataResult,
  options: TensorToImageOptions = {}
): Promise<TensorToImageResult> {
  try {
    const dataArray = Array.isArray(pixelData.data)
      ? pixelData.data
      : Array.from(pixelData.data as ArrayLike<number>);

    const result = (await VisionUtils.tensorToImage(
      dataArray,
      pixelData.width,
      pixelData.height,
      {
        ...options,
        dataLayout: pixelData.dataLayout,
        colorFormat: pixelData.colorFormat,
        channels: pixelData.channels,
      } as unknown as Object
    )) as TensorToImageResult;
    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Multi-Crop API
// =============================================================================

/**
 * Perform five-crop operation (4 corners + center)
 *
 * @param options - Base pixel data options (source required)
 * @param cropOptions - Crop dimensions
 * @returns Promise resolving to multi-crop result
 *
 * @example
 * const result = await fiveCrop(
 *   { source: { type: 'url', value: 'https://example.com/image.jpg' } },
 *   { width: 224, height: 224 }
 * );
 * // result.crops contains 5 results: top-left, top-right, bottom-left, bottom-right, center
 */
export async function fiveCrop(
  options: GetPixelDataOptions,
  cropOptions: FiveCropOptions
): Promise<MultiCropResult> {
  try {
    validateSource(options.source);
    const preparedOptions = prepareOptions(options);

    const result = (await VisionUtils.fiveCrop(
      options.source as unknown as Object,
      cropOptions as unknown as Object,
      preparedOptions as unknown as Object
    )) as MultiCropResult;
    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Perform ten-crop operation (five-crop + horizontal flips)
 *
 * @param options - Base pixel data options (source required)
 * @param cropOptions - Crop dimensions and flip options
 * @returns Promise resolving to multi-crop result
 *
 * @example
 * const result = await tenCrop(
 *   { source: { type: 'url', value: 'https://example.com/image.jpg' } },
 *   { width: 224, height: 224 }
 * );
 * // result.crops contains 10 results: 5 crops + 5 flipped versions
 */
export async function tenCrop(
  options: GetPixelDataOptions,
  cropOptions: TenCropOptions
): Promise<MultiCropResult> {
  try {
    validateSource(options.source);
    const preparedOptions = prepareOptions(options);

    const result = (await VisionUtils.tenCrop(
      options.source as unknown as Object,
      cropOptions as unknown as Object,
      preparedOptions as unknown as Object
    )) as MultiCropResult;
    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Tensor Manipulation API
// =============================================================================

/**
 * Extract a specific channel from pixel data
 *
 * @param pixelData - Pixel data result
 * @param channelIndex - Index of the channel to extract (0-based)
 * @returns Promise resolving to single-channel pixel data
 *
 * @example
 * const redChannel = await extractChannel(rgbResult, 0);
 * console.log(redChannel.channels);  // 1
 */
export async function extractChannel(
  pixelData: PixelDataResult,
  channelIndex: number
): Promise<PixelDataResult> {
  try {
    if (channelIndex < 0 || channelIndex >= pixelData.channels) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        `Channel index ${channelIndex} is out of bounds. Image has ${pixelData.channels} channels.`
      );
    }

    const dataArray = Array.isArray(pixelData.data)
      ? pixelData.data
      : Array.from(pixelData.data as ArrayLike<number>);

    const result = (await VisionUtils.extractChannel(
      dataArray,
      pixelData.width,
      pixelData.height,
      pixelData.channels,
      channelIndex,
      pixelData.dataLayout
    )) as Record<string, unknown>;

    return {
      data: result.data as number[],
      width: pixelData.width,
      height: pixelData.height,
      channels: 1,
      colorFormat: 'grayscale',
      dataLayout: pixelData.dataLayout,
      shape: result.shape as number[],
      processingTimeMs: result.processingTimeMs as number,
    };
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Extract a patch/region from pixel data
 *
 * @param pixelData - Pixel data result
 * @param patchOptions - Patch extraction options
 * @returns Promise resolving to extracted patch
 *
 * @example
 * const patch = await extractPatch(result, { x: 10, y: 10, width: 32, height: 32 });
 */
export async function extractPatch(
  pixelData: PixelDataResult,
  patchOptions: ExtractPatchOptions
): Promise<PixelDataResult> {
  try {
    if (
      patchOptions.x < 0 ||
      patchOptions.y < 0 ||
      patchOptions.x + patchOptions.width > pixelData.width ||
      patchOptions.y + patchOptions.height > pixelData.height
    ) {
      throw new VisionUtilsException(
        'INVALID_ROI',
        'Patch extends beyond image bounds'
      );
    }

    const dataArray = Array.isArray(pixelData.data)
      ? pixelData.data
      : Array.from(pixelData.data as ArrayLike<number>);

    const result = (await VisionUtils.extractPatch(
      dataArray,
      pixelData.width,
      pixelData.height,
      pixelData.channels,
      patchOptions as unknown as Object,
      pixelData.dataLayout
    )) as Record<string, unknown>;

    return {
      data: result.data as number[],
      width: patchOptions.width,
      height: patchOptions.height,
      channels: pixelData.channels,
      colorFormat: pixelData.colorFormat,
      dataLayout: pixelData.dataLayout,
      shape: result.shape as number[],
      processingTimeMs: result.processingTimeMs as number,
    };
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Concatenate multiple pixel data results into a batch tensor
 *
 * @param results - Array of pixel data results (must have same dimensions)
 * @returns Promise resolving to batched pixel data
 *
 * @example
 * const batch = await concatenateToBatch([result1, result2, result3]);
 * // batch.shape = [3, height, width, channels] for NHWC
 */
export async function concatenateToBatch(
  results: PixelDataResult[]
): Promise<PixelDataResult> {
  try {
    if (results.length === 0) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        'Cannot concatenate empty array'
      );
    }

    // Validate all results have same dimensions
    const first = results[0]!;
    for (let i = 1; i < results.length; i++) {
      const current = results[i]!;
      if (
        current.width !== first.width ||
        current.height !== first.height ||
        current.channels !== first.channels
      ) {
        throw new VisionUtilsException(
          'INVALID_SOURCE',
          `All results must have same dimensions. Result ${i} differs from result 0.`
        );
      }
    }

    const serializedResults = results.map((r) => ({
      data: Array.isArray(r.data)
        ? r.data
        : Array.from(r.data as ArrayLike<number>),
      width: r.width,
      height: r.height,
      channels: r.channels,
      dataLayout: r.dataLayout,
    }));

    const result = (await VisionUtils.concatenateToBatch(
      serializedResults as unknown as Object[]
    )) as Record<string, unknown>;

    return {
      data: result.data as number[],
      width: first.width,
      height: first.height,
      channels: first.channels,
      colorFormat: first.colorFormat,
      dataLayout: first.dataLayout,
      shape: result.shape as number[],
      processingTimeMs: result.processingTimeMs as number,
    };
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Permute/transpose tensor dimensions
 *
 * @param pixelData - Pixel data result
 * @param order - New dimension order
 * @returns Promise resolving to permuted pixel data
 *
 * @example
 * // Convert from HWC to CHW
 * const chwData = await permute(hwcData, [2, 0, 1]);
 */
export async function permute(
  pixelData: PixelDataResult,
  order: number[]
): Promise<PixelDataResult> {
  try {
    if (order.length !== pixelData.shape.length) {
      throw new VisionUtilsException(
        'INVALID_SOURCE',
        `Order length (${order.length}) must match shape dimensions (${pixelData.shape.length})`
      );
    }

    const dataArray = Array.isArray(pixelData.data)
      ? pixelData.data
      : Array.from(pixelData.data as ArrayLike<number>);

    const result = (await VisionUtils.permute(
      dataArray,
      pixelData.shape,
      order
    )) as Record<string, unknown>;

    // Calculate new layout based on permutation
    const newShape = result.shape as number[];

    return {
      data: result.data as number[],
      width: pixelData.width,
      height: pixelData.height,
      channels: pixelData.channels,
      colorFormat: pixelData.colorFormat,
      dataLayout: pixelData.dataLayout, // Note: layout semantics may change
      shape: newShape,
      processingTimeMs: result.processingTimeMs as number,
    };
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Augmentation API
// =============================================================================

/**
 * Apply augmentations to an image and return as base64
 *
 * @param source - Image source specification
 * @param augmentations - Augmentation options to apply
 * @returns Promise resolving to augmented image as base64
 *
 * @example
 * const augmented = await applyAugmentations(
 *   { type: 'file', value: '/path/to/image.jpg' },
 *   { horizontalFlip: true, brightness: 0.2, rotation: 15 }
 * );
 */
export async function applyAugmentations(
  source: ImageSource,
  augmentations: AugmentationOptions
): Promise<{ base64: string; processingTimeMs: number }> {
  try {
    validateSource(source);
    validateAugmentation(augmentations);

    const result = (await VisionUtils.applyAugmentations(
      source as unknown as Object,
      augmentations as unknown as Object
    )) as { base64: string; processingTimeMs: number };
    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Quantization API
// =============================================================================

/**
 * Quantize floating-point pixel data to integer format for ML inference
 *
 * Supports per-tensor and per-channel quantization with int8, uint8, or int16 output.
 * Per-channel quantization is common in TFLite models for better accuracy.
 *
 * @param data Float pixel data (normalized values, typically [0, 1] or [-1, 1])
 * @param options Quantization options including scale, zeroPoint, and dtype
 * @returns Promise resolving to QuantizeResult with typed array
 *
 * @example
 * // Per-tensor int8 quantization (TFLite style)
 * const pixelData = await getPixelData({
 *   source: { type: 'url', value: imageUrl },
 *   normalization: { preset: 'scale' } // [0, 1]
 * });
 *
 * const quantized = await quantize(Array.from(pixelData.data), {
 *   dtype: 'int8',
 *   mode: 'per-tensor',
 *   scale: 0.00784,    // 1/127.5 for [-1, 1] to [-128, 127]
 *   zeroPoint: 0
 * });
 *
 * @example
 * // Per-channel uint8 quantization
 * const quantized = await quantize(Array.from(pixelData.data), {
 *   dtype: 'uint8',
 *   mode: 'per-channel',
 *   scale: [0.0039, 0.0039, 0.0039],    // 1/255 per channel
 *   zeroPoint: [0, 0, 0],
 *   channels: 3,
 *   dataLayout: 'hwc'
 * });
 */
export async function quantize(
  data: number[] | Float32Array,
  options: QuantizeOptions
): Promise<QuantizeResult> {
  try {
    // Validate options
    if (options.scale === undefined || options.zeroPoint === undefined) {
      throw new VisionUtilsException(
        'INVALID_OPTIONS',
        'scale and zeroPoint are required for quantization'
      );
    }

    const mode = options.mode ?? 'per-tensor';
    const dtype = options.dtype ?? 'int8';

    // Validate per-channel requirements
    if (mode === 'per-channel') {
      if (!Array.isArray(options.scale) || !Array.isArray(options.zeroPoint)) {
        throw new VisionUtilsException(
          'INVALID_OPTIONS',
          'Per-channel mode requires scale and zeroPoint to be arrays'
        );
      }
      if (!options.channels) {
        throw new VisionUtilsException(
          'INVALID_OPTIONS',
          'Per-channel mode requires channels to be specified'
        );
      }
    }

    const dataArray = data instanceof Float32Array ? Array.from(data) : data;

    const result = (await VisionUtils.quantize(dataArray, {
      dtype,
      mode,
      scale: options.scale,
      zeroPoint: options.zeroPoint,
      dataLayout: options.dataLayout ?? 'hwc',
      channels: options.channels ?? 3,
      width: options.width,
      height: options.height,
    })) as {
      data: number[];
      dtype: QuantizationDtype;
      mode: QuantizationMode;
      scale: number | number[];
      zeroPoint: number | number[];
      processingTimeMs: number;
    };

    // Convert to appropriate typed array
    let typedData: Int8Array | Uint8Array | Int16Array;
    switch (result.dtype) {
      case 'int8':
        typedData = new Int8Array(result.data);
        break;
      case 'uint8':
        typedData = new Uint8Array(result.data);
        break;
      case 'int16':
        typedData = new Int16Array(result.data);
        break;
      default:
        typedData = new Int8Array(result.data);
    }

    return {
      data: typedData,
      dtype: result.dtype,
      mode: result.mode,
      scale: result.scale,
      zeroPoint: result.zeroPoint,
      processingTimeMs: result.processingTimeMs,
    };
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Dequantize integer data back to floating-point
 *
 * Reverses the quantization process: float = (int - zeroPoint) * scale
 *
 * @param data Quantized integer data
 * @param options Dequantization options (must match original quantization params)
 * @returns Promise resolving to DequantizeResult with Float32Array
 *
 * @example
 * // Dequantize after model inference
 * const dequantized = await dequantize(modelOutput, {
 *   dtype: 'int8',
 *   scale: 0.00784,
 *   zeroPoint: 0
 * });
 */
export async function dequantize(
  data: number[] | Int8Array | Uint8Array | Int16Array,
  options: DequantizeOptions
): Promise<DequantizeResult> {
  try {
    if (options.scale === undefined || options.zeroPoint === undefined) {
      throw new VisionUtilsException(
        'INVALID_OPTIONS',
        'scale and zeroPoint are required for dequantization'
      );
    }

    const dataArray = Array.isArray(data) ? data : Array.from(data);

    const result = (await VisionUtils.dequantize(dataArray, {
      dtype: options.dtype,
      mode: options.mode ?? 'per-tensor',
      scale: options.scale,
      zeroPoint: options.zeroPoint,
      dataLayout: options.dataLayout ?? 'hwc',
      channels: options.channels ?? 3,
      width: options.width,
      height: options.height,
    })) as {
      data: number[];
      processingTimeMs: number;
    };

    return {
      data: new Float32Array(result.data),
      processingTimeMs: result.processingTimeMs,
    };
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Calculate optimal quantization parameters (scale and zeroPoint) from data
 *
 * Analyzes min/max values to determine the best quantization parameters.
 * Useful when you don't have pre-determined quantization params from a model.
 *
 * @param data Float pixel data to analyze
 * @param options Options for parameter calculation
 * @returns Promise resolving to QuantizationParams
 *
 * @example
 * // Calculate params for int8 quantization
 * const params = await calculateQuantizationParams(Array.from(pixelData.data), {
 *   dtype: 'int8',
 *   symmetric: true  // Forces zeroPoint = 0
 * });
 *
 * // Use calculated params for quantization
 * const quantized = await quantize(data, {
 *   dtype: 'int8',
 *   scale: params.scale,
 *   zeroPoint: params.zeroPoint
 * });
 *
 * @example
 * // Per-channel parameter calculation
 * const params = await calculateQuantizationParams(Array.from(pixelData.data), {
 *   dtype: 'uint8',
 *   mode: 'per-channel',
 *   channels: 3,
 *   dataLayout: 'hwc'
 * });
 */
export async function calculateQuantizationParams(
  data: number[] | Float32Array,
  options?: CalculateQuantizationParamsOptions
): Promise<QuantizationParams> {
  try {
    const dataArray = data instanceof Float32Array ? Array.from(data) : data;

    const result = (await VisionUtils.calculateQuantizationParams(dataArray, {
      dtype: options?.dtype ?? 'int8',
      mode: options?.mode ?? 'per-tensor',
      channels: options?.channels ?? 3,
      dataLayout: options?.dataLayout ?? 'hwc',
      width: options?.width,
      height: options?.height,
      symmetric: options?.symmetric ?? false,
    })) as QuantizationParams;

    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Cache Management API
// =============================================================================

/**
 * Clear the pixel data cache
 *
 * @example
 * await clearCache();
 */
export async function clearCache(): Promise<void> {
  try {
    await VisionUtils.clearCache();
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Get cache statistics
 *
 * @returns Promise resolving to cache statistics
 *
 * @example
 * const stats = await getCacheStats();
 * console.log(stats.hitCount, stats.missCount, stats.size);
 */
export async function getCacheStats(): Promise<{
  hitCount: number;
  missCount: number;
  size: number;
  maxSize: number;
}> {
  try {
    const result = (await VisionUtils.getCacheStats()) as {
      hitCount: number;
      missCount: number;
      size: number;
      maxSize: number;
    };
    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Label Database Functions
// =============================================================================

/**
 * Get a label by index from a dataset
 *
 * Returns the human-readable label for a given class index.
 * Supports COCO (80/91 classes), ImageNet (1000 classes), VOC, CIFAR, and more.
 *
 * @param index - The class index from model output
 * @param dataset - The dataset to use (default: 'coco')
 * @param includeMetadata - Whether to return full LabelInfo with supercategory
 * @returns Promise resolving to label string or LabelInfo object
 *
 * @example
 * // Simple usage - get label string
 * const label = await getLabel(0, 'coco'); // "person"
 *
 * // With metadata
 * const info = await getLabel(16, 'coco', true);
 * // { index: 16, name: "dog", supercategory: "animal" }
 *
 * // ImageNet classification
 * const breed = await getLabel(208, 'imagenet'); // "Labrador retriever"
 */
export async function getLabel(
  index: number,
  dataset: LabelDataset = 'coco',
  includeMetadata: boolean = false
): Promise<string | LabelInfo> {
  if (typeof index !== 'number' || index < 0 || !Number.isInteger(index)) {
    throw new VisionUtilsException(
      'Label index must be a non-negative integer',
      'INVALID_LABEL_INDEX'
    );
  }

  try {
    const result = await VisionUtils.getLabel(index, dataset, includeMetadata);
    return result as string | LabelInfo;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Get top-K labels from classification scores
 *
 * Takes model output scores and returns the top K predictions with labels
 * and confidence scores. Useful for classification model results.
 *
 * @param scores - Array of confidence scores from model output
 * @param options - Options including dataset, k value, and minimum confidence
 * @returns Promise resolving to array of TopLabelResult
 *
 * @example
 * // Get top 5 predictions from MobileNet output
 * const modelOutput = [...]; // 1000 float scores from model
 * const top5 = await getTopLabels(modelOutput, {
 *   dataset: 'imagenet',
 *   k: 5,
 *   minConfidence: 0.01
 * });
 * // [
 * //   { index: 208, label: "Labrador retriever", confidence: 0.85 },
 * //   { index: 209, label: "Golden retriever", confidence: 0.08 },
 * //   ...
 * // ]
 */
export async function getTopLabels(
  scores: number[],
  options: GetTopLabelsOptions = {}
): Promise<TopLabelResult[]> {
  if (!Array.isArray(scores) || scores.length === 0) {
    throw new VisionUtilsException(
      'Scores must be a non-empty array',
      'INVALID_SCORES'
    );
  }

  const opts = {
    dataset: options.dataset,
    k: options.k ?? 5,
    minConfidence: options.minConfidence ?? 0,
    includeMetadata: options.includeMetadata ?? false,
  };

  try {
    const result = await VisionUtils.getTopLabels(scores, opts);
    return result as TopLabelResult[];
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Get all labels for a dataset
 *
 * Returns the complete list of labels for a given dataset.
 * Useful for building UI dropdowns or mapping all indices.
 *
 * @param dataset - The dataset name
 * @returns Promise resolving to array of all label strings
 *
 * @example
 * const cocoLabels = await getAllLabels('coco');
 * // ["person", "bicycle", "car", ..., "toothbrush"] (80 items)
 *
 * const imagenetLabels = await getAllLabels('imagenet');
 * // ["tench", "goldfish", ..., "toilet tissue"] (1000 items)
 */
export async function getAllLabels(dataset: LabelDataset): Promise<string[]> {
  try {
    const result = await VisionUtils.getAllLabels(dataset);
    return result;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Get information about a dataset
 *
 * Returns metadata about a dataset including number of classes,
 * description, and availability status.
 *
 * @param dataset - The dataset name
 * @returns Promise resolving to DatasetInfo
 *
 * @example
 * const info = await getDatasetInfo('imagenet');
 * // {
 * //   name: "imagenet",
 * //   numClasses: 1000,
 * //   description: "ImageNet ILSVRC 2012 classification labels",
 * //   isAvailable: true
 * // }
 */
export async function getDatasetInfo(
  dataset: LabelDataset
): Promise<DatasetInfo> {
  try {
    const result = await VisionUtils.getDatasetInfo(dataset);
    return result as DatasetInfo;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Get list of all available datasets
 *
 * Returns the names of all built-in label datasets that can be used
 * with getLabel() and getTopLabels().
 *
 * @returns Promise resolving to array of dataset names
 *
 * @example
 * const datasets = await getAvailableDatasets();
 * // ["coco", "coco91", "imagenet", "voc", "cifar10", "cifar100"]
 */
export async function getAvailableDatasets(): Promise<LabelDataset[]> {
  try {
    const result = await VisionUtils.getAvailableDatasets();
    return result as LabelDataset[];
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Camera Frame Functions
// =============================================================================

/**
 * Process a camera frame directly to tensor data
 *
 * Converts raw camera frame buffers (YUV, NV12, etc.) directly to normalized
 * tensor data without intermediate JPEG encoding. This is significantly faster
 * than saving frames to disk or converting to base64.
 *
 * Works with react-native-vision-camera frame processors and similar libraries.
 *
 * @param frameSource - Camera frame source with buffer pointer and metadata
 * @param options - Processing options (resize, normalization, etc.)
 * @returns Promise resolving to processed tensor data
 *
 * @example
 * // In a vision-camera frame processor
 * const frameProcessor = useFrameProcessor((frame) => {
 *   'worklet';
 *   const result = processCameraFrame({
 *     bufferType: 'pointer',
 *     buffer: frame.getNativeBuffer().toString(),
 *     width: frame.width,
 *     height: frame.height,
 *     pixelFormat: 'yuv420',
 *     bytesPerRow: frame.bytesPerRow,
 *     orientation: 'portrait'
 *   }, {
 *     targetSize: { width: 224, height: 224 },
 *     normalization: { preset: 'imagenet' },
 *     outputFormat: 'rgb'
 *   });
 *   // result.data is ready for inference
 * }, []);
 */
export async function processCameraFrame(
  frameSource: CameraFrameSource,
  options: CameraFrameOptions = {}
): Promise<CameraFrameResult> {
  // Validate frame source
  if (!frameSource.dataBase64) {
    throw new VisionUtilsException(
      'Frame dataBase64 is required',
      'MISSING_FRAME_DATA'
    );
  }

  if (!frameSource.width || !frameSource.height) {
    throw new VisionUtilsException(
      'Frame width and height are required',
      'MISSING_FRAME_DIMENSIONS'
    );
  }

  if (!frameSource.pixelFormat) {
    throw new VisionUtilsException(
      'Frame pixel format is required',
      'MISSING_PIXEL_FORMAT'
    );
  }

  const opts = {
    outputWidth: options.outputWidth,
    outputHeight: options.outputHeight,
    outputFormat: options.outputFormat ?? 'rgb',
    normalize: options.normalize ?? true,
    mean: options.mean,
    std: options.std,
  };

  try {
    const result = await VisionUtils.processCameraFrame(frameSource, opts);
    return result as CameraFrameResult;
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

/**
 * Convert YUV camera frame to RGB
 *
 * Low-level function to convert YUV/NV12/NV21 buffers to RGB.
 * Use processCameraFrame() for a higher-level API with resize and normalization.
 *
 * @param yBuffer - Y plane buffer pointer or base64 data
 * @param uBuffer - U plane buffer pointer or base64 data
 * @param vBuffer - V plane buffer pointer or base64 data
 * @param width - Frame width
 * @param height - Frame height
 * @param pixelFormat - YUV format type
 * @returns Promise resolving to RGB pixel data
 *
 * @example
 * const rgb = await convertYUVToRGB(
 *   frame.yPlane.toString(),
 *   frame.uPlane.toString(),
 *   frame.vPlane.toString(),
 *   1920, 1080,
 *   'nv21'
 * );
 */
export async function convertYUVToRGB(
  yBuffer: string,
  uBuffer: string,
  vBuffer: string,
  width: number,
  height: number,
  pixelFormat: CameraPixelFormat
): Promise<{ data: number[]; width: number; height: number }> {
  if (width <= 0 || height <= 0) {
    throw new VisionUtilsException(
      'Width and height must be positive',
      'INVALID_DIMENSIONS'
    );
  }

  try {
    const result = await VisionUtils.convertYUVToRGB(
      yBuffer,
      uBuffer,
      vBuffer,
      width,
      height,
      pixelFormat
    );
    return result as { data: number[]; width: number; height: number };
  } catch (error) {
    throw VisionUtilsException.fromNativeError(error);
  }
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get the number of channels for a given color format
 *
 * @param format The color format
 * @returns Number of channels
 *
 * @example
 * const channels = getChannelCount('rgb'); // 3
 * const alphaChannels = getChannelCount('rgba'); // 4
 */
export function getChannelCount(format: ColorFormat): number {
  switch (format) {
    case 'grayscale':
      return 1;
    case 'rgb':
    case 'bgr':
    case 'hsv':
    case 'hsl':
    case 'lab':
    case 'yuv':
    case 'ycbcr':
      return 3;
    case 'rgba':
    case 'bgra':
      return 4;
    default:
      return 3; // Default to 3 channels
  }
}

/**
 * Type guard to check if a result is an error result from batch processing
 *
 * @param result The result to check
 * @returns True if the result is an error
 *
 * @example
 * const results = await batchGetPixelData([...]);
 * const errors = results.results.filter(isVisionUtilsError);
 * const successes = results.results.filter(r => !isVisionUtilsError(r));
 */
export function isVisionUtilsError(
  result: unknown
): result is { error: true; message: string; code: string; index: number } {
  return (
    typeof result === 'object' &&
    result !== null &&
    (result as { error?: boolean }).error === true
  );
}
