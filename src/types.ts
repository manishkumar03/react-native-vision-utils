/**
 * VisionUtils - Type Definitions
 *
 * Comprehensive TypeScript types for the vision-utils library,
 * designed for extracting pixel data from images for ML inference.
 */

// =============================================================================
// Image Source Types
// =============================================================================

/**
 * URL-based image source (http:// or https://)
 */
export interface UrlSource {
  type: 'url';
  value: string;
}

/**
 * Local file path source (file:// or absolute path)
 */
export interface FileSource {
  type: 'file';
  value: string;
}

/**
 * Base64 encoded image data (with or without data URI prefix)
 */
export interface Base64Source {
  type: 'base64';
  value: string;
}

/**
 * React Native asset reference (require() result)
 */
export interface AssetSource {
  type: 'asset';
  value: string | number; // Can be string path or require() number
}

/**
 * Photo library reference (iOS: localIdentifier, Android: content:// URI)
 */
export interface PhotoLibrarySource {
  type: 'photoLibrary';
  value: string;
}

/**
 * Union type of all supported image sources
 */
export type ImageSource =
  | UrlSource
  | FileSource
  | Base64Source
  | AssetSource
  | PhotoLibrarySource;

// =============================================================================
// Processing Options
// =============================================================================

/**
 * Color format for output pixel data
 */
export type ColorFormat =
  | 'rgb'
  | 'rgba'
  | 'bgr'
  | 'bgra'
  | 'grayscale'
  | 'hsv'
  | 'hsl'
  | 'lab'
  | 'yuv'
  | 'ycbcr';

/**
 * Extended color format including all supported color spaces
 */
export type ExtendedColorFormat = ColorFormat;

/**
 * Normalization preset for common ML frameworks
 */
export type NormalizationPreset =
  | 'imagenet'
  | 'tensorflow'
  | 'scale'
  | 'raw'
  | 'custom';

/**
 * Normalization options for pixel values
 *
 * @example
 * // ImageNet normalization
 * { preset: 'imagenet' }
 *
 * @example
 * // Custom normalization
 * {
 *   preset: 'custom',
 *   mean: [0.5, 0.5, 0.5],
 *   std: [0.5, 0.5, 0.5],
 *   scale: 1/255
 * }
 */
export interface Normalization {
  preset: NormalizationPreset;
  /** Custom mean values per channel (required for 'custom' preset) */
  mean?: number[];
  /** Custom standard deviation values per channel (required for 'custom' preset) */
  std?: number[];
  /** Scale factor applied before mean/std normalization (default: 1/255 for 'custom') */
  scale?: number;
}

/**
 * Resize strategy options
 *
 * - 'cover': Scale to fill, cropping edges as needed (preserves aspect ratio)
 * - 'contain': Scale to fit within bounds, padding as needed (preserves aspect ratio)
 * - 'stretch': Stretch to exact dimensions (may distort aspect ratio)
 * - 'letterbox': Preserve aspect ratio with letterbox bars (YOLO-style)
 */
export type ResizeStrategy = 'cover' | 'contain' | 'stretch' | 'letterbox';

/**
 * Resize options
 */
export interface ResizeOptions {
  /** Target width in pixels */
  width: number;
  /** Target height in pixels */
  height: number;
  /** Resize strategy (default: 'cover') */
  strategy?: ResizeStrategy;
  /** Padding color for 'contain' strategy [R, G, B, A] (default: [0, 0, 0, 255]) */
  padColor?: [number, number, number, number];
  /** Letterbox color for 'letterbox' strategy [R, G, B] (default: [114, 114, 114] - YOLO gray) */
  letterboxColor?: [number, number, number];
}

/**
 * Data layout format for output tensor
 *
 * - 'hwc': Height × Width × Channels (standard image format)
 * - 'chw': Channels × Height × Width (PyTorch format)
 * - 'nhwc': Batch × Height × Width × Channels (TensorFlow format)
 * - 'nchw': Batch × Channels × Height × Width (PyTorch batched format)
 */
export type DataLayout = 'hwc' | 'chw' | 'nhwc' | 'nchw';

/**
 * Memory layout format
 * - 'interleaved': RGBRGBRGB... (default)
 * - 'planar': RRR...GGG...BBB...
 */
export type MemoryLayout = 'interleaved' | 'planar';

/**
 * Region of interest for cropping
 */
export interface Roi {
  /** X coordinate of top-left corner */
  x: number;
  /** Y coordinate of top-left corner */
  y: number;
  /** Width of the region */
  width: number;
  /** Height of the region */
  height: number;
}

/**
 * Center crop options
 */
export interface CenterCropOptions {
  /** Width of the center crop */
  width: number;
  /** Height of the center crop */
  height: number;
}

/**
 * Output format for pixel data
 *
 * - 'array': Regular JavaScript array (most compatible)
 * - 'float32Array': Float32Array (most efficient for ML)
 * - 'uint8Array': Uint8Array (raw byte values, no normalization)
 * - 'int8Array': Int8Array (for quantized models)
 * - 'int16Array': Int16Array (for quantized models)
 */
export type OutputFormat =
  | 'array'
  | 'float32Array'
  | 'uint8Array'
  | 'int8Array'
  | 'int16Array';

/**
 * Quantization options for int8/int16 output
 */
export interface QuantizationOptions {
  /** Scale factor for quantization */
  scale: number;
  /** Zero point for quantization */
  zeroPoint: number;
}

// =============================================================================
// Advanced Quantization Types (for dedicated quantize function)
// =============================================================================

/**
 * Quantization mode
 * - 'per-tensor': Single scale/zeroPoint for entire tensor
 * - 'per-channel': Different scale/zeroPoint for each channel
 */
export type QuantizationMode = 'per-tensor' | 'per-channel';

/**
 * Output data type for quantization
 * - 'int8': Signed 8-bit integer [-128, 127]
 * - 'uint8': Unsigned 8-bit integer [0, 255]
 * - 'int16': Signed 16-bit integer [-32768, 32767]
 */
export type QuantizationDtype = 'int8' | 'uint8' | 'int16';

/**
 * Options for the dedicated quantize function
 *
 * @example
 * // Per-tensor quantization (TFLite style)
 * {
 *   dtype: 'int8',
 *   mode: 'per-tensor',
 *   scale: 0.00784,
 *   zeroPoint: 0
 * }
 *
 * @example
 * // Per-channel quantization (for RGB channels)
 * {
 *   dtype: 'uint8',
 *   mode: 'per-channel',
 *   scale: [0.00784, 0.00784, 0.00784],
 *   zeroPoint: [128, 128, 128]
 * }
 */
export interface QuantizeOptions {
  /** Output data type (default: 'int8') */
  dtype?: QuantizationDtype;
  /** Quantization mode (default: 'per-tensor') */
  mode?: QuantizationMode;
  /**
   * Scale factor(s) for quantization
   * - For 'per-tensor': single number
   * - For 'per-channel': array of numbers (one per channel)
   */
  scale: number | number[];
  /**
   * Zero point(s) for quantization
   * - For 'per-tensor': single number
   * - For 'per-channel': array of numbers (one per channel)
   */
  zeroPoint: number | number[];
  /** Data layout of input (needed for per-channel mode, default: 'hwc') */
  dataLayout?: DataLayout;
  /** Number of channels in input data (required for per-channel mode) */
  channels?: number;
  /** Image width (required for per-channel mode with CHW layout) */
  width?: number;
  /** Image height (required for per-channel mode with CHW layout) */
  height?: number;
}

/**
 * Result from the quantize function
 */
export interface QuantizeResult {
  /** Quantized data as typed array */
  data: Int8Array | Uint8Array | Int16Array;
  /** Output data type */
  dtype: QuantizationDtype;
  /** Quantization mode used */
  mode: QuantizationMode;
  /** Scale factor(s) used */
  scale: number | number[];
  /** Zero point(s) used */
  zeroPoint: number | number[];
  /** Processing time in milliseconds */
  processingTimeMs: number;
}

/**
 * Options for dequantization (inverse of quantize)
 */
export interface DequantizeOptions {
  /** Input data type */
  dtype: QuantizationDtype;
  /** Quantization mode used for original quantization */
  mode?: QuantizationMode;
  /** Scale factor(s) used for original quantization */
  scale: number | number[];
  /** Zero point(s) used for original quantization */
  zeroPoint: number | number[];
  /** Data layout (needed for per-channel mode) */
  dataLayout?: DataLayout;
  /** Number of channels (required for per-channel mode) */
  channels?: number;
  /** Image width (required for per-channel mode with CHW layout) */
  width?: number;
  /** Image height (required for per-channel mode with CHW layout) */
  height?: number;
}

/**
 * Result from the dequantize function
 */
export interface DequantizeResult {
  /** Dequantized data as Float32Array */
  data: Float32Array;
  /** Processing time in milliseconds */
  processingTimeMs: number;
}

/**
 * Options for calculating quantization parameters from data
 */
export interface CalculateQuantizationParamsOptions {
  /** Target data type */
  dtype?: QuantizationDtype;
  /** Quantization mode */
  mode?: QuantizationMode;
  /** Number of channels (for per-channel mode) */
  channels?: number;
  /** Data layout (for per-channel mode) */
  dataLayout?: DataLayout;
  /** Image width (for per-channel mode) */
  width?: number;
  /** Image height (for per-channel mode) */
  height?: number;
  /** Use symmetric quantization (zeroPoint = 0 for int8) */
  symmetric?: boolean;
}

/**
 * Result from calculateQuantizationParams
 */
export interface QuantizationParams {
  /** Calculated scale(s) */
  scale: number | number[];
  /** Calculated zero point(s) */
  zeroPoint: number | number[];
  /** Min value(s) found in data */
  min: number | number[];
  /** Max value(s) found in data */
  max: number | number[];
}

// =============================================================================
// Image Augmentation Types
// =============================================================================

/**
 * Noise type for augmentation
 */
export type NoiseType = 'gaussian' | 'salt-pepper';

/**
 * Blur type for augmentation
 */
export type BlurType = 'gaussian' | 'box';

/**
 * Noise options
 */
export interface NoiseOptions {
  /** Type of noise to apply */
  type: NoiseType;
  /** Intensity of noise (0 to 1) */
  intensity: number;
}

/**
 * Blur options
 */
export interface BlurOptions {
  /** Type of blur to apply */
  type: BlurType;
  /** Blur radius in pixels */
  radius: number;
}

/**
 * Image augmentation options for training/inference robustness
 */
export interface AugmentationOptions {
  /** Apply horizontal flip */
  horizontalFlip?: boolean;
  /** Apply vertical flip */
  verticalFlip?: boolean;
  /** Rotation angle in degrees (0-360) */
  rotation?: number;
  /** Brightness adjustment (-1 to 1) */
  brightness?: number;
  /** Contrast adjustment (0 to 2, 1 is normal) */
  contrast?: number;
  /** Saturation adjustment (0 to 2, 1 is normal) */
  saturation?: number;
  /** Hue adjustment (-180 to 180 degrees) */
  hue?: number;
  /** Add noise to the image */
  noise?: NoiseOptions;
  /** Apply blur to the image */
  blur?: BlurOptions;
}

// =============================================================================
// Edge Detection Types
// =============================================================================

/**
 * Edge detection algorithm type
 */
export type EdgeDetectionType = 'sobel' | 'canny' | 'laplacian';

/**
 * Edge detection options
 */
export interface EdgeDetectionOptions {
  /** Type of edge detection algorithm */
  type: EdgeDetectionType;
  /** Add edge map as an additional channel instead of replacing image */
  outputAsChannel?: boolean;
  /** Lower threshold for Canny edge detection */
  lowThreshold?: number;
  /** Upper threshold for Canny edge detection */
  highThreshold?: number;
}

// =============================================================================
// Padding Types
// =============================================================================

/**
 * Padding mode
 */
export type PaddingMode = 'constant' | 'reflect' | 'replicate' | 'circular';

/**
 * Padding options
 */
export interface PaddingOptions {
  /** Padding mode */
  mode: PaddingMode;
  /** Padding value for 'constant' mode [R, G, B, A] */
  value?: number[];
  /** Padding size: single number for all sides, or [top, right, bottom, left] */
  size: number | [number, number, number, number];
}

// =============================================================================
// Filter/Preprocessing Types
// =============================================================================

/**
 * Denoise algorithm type
 */
export type DenoiseType = 'bilateral' | 'nlm';

/**
 * Denoise options
 */
export interface DenoiseOptions {
  /** Denoising strength (0 to 1) */
  strength: number;
  /** Denoising algorithm type */
  type: DenoiseType;
}

/**
 * CLAHE (Contrast Limited Adaptive Histogram Equalization) options
 */
export interface ClaheOptions {
  /** Clip limit for contrast limiting */
  clipLimit: number;
  /** Tile size for adaptive processing */
  tileSize: number;
}

/**
 * Preprocessing filter options
 */
export interface PreprocessingOptions {
  /** Apply histogram equalization */
  histogramEqualization?: boolean;
  /** Apply CLAHE (adaptive histogram equalization) */
  clahe?: ClaheOptions;
}

/**
 * Image filter options
 */
export interface FilterOptions {
  /** Sharpen intensity (0 to 1) */
  sharpen?: number;
  /** Denoise options */
  denoise?: DenoiseOptions;
  /** Median filter kernel size (must be odd number) */
  medianFilter?: number;
}

// =============================================================================
// Model Preset Types
// =============================================================================

/**
 * Pre-configured model presets
 */
export type ModelPreset =
  | 'yolo'
  | 'yolov8'
  | 'mobilenet'
  | 'mobilenet_v2'
  | 'mobilenet_v3'
  | 'efficientnet'
  | 'resnet'
  | 'resnet50'
  | 'vit'
  | 'clip'
  | 'sam'
  | 'dino'
  | 'detr';

// =============================================================================
// Acceleration Types
// =============================================================================

/**
 * Hardware acceleration options
 */
export type AccelerationType = 'cpu' | 'gpu' | 'npu' | 'auto';

/**
 * Target output for platform-specific ML frameworks
 */
export type OutputTarget = 'default' | 'coreml' | 'nnapi' | 'tflite';

// =============================================================================
// Cache Types
// =============================================================================

/**
 * Cache options for pixel data
 */
export interface CacheOptions {
  /** Enable caching */
  enabled: boolean;
  /** Custom cache key */
  key?: string;
  /** Time-to-live in milliseconds */
  ttl?: number;
}

// =============================================================================
// Main Options Interface
// =============================================================================

/**
 * Complete options for getPixelData
 *
 * @example
 * // Minimal usage
 * {
 *   source: { type: 'url', value: 'https://example.com/image.jpg' }
 * }
 *
 * @example
 * // Full ML pipeline options
 * {
 *   source: { type: 'file', value: '/path/to/image.jpg' },
 *   resize: { width: 224, height: 224, strategy: 'cover' },
 *   colorFormat: 'rgb',
 *   normalization: { preset: 'imagenet' },
 *   dataLayout: 'nchw',
 *   outputFormat: 'float32Array'
 * }
 */
export interface GetPixelDataOptions {
  /** Image source specification */
  source: ImageSource;
  /** Color format for output (default: 'rgb') */
  colorFormat?: ColorFormat;
  /** Normalization settings (default: { preset: 'scale' }) */
  normalization?: Normalization;
  /** Resize options (optional, uses original size if not specified) */
  resize?: ResizeOptions;
  /** Region of interest to extract (optional) */
  roi?: Roi;
  /** Center crop options (applied after resize) */
  centerCrop?: CenterCropOptions;
  /** Data layout format (default: 'hwc') */
  dataLayout?: DataLayout;
  /** Memory layout format (default: 'interleaved') */
  memoryLayout?: MemoryLayout;
  /** Output format (default: 'array') */
  outputFormat?: OutputFormat;
  /** Quantization options for int8/int16 output */
  quantization?: QuantizationOptions;
  /** Image augmentation options */
  augmentation?: AugmentationOptions;
  /** Edge detection options */
  edgeDetection?: EdgeDetectionOptions;
  /** Padding options */
  padding?: PaddingOptions;
  /** Preprocessing filter options */
  preprocessing?: PreprocessingOptions;
  /** Image filter options */
  filters?: FilterOptions;
  /** Model preset (auto-configures resize, normalization, layout) */
  modelPreset?: ModelPreset;
  /** Hardware acceleration option */
  acceleration?: AccelerationType;
  /** Target output format for platform ML frameworks */
  outputTarget?: OutputTarget;
  /** Cache options */
  cache?: CacheOptions;
}

// =============================================================================
// Result Types
// =============================================================================

/**
 * Pixel data extraction result - base interface
 */
interface PixelDataResultBase {
  /** Output image width */
  width: number;
  /** Output image height */
  height: number;
  /** Number of color channels */
  channels: number;
  /** Color format of the data */
  colorFormat: ColorFormat;
  /** Data layout of the output */
  dataLayout: DataLayout;
  /** Shape of the data tensor (e.g., [224, 224, 3] for HWC) */
  shape: number[];
  /** Processing time in milliseconds */
  processingTimeMs: number;
}

/**
 * Result with regular JavaScript array
 */
export interface PixelDataResultArray extends PixelDataResultBase {
  /** Pixel data as a regular JavaScript array */
  data: number[];
}

/**
 * Result with Float32Array
 */
export interface PixelDataResultFloat32 extends PixelDataResultBase {
  /** Pixel data as Float32Array */
  data: Float32Array;
}

/**
 * Result with Uint8Array
 */
export interface PixelDataResultUint8 extends PixelDataResultBase {
  /** Pixel data as Uint8Array */
  data: Uint8Array;
}

/**
 * Result with Int8Array (for quantized models)
 */
export interface PixelDataResultInt8 extends PixelDataResultBase {
  /** Pixel data as Int8Array */
  data: Int8Array;
}

/**
 * Result with Int16Array (for quantized models)
 */
export interface PixelDataResultInt16 extends PixelDataResultBase {
  /** Pixel data as Int16Array */
  data: Int16Array;
}

/**
 * Union type for all result formats
 */
export type PixelDataResult =
  | PixelDataResultArray
  | PixelDataResultFloat32
  | PixelDataResultUint8
  | PixelDataResultInt8
  | PixelDataResultInt16;

// =============================================================================
// Batch Processing Types
// =============================================================================

/**
 * Options for batch processing
 */
export interface BatchOptions {
  /** Maximum concurrent image processing (default: 4) */
  concurrency?: number;
}

/**
 * Individual result in a batch (may contain error)
 */
export type BatchResultItem =
  | PixelDataResult
  | {
      error: true;
      message: string;
      code: string;
      index: number;
    };

/**
 * Complete batch processing result
 */
export interface BatchResult {
  /** Array of results or errors, in the same order as input */
  results: BatchResultItem[];
  /** Total processing time in milliseconds */
  totalTimeMs: number;
}

// =============================================================================
// Error Types
// =============================================================================

/**
 * Error codes used by VisionUtils
 */
export type ErrorCode =
  | 'INVALID_SOURCE'
  | 'LOAD_ERROR'
  | 'LOAD_FAILED'
  | 'FILE_NOT_FOUND'
  | 'PERMISSION_DENIED'
  | 'PROCESSING_ERROR'
  | 'PROCESSING_FAILED'
  | 'INVALID_ROI'
  | 'INVALID_RESIZE'
  | 'INVALID_NORMALIZATION'
  | 'INVALID_OPTIONS'
  | 'INVALID_CHANNEL'
  | 'INVALID_PATCH'
  | 'DIMENSION_MISMATCH'
  | 'EMPTY_BATCH'
  | 'DECODE_ERROR'
  | 'CANCELLED'
  | 'UNKNOWN'
  | (string & {}); // Allow any string for extensibility

/**
 * Type alias for VisionUtilsException (for compatibility)
 */
export type VisionUtilsError = VisionUtilsException;

/**
 * Custom exception class for VisionUtils errors
 */
export class VisionUtilsException extends Error {
  /** Error code for programmatic handling */
  public readonly code: ErrorCode;
  /** Human-readable error message */
  public readonly originalMessage: string;

  constructor(code: ErrorCode, message: string) {
    super(`[${code}] ${message}`);
    this.name = 'VisionUtilsException';
    this.code = code;
    this.originalMessage = message;

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    const ErrorWithCaptureStackTrace = Error as typeof Error & {
      captureStackTrace?: (
        targetObject: object,
        constructorOpt?: Function
      ) => void;
    };
    if (typeof ErrorWithCaptureStackTrace.captureStackTrace === 'function') {
      ErrorWithCaptureStackTrace.captureStackTrace(this, VisionUtilsException);
    }
  }

  /**
   * Create a VisionUtilsException from a native error
   */
  static fromNativeError(error: unknown): VisionUtilsException {
    if (error instanceof VisionUtilsException) {
      return error;
    }

    if (error && typeof error === 'object') {
      const nativeError = error as { code?: string; message?: string };
      const code = (nativeError.code as ErrorCode) || 'UNKNOWN';
      const message = nativeError.message || 'An unknown error occurred';
      return new VisionUtilsException(code, message);
    }

    return new VisionUtilsException(
      'UNKNOWN',
      String(error) || 'An unknown error occurred'
    );
  }
}

// =============================================================================
// Type Guards
// =============================================================================

/**
 * Type guard to check if a result is an error
 */
export function isErrorResult(
  result: BatchResultItem
): result is { error: true; message: string; code: string; index: number } {
  return (result as { error?: boolean }).error === true;
}

/**
 * Type guard to check if result has Float32Array data
 */
export function isFloat32Result(
  result: PixelDataResult
): result is PixelDataResultFloat32 {
  return result.data instanceof Float32Array;
}

/**
 * Type guard to check if result has Uint8Array data
 */
export function isUint8Result(
  result: PixelDataResult
): result is PixelDataResultUint8 {
  return result.data instanceof Uint8Array;
}

/**
 * Type guard to check if result has Int8Array data
 */
export function isInt8Result(
  result: PixelDataResult
): result is PixelDataResultInt8 {
  return result.data instanceof Int8Array;
}

/**
 * Type guard to check if result has Int16Array data
 */
export function isInt16Result(
  result: PixelDataResult
): result is PixelDataResultInt16 {
  return result.data instanceof Int16Array;
}

// =============================================================================
// Image Statistics Types
// =============================================================================

/**
 * Image histogram data
 */
export interface ImageHistogram {
  /** Red channel histogram (256 bins) */
  red: number[];
  /** Green channel histogram (256 bins) */
  green: number[];
  /** Blue channel histogram (256 bins) */
  blue: number[];
  /** Luminance histogram (256 bins) */
  luminance: number[];
}

/**
 * Image statistics result
 */
export interface ImageStatistics {
  /** Mean values per channel [R, G, B] or [Gray] */
  mean: number[];
  /** Standard deviation per channel */
  std: number[];
  /** Minimum pixel value */
  min: number;
  /** Maximum pixel value */
  max: number;
  /** Image histogram */
  histogram: ImageHistogram;
  /** Processing time in milliseconds */
  processingTimeMs: number;
}

// =============================================================================
// Image Metadata Types
// =============================================================================

/**
 * EXIF metadata (optional)
 */
export interface ExifMetadata {
  /** Camera make */
  make?: string;
  /** Camera model */
  model?: string;
  /** Date taken */
  dateTime?: string;
  /** Exposure time */
  exposureTime?: string;
  /** F-number */
  fNumber?: number;
  /** ISO speed */
  iso?: number;
  /** Focal length */
  focalLength?: number;
  /** GPS latitude */
  latitude?: number;
  /** GPS longitude */
  longitude?: number;
  /** Image orientation (1-8) */
  orientation?: number;
}

/**
 * Image metadata result
 */
export interface ImageMetadata {
  /** Image width in pixels */
  width: number;
  /** Image height in pixels */
  height: number;
  /** Image format (jpeg, png, webp, etc.) */
  format: string;
  /** Color space (sRGB, Adobe RGB, etc.) */
  colorSpace: string;
  /** Whether the image has alpha channel */
  hasAlpha: boolean;
  /** Bits per component */
  bitsPerComponent: number;
  /** File size in bytes (if available) */
  fileSize?: number;
  /** EXIF metadata (if available) */
  exif?: ExifMetadata;
  /** Processing time in milliseconds */
  processingTimeMs: number;
}

// =============================================================================
// Image Validation Types
// =============================================================================

/**
 * Image validation options
 */
export interface ImageValidationOptions {
  /** Minimum width in pixels */
  minWidth?: number;
  /** Minimum height in pixels */
  minHeight?: number;
  /** Maximum width in pixels */
  maxWidth?: number;
  /** Maximum height in pixels */
  maxHeight?: number;
  /** Allowed image formats */
  allowedFormats?: string[];
  /** Maximum file size in bytes */
  maxFileSize?: number;
  /** Require alpha channel */
  requireAlpha?: boolean;
}

/**
 * Image validation result
 */
export interface ImageValidationResult {
  /** Whether the image is valid */
  isValid: boolean;
  /** Validation errors (empty if valid) */
  errors: string[];
  /** Image metadata */
  metadata: ImageMetadata;
}

// =============================================================================
// Tensor Operation Types
// =============================================================================

/**
 * Options for tensor to image conversion
 */
export interface TensorToImageOptions {
  /** Denormalization settings (reverse of normalization) */
  denormalization?: Normalization;
  /** Output image format */
  format?: 'png' | 'jpeg' | 'webp';
  /** JPEG/WebP quality (0-100) */
  quality?: number;
  /** Data layout of input tensor */
  dataLayout?: DataLayout;
  /** Color format of input tensor */
  colorFormat?: ColorFormat;
}

/**
 * Result from tensor to image conversion
 */
export interface TensorToImageResult {
  /** Base64 encoded image data */
  base64: string;
  /** Image width */
  width: number;
  /** Image height */
  height: number;
  /** Image format */
  format: string;
  /** Processing time in milliseconds */
  processingTimeMs: number;
}

/**
 * Options for extracting a channel from tensor
 */
export interface ExtractChannelOptions {
  /** Channel index to extract */
  channelIndex: number;
}

/**
 * Options for extracting a patch from tensor
 */
export interface ExtractPatchOptions {
  /** X coordinate of top-left corner */
  x: number;
  /** Y coordinate of top-left corner */
  y: number;
  /** Width of the patch */
  width: number;
  /** Height of the patch */
  height: number;
}

/**
 * Options for permuting tensor dimensions
 */
export interface PermuteOptions {
  /** New dimension order (e.g., [2, 0, 1] for HWC -> CHW) */
  order: number[];
}

/**
 * Multi-crop result item
 */
export interface CropResultItem {
  /** Crop position identifier */
  position:
    | 'top-left'
    | 'top-right'
    | 'bottom-left'
    | 'bottom-right'
    | 'center';
  /** Whether this is a flipped version */
  flipped: boolean;
  /** Pixel data result */
  result: PixelDataResult;
}

/**
 * Multi-crop result
 */
export interface MultiCropResult {
  /** Array of crop results */
  crops: CropResultItem[];
  /** Total processing time in milliseconds */
  totalTimeMs: number;
}

/**
 * Options for five-crop operation
 */
export interface FiveCropOptions {
  /** Crop width */
  width: number;
  /** Crop height */
  height: number;
}

/**
 * Options for ten-crop operation (five-crop + horizontal flips)
 */
export interface TenCropOptions extends FiveCropOptions {
  /** Include vertical flips instead of horizontal (default: false) */
  verticalFlip?: boolean;
}

// =============================================================================
// Utility Types
// =============================================================================

/**
 * Deep partial type for optional nested properties
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Type for internal prepared options (with all defaults applied)
 */
export interface PreparedOptions
  extends Required<
    Omit<
      GetPixelDataOptions,
      | 'roi'
      | 'resize'
      | 'centerCrop'
      | 'augmentation'
      | 'edgeDetection'
      | 'padding'
      | 'preprocessing'
      | 'filters'
      | 'quantization'
      | 'cache'
      | 'modelPreset'
    >
  > {
  roi?: Roi;
  resize?: Required<ResizeOptions>;
  centerCrop?: CenterCropOptions;
  augmentation?: AugmentationOptions;
  edgeDetection?: EdgeDetectionOptions;
  padding?: PaddingOptions;
  preprocessing?: PreprocessingOptions;
  filters?: FilterOptions;
  quantization?: QuantizationOptions;
  cache?: CacheOptions;
  modelPreset?: ModelPreset;
}
