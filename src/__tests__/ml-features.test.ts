/**
 * Tests for ML preprocessing features
 */

import {
  mockGetPixelData,
  mockGetImageStatistics,
  mockGetImageMetadata,
  mockValidateImage,
  mockTensorToImage,
  mockFiveCrop,
  mockTenCrop,
  mockExtractChannel,
  mockExtractPatch,
  mockConcatenateToBatch,
  mockPermute,
  mockApplyAugmentations,
  mockClearCache,
  mockGetCacheStats,
} from './jest.setup';
import {
  getPixelData,
  getImageStatistics,
  getImageMetadata,
  validateImage,
  tensorToImage,
  fiveCrop,
  tenCrop,
  extractChannel,
  extractPatch,
  concatenateToBatch,
  permute,
  applyAugmentations,
  clearCache,
  getCacheStats,
} from '../index';
import type {
  PixelDataResult,
  ImageStatistics,
  ImageMetadata,
  ImageValidationResult,
  TensorToImageResult,
  MultiCropResult,
} from '../types';

describe('Model Presets', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should apply YOLO preset configuration', async () => {
    const mockResult: PixelDataResult = {
      data: new Array(640 * 640 * 3).fill(0.5),
      width: 640,
      height: 640,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'nchw',
      shape: [1, 3, 640, 640],
      processingTimeMs: 10,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      modelPreset: 'yolo',
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    expect(calledOptions.resize.width).toBe(640);
    expect(calledOptions.resize.height).toBe(640);
    expect(calledOptions.resize.strategy).toBe('letterbox');
    expect(calledOptions.dataLayout).toBe('nchw');
    expect(calledOptions.normalization.preset).toBe('scale');
  });

  it('should apply MobileNet preset configuration', async () => {
    const mockResult: PixelDataResult = {
      data: new Array(224 * 224 * 3).fill(0.5),
      width: 224,
      height: 224,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'nhwc',
      shape: [1, 224, 224, 3],
      processingTimeMs: 10,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      modelPreset: 'mobilenet',
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    expect(calledOptions.resize.width).toBe(224);
    expect(calledOptions.resize.height).toBe(224);
    expect(calledOptions.normalization.preset).toBe('imagenet');
    expect(calledOptions.dataLayout).toBe('nhwc');
  });

  it('should apply CLIP preset configuration', async () => {
    const mockResult: PixelDataResult = {
      data: new Array(224 * 224 * 3).fill(0.5),
      width: 224,
      height: 224,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'nchw',
      shape: [1, 3, 224, 224],
      processingTimeMs: 10,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      modelPreset: 'clip',
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    expect(calledOptions.normalization.preset).toBe('custom');
    expect(calledOptions.normalization.mean).toEqual([
      0.48145466, 0.4578275, 0.40821073,
    ]);
  });

  it('should allow overriding preset options', async () => {
    const mockResult: PixelDataResult = {
      data: new Array(256 * 256 * 3).fill(0.5),
      width: 256,
      height: 256,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'nchw',
      shape: [1, 3, 256, 256],
      processingTimeMs: 10,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      modelPreset: 'yolo',
      resize: { width: 256, height: 256, strategy: 'cover' },
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    // User-specified resize should override preset
    expect(calledOptions.resize.width).toBe(256);
    expect(calledOptions.resize.height).toBe(256);
    expect(calledOptions.resize.strategy).toBe('cover');
  });
});

describe('Augmentation Options', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should pass augmentation options to native module', async () => {
    const mockResult: PixelDataResult = {
      data: [0.5],
      width: 224,
      height: 224,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [224, 224, 3],
      processingTimeMs: 10,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      augmentation: {
        horizontalFlip: true,
        rotation: 15,
        brightness: 0.2,
        contrast: 1.1,
      },
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    expect(calledOptions.augmentation).toEqual({
      horizontalFlip: true,
      rotation: 15,
      brightness: 0.2,
      contrast: 1.1,
    });
  });

  it('should validate rotation range', async () => {
    await expect(
      getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        augmentation: { rotation: 400 },
      })
    ).rejects.toThrow('Rotation must be between 0 and 360 degrees');
  });

  it('should validate brightness range', async () => {
    await expect(
      getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        augmentation: { brightness: 2 },
      })
    ).rejects.toThrow('Brightness must be between -1 and 1');
  });

  it('should validate noise options', async () => {
    await expect(
      getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        augmentation: { noise: { type: 'gaussian', intensity: 2 } },
      })
    ).rejects.toThrow('Noise intensity must be between 0 and 1');
  });
});

describe('Color Format Extensions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should accept HSV color format', async () => {
    const mockResult: PixelDataResult = {
      data: [0.5, 0.5, 0.5],
      width: 1,
      height: 1,
      channels: 3,
      colorFormat: 'hsv',
      dataLayout: 'hwc',
      shape: [1, 1, 3],
      processingTimeMs: 1,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      colorFormat: 'hsv',
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    expect(calledOptions.colorFormat).toBe('hsv');
  });

  it('should accept LAB color format', async () => {
    const mockResult: PixelDataResult = {
      data: [0.5, 0.5, 0.5],
      width: 1,
      height: 1,
      channels: 3,
      colorFormat: 'lab',
      dataLayout: 'hwc',
      shape: [1, 1, 3],
      processingTimeMs: 1,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      colorFormat: 'lab',
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    expect(calledOptions.colorFormat).toBe('lab');
  });
});

describe('Output Format Extensions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return Int8Array for outputFormat int8Array', async () => {
    const mockResult = {
      data: [0.5, -0.5, 1.0, -1.0],
      width: 2,
      height: 1,
      channels: 2,
      colorFormat: 'grayscale',
      dataLayout: 'hwc',
      shape: [1, 2, 2],
      processingTimeMs: 1,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    const result = await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      outputFormat: 'int8Array',
      quantization: { scale: 0.00784, zeroPoint: 0 },
    });

    expect(result.data).toBeInstanceOf(Int8Array);
  });

  it('should return Int16Array for outputFormat int16Array', async () => {
    const mockResult = {
      data: [0.5, -0.5, 1.0, -1.0],
      width: 2,
      height: 1,
      channels: 2,
      colorFormat: 'grayscale',
      dataLayout: 'hwc',
      shape: [1, 2, 2],
      processingTimeMs: 1,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    const result = await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      outputFormat: 'int16Array',
    });

    expect(result.data).toBeInstanceOf(Int16Array);
  });
});

describe('Edge Detection Options', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should pass edge detection options', async () => {
    const mockResult: PixelDataResult = {
      data: [0.5],
      width: 224,
      height: 224,
      channels: 1,
      colorFormat: 'grayscale',
      dataLayout: 'hwc',
      shape: [224, 224, 1],
      processingTimeMs: 10,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      edgeDetection: { type: 'canny', lowThreshold: 50, highThreshold: 150 },
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    expect(calledOptions.edgeDetection).toEqual({
      type: 'canny',
      lowThreshold: 50,
      highThreshold: 150,
    });
  });

  it('should validate edge detection type', async () => {
    await expect(
      getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        edgeDetection: { type: 'invalid' as any },
      })
    ).rejects.toThrow('Invalid edge detection type');
  });
});

describe('Filter Options', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should pass filter options', async () => {
    const mockResult: PixelDataResult = {
      data: [0.5],
      width: 224,
      height: 224,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [224, 224, 3],
      processingTimeMs: 10,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      filters: {
        sharpen: 0.5,
        medianFilter: 3,
      },
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    expect(calledOptions.filters).toEqual({
      sharpen: 0.5,
      medianFilter: 3,
    });
  });

  it('should validate median filter kernel size is odd', async () => {
    await expect(
      getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        filters: { medianFilter: 4 },
      })
    ).rejects.toThrow('Median filter kernel size must be an odd number');
  });
});

describe('Letterbox Resize Strategy', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should pass letterbox resize strategy', async () => {
    const mockResult: PixelDataResult = {
      data: [0.5],
      width: 640,
      height: 640,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [640, 640, 3],
      processingTimeMs: 10,
    };
    mockGetPixelData.mockResolvedValue(mockResult);

    await getPixelData({
      source: { type: 'url', value: 'https://example.com/image.jpg' },
      resize: {
        width: 640,
        height: 640,
        strategy: 'letterbox',
        letterboxColor: [114, 114, 114],
      },
    });

    const calledOptions = mockGetPixelData.mock.calls[0][0];
    expect(calledOptions.resize.strategy).toBe('letterbox');
    expect(calledOptions.resize.letterboxColor).toEqual([114, 114, 114]);
  });
});

describe('Image Statistics API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should get image statistics', async () => {
    const mockStats: ImageStatistics = {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      min: 0,
      max: 255,
      histogram: {
        red: new Array(256).fill(100),
        green: new Array(256).fill(100),
        blue: new Array(256).fill(100),
        luminance: new Array(256).fill(100),
      },
      processingTimeMs: 5,
    };
    mockGetImageStatistics.mockResolvedValue(mockStats);

    const stats = await getImageStatistics({
      type: 'url',
      value: 'https://example.com/image.jpg',
    });

    expect(stats.mean).toEqual([0.485, 0.456, 0.406]);
    expect(stats.std).toEqual([0.229, 0.224, 0.225]);
  });
});

describe('Image Metadata API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should get image metadata', async () => {
    const mockMetadata: ImageMetadata = {
      width: 1920,
      height: 1080,
      format: 'jpeg',
      colorSpace: 'sRGB',
      hasAlpha: false,
      bitsPerComponent: 8,
      fileSize: 1024000,
      processingTimeMs: 2,
    };
    mockGetImageMetadata.mockResolvedValue(mockMetadata);

    const metadata = await getImageMetadata({
      type: 'file',
      value: '/path/to/image.jpg',
    });

    expect(metadata.width).toBe(1920);
    expect(metadata.height).toBe(1080);
    expect(metadata.format).toBe('jpeg');
  });
});

describe('Image Validation API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should validate image successfully', async () => {
    const mockValidationResult: ImageValidationResult = {
      isValid: true,
      errors: [],
      metadata: {
        width: 500,
        height: 500,
        format: 'png',
        colorSpace: 'sRGB',
        hasAlpha: true,
        bitsPerComponent: 8,
        processingTimeMs: 1,
      },
    };
    mockValidateImage.mockResolvedValue(mockValidationResult);

    const result = await validateImage(
      { type: 'url', value: 'https://example.com/image.png' },
      { minWidth: 224, minHeight: 224, allowedFormats: ['png', 'jpeg'] }
    );

    expect(result.isValid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should return validation errors', async () => {
    const mockValidationResult: ImageValidationResult = {
      isValid: false,
      errors: ['Image width (100) is less than minimum (224)'],
      metadata: {
        width: 100,
        height: 100,
        format: 'png',
        colorSpace: 'sRGB',
        hasAlpha: false,
        bitsPerComponent: 8,
        processingTimeMs: 1,
      },
    };
    mockValidateImage.mockResolvedValue(mockValidationResult);

    const result = await validateImage(
      { type: 'url', value: 'https://example.com/small.png' },
      { minWidth: 224, minHeight: 224 }
    );

    expect(result.isValid).toBe(false);
    expect(result.errors).toContain(
      'Image width (100) is less than minimum (224)'
    );
  });
});

describe('Tensor to Image API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should convert tensor back to image', async () => {
    const mockImageResult: TensorToImageResult = {
      base64: 'data:image/png;base64,iVBORw0KGgo=',
      width: 224,
      height: 224,
      format: 'png',
      processingTimeMs: 3,
    };
    mockTensorToImage.mockResolvedValue(mockImageResult);

    const pixelData: PixelDataResult = {
      data: new Array(224 * 224 * 3).fill(0.5),
      width: 224,
      height: 224,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [224, 224, 3],
      processingTimeMs: 10,
    };

    const result = await tensorToImage(pixelData, {
      denormalization: { preset: 'imagenet' },
      format: 'png',
    });

    expect(result.base64).toContain('data:image/png;base64');
    expect(result.width).toBe(224);
  });
});

describe('Multi-Crop API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should perform five-crop operation', async () => {
    const mockFiveCropResult: MultiCropResult = {
      crops: [
        {
          position: 'top-left',
          flipped: false,
          result: {
            data: [0.5],
            width: 224,
            height: 224,
            channels: 3,
            colorFormat: 'rgb',
            dataLayout: 'hwc',
            shape: [224, 224, 3],
            processingTimeMs: 2,
          },
        },
        {
          position: 'top-right',
          flipped: false,
          result: {
            data: [0.5],
            width: 224,
            height: 224,
            channels: 3,
            colorFormat: 'rgb',
            dataLayout: 'hwc',
            shape: [224, 224, 3],
            processingTimeMs: 2,
          },
        },
        {
          position: 'bottom-left',
          flipped: false,
          result: {
            data: [0.5],
            width: 224,
            height: 224,
            channels: 3,
            colorFormat: 'rgb',
            dataLayout: 'hwc',
            shape: [224, 224, 3],
            processingTimeMs: 2,
          },
        },
        {
          position: 'bottom-right',
          flipped: false,
          result: {
            data: [0.5],
            width: 224,
            height: 224,
            channels: 3,
            colorFormat: 'rgb',
            dataLayout: 'hwc',
            shape: [224, 224, 3],
            processingTimeMs: 2,
          },
        },
        {
          position: 'center',
          flipped: false,
          result: {
            data: [0.5],
            width: 224,
            height: 224,
            channels: 3,
            colorFormat: 'rgb',
            dataLayout: 'hwc',
            shape: [224, 224, 3],
            processingTimeMs: 2,
          },
        },
      ],
      totalTimeMs: 15,
    };
    mockFiveCrop.mockResolvedValue(mockFiveCropResult);

    const result = await fiveCrop(
      { source: { type: 'url', value: 'https://example.com/image.jpg' } },
      { width: 224, height: 224 }
    );

    expect(result.crops).toHaveLength(5);
    expect(result.crops.map((c) => c.position)).toEqual([
      'top-left',
      'top-right',
      'bottom-left',
      'bottom-right',
      'center',
    ]);
  });

  it('should perform ten-crop operation', async () => {
    const mockTenCropResult: MultiCropResult = {
      crops: new Array(10).fill(null).map((_, i) => ({
        position: [
          'top-left',
          'top-right',
          'bottom-left',
          'bottom-right',
          'center',
        ][i % 5] as any,
        flipped: i >= 5,
        result: {
          data: [0.5],
          width: 224,
          height: 224,
          channels: 3,
          colorFormat: 'rgb',
          dataLayout: 'hwc',
          shape: [224, 224, 3],
          processingTimeMs: 2,
        },
      })),
      totalTimeMs: 25,
    };
    mockTenCrop.mockResolvedValue(mockTenCropResult);

    const result = await tenCrop(
      { source: { type: 'url', value: 'https://example.com/image.jpg' } },
      { width: 224, height: 224 }
    );

    expect(result.crops).toHaveLength(10);
    expect(result.crops.filter((c) => c.flipped)).toHaveLength(5);
  });
});

describe('Tensor Manipulation API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should extract a channel from pixel data', async () => {
    const mockChannelResult = {
      data: new Array(224 * 224).fill(0.5),
      shape: [224, 224, 1],
      processingTimeMs: 1,
    };
    mockExtractChannel.mockResolvedValue(mockChannelResult);

    const pixelData: PixelDataResult = {
      data: new Array(224 * 224 * 3).fill(0.5),
      width: 224,
      height: 224,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [224, 224, 3],
      processingTimeMs: 10,
    };

    const result = await extractChannel(pixelData, 0);

    expect(result.channels).toBe(1);
    expect(result.colorFormat).toBe('grayscale');
  });

  it('should throw error for invalid channel index', async () => {
    const pixelData: PixelDataResult = {
      data: [0.5],
      width: 1,
      height: 1,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [1, 1, 3],
      processingTimeMs: 1,
    };

    await expect(extractChannel(pixelData, 5)).rejects.toThrow(
      'Channel index 5 is out of bounds'
    );
  });

  it('should extract a patch from pixel data', async () => {
    const mockPatchResult = {
      data: new Array(32 * 32 * 3).fill(0.5),
      shape: [32, 32, 3],
      processingTimeMs: 1,
    };
    mockExtractPatch.mockResolvedValue(mockPatchResult);

    const pixelData: PixelDataResult = {
      data: new Array(224 * 224 * 3).fill(0.5),
      width: 224,
      height: 224,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [224, 224, 3],
      processingTimeMs: 10,
    };

    const result = await extractPatch(pixelData, {
      x: 10,
      y: 10,
      width: 32,
      height: 32,
    });

    expect(result.width).toBe(32);
    expect(result.height).toBe(32);
  });

  it('should throw error for patch outside bounds', async () => {
    const pixelData: PixelDataResult = {
      data: [0.5],
      width: 100,
      height: 100,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [100, 100, 3],
      processingTimeMs: 1,
    };

    await expect(
      extractPatch(pixelData, { x: 90, y: 90, width: 32, height: 32 })
    ).rejects.toThrow('Patch extends beyond image bounds');
  });

  it('should concatenate results to batch', async () => {
    const mockBatchResult = {
      data: new Array(3 * 224 * 224 * 3).fill(0.5),
      shape: [3, 224, 224, 3],
      processingTimeMs: 2,
    };
    mockConcatenateToBatch.mockResolvedValue(mockBatchResult);

    const results: PixelDataResult[] = [
      {
        data: new Array(224 * 224 * 3).fill(0.5),
        width: 224,
        height: 224,
        channels: 3,
        colorFormat: 'rgb',
        dataLayout: 'hwc',
        shape: [224, 224, 3],
        processingTimeMs: 10,
      },
      {
        data: new Array(224 * 224 * 3).fill(0.5),
        width: 224,
        height: 224,
        channels: 3,
        colorFormat: 'rgb',
        dataLayout: 'hwc',
        shape: [224, 224, 3],
        processingTimeMs: 10,
      },
      {
        data: new Array(224 * 224 * 3).fill(0.5),
        width: 224,
        height: 224,
        channels: 3,
        colorFormat: 'rgb',
        dataLayout: 'hwc',
        shape: [224, 224, 3],
        processingTimeMs: 10,
      },
    ];

    const result = await concatenateToBatch(results);

    expect(result.shape).toEqual([3, 224, 224, 3]);
  });

  it('should throw error when concatenating mismatched dimensions', async () => {
    const results: PixelDataResult[] = [
      {
        data: [0.5],
        width: 224,
        height: 224,
        channels: 3,
        colorFormat: 'rgb',
        dataLayout: 'hwc',
        shape: [224, 224, 3],
        processingTimeMs: 1,
      },
      {
        data: [0.5],
        width: 256,
        height: 256,
        channels: 3,
        colorFormat: 'rgb',
        dataLayout: 'hwc',
        shape: [256, 256, 3],
        processingTimeMs: 1,
      },
    ];

    await expect(concatenateToBatch(results)).rejects.toThrow(
      'All results must have same dimensions'
    );
  });

  it('should permute tensor dimensions', async () => {
    const mockPermuteResult = {
      data: new Array(224 * 224 * 3).fill(0.5),
      shape: [3, 224, 224],
      processingTimeMs: 1,
    };
    mockPermute.mockResolvedValue(mockPermuteResult);

    const pixelData: PixelDataResult = {
      data: new Array(224 * 224 * 3).fill(0.5),
      width: 224,
      height: 224,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [224, 224, 3],
      processingTimeMs: 10,
    };

    const result = await permute(pixelData, [2, 0, 1]);

    expect(result.shape).toEqual([3, 224, 224]);
  });
});

describe('Augmentation Function API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should apply augmentations and return base64', async () => {
    mockApplyAugmentations.mockResolvedValue({
      base64: 'data:image/png;base64,augmented',
      processingTimeMs: 5,
    });

    const result = await applyAugmentations(
      { type: 'url', value: 'https://example.com/image.jpg' },
      { horizontalFlip: true, brightness: 0.2 }
    );

    expect(result.base64).toContain('data:image/png;base64');
    expect(mockApplyAugmentations).toHaveBeenCalledTimes(1);
  });
});

describe('Cache Management API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should clear cache', async () => {
    mockClearCache.mockResolvedValue(undefined);

    await clearCache();

    expect(mockClearCache).toHaveBeenCalledTimes(1);
  });

  it('should get cache stats', async () => {
    mockGetCacheStats.mockResolvedValue({
      hitCount: 100,
      missCount: 20,
      size: 50,
      maxSize: 100,
    });

    const stats = await getCacheStats();

    expect(stats.hitCount).toBe(100);
    expect(stats.missCount).toBe(20);
    expect(stats.size).toBe(50);
  });
});
