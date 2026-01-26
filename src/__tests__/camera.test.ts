/**
 * Camera Frame Processing Tests
 */

import type { CameraPixelFormat, FrameOrientation } from '../types';
import { mockProcessCameraFrame, mockConvertYUVToRGB } from './setup';
import * as VisionUtils from '../index';

describe('Camera Frame Processing', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('processCameraFrame', () => {
    const createMockSource = (overrides = {}) => ({
      width: 640,
      height: 480,
      pixelFormat: 'yuv420' as CameraPixelFormat,
      bytesPerRow: 640,
      dataBase64: 'base64encodeddata...',
      ...overrides,
    });

    it('should process camera frame with default options', async () => {
      mockProcessCameraFrame.mockResolvedValue({
        tensor: Array(640 * 480 * 3).fill(0.5),
        shape: [480, 640, 3],
        width: 640,
        height: 480,
        processingTimeMs: 15.5,
      });

      const source = createMockSource();
      const result = await VisionUtils.processCameraFrame(source);

      expect(mockProcessCameraFrame).toHaveBeenCalledWith(source, {
        outputWidth: undefined,
        outputHeight: undefined,
        normalize: true,
        outputFormat: 'rgb',
        mean: undefined,
        std: undefined,
      });

      expect(result.tensor).toBeDefined();
      expect(result.shape).toEqual([480, 640, 3]);
      expect(result.processingTimeMs).toBeGreaterThan(0);
    });

    it('should resize to specified output dimensions', async () => {
      mockProcessCameraFrame.mockResolvedValue({
        tensor: Array(224 * 224 * 3).fill(0.5),
        shape: [224, 224, 3],
        width: 224,
        height: 224,
        processingTimeMs: 10,
      });

      const source = createMockSource();
      const result = await VisionUtils.processCameraFrame(source, {
        outputWidth: 224,
        outputHeight: 224,
      });

      expect(mockProcessCameraFrame).toHaveBeenCalledWith(source, {
        outputWidth: 224,
        outputHeight: 224,
        normalize: true,
        outputFormat: 'rgb',
        mean: undefined,
        std: undefined,
      });

      expect(result.width).toBe(224);
      expect(result.height).toBe(224);
    });

    it('should apply custom normalization (ImageNet)', async () => {
      mockProcessCameraFrame.mockResolvedValue({
        tensor: Array(224 * 224 * 3).fill(0),
        shape: [224, 224, 3],
        width: 224,
        height: 224,
        processingTimeMs: 12,
      });

      const source = createMockSource();
      await VisionUtils.processCameraFrame(source, {
        outputWidth: 224,
        outputHeight: 224,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
      });

      expect(mockProcessCameraFrame).toHaveBeenCalledWith(
        source,
        expect.objectContaining({
          mean: [0.485, 0.456, 0.406],
          std: [0.229, 0.224, 0.225],
        })
      );
    });

    it('should process grayscale output', async () => {
      mockProcessCameraFrame.mockResolvedValue({
        tensor: Array(224 * 224).fill(0.5),
        shape: [224, 224, 1],
        width: 224,
        height: 224,
        processingTimeMs: 8,
      });

      const source = createMockSource();
      const result = await VisionUtils.processCameraFrame(source, {
        outputWidth: 224,
        outputHeight: 224,
        outputFormat: 'grayscale',
      });

      expect(result.shape[2]).toBe(1);
    });

    it('should skip normalization when requested', async () => {
      mockProcessCameraFrame.mockResolvedValue({
        tensor: Array(224 * 224 * 3).fill(127), // Raw 0-255 values
        shape: [224, 224, 3],
        width: 224,
        height: 224,
        processingTimeMs: 7,
      });

      const source = createMockSource();
      await VisionUtils.processCameraFrame(source, {
        normalize: false,
      });

      expect(mockProcessCameraFrame).toHaveBeenCalledWith(
        source,
        expect.objectContaining({ normalize: false })
      );
    });

    it('should handle different pixel formats', async () => {
      const formats: CameraPixelFormat[] = [
        'yuv420',
        'yuv422',
        'nv12',
        'nv21',
        'bgra',
        'rgba',
        'rgb',
      ];

      for (const pixelFormat of formats) {
        mockProcessCameraFrame.mockResolvedValue({
          tensor: [],
          shape: [224, 224, 3],
          width: 224,
          height: 224,
          processingTimeMs: 10,
        });

        const source = createMockSource({ pixelFormat });
        await VisionUtils.processCameraFrame(source);

        expect(mockProcessCameraFrame).toHaveBeenLastCalledWith(
          expect.objectContaining({ pixelFormat }),
          expect.any(Object)
        );
      }
    });

    it('should handle frame orientation', async () => {
      const orientations: FrameOrientation[] = [0, 1, 2, 3];

      for (const orientation of orientations) {
        mockProcessCameraFrame.mockResolvedValue({
          tensor: [],
          shape: [224, 224, 3],
          width: 224,
          height: 224,
          processingTimeMs: 10,
        });

        const source = createMockSource({ orientation });
        await VisionUtils.processCameraFrame(source);

        expect(mockProcessCameraFrame).toHaveBeenLastCalledWith(
          expect.objectContaining({ orientation }),
          expect.any(Object)
        );
      }
    });

    it('should reject for invalid source', async () => {
      const invalidSource = createMockSource({ dataBase64: undefined });
      await expect(
        VisionUtils.processCameraFrame(invalidSource as any)
      ).rejects.toThrow();
    });

    it('should reject for unsupported pixel format', async () => {
      mockProcessCameraFrame.mockRejectedValue(
        new Error('Unsupported pixel format: unknown')
      );

      const source = createMockSource({ pixelFormat: 'unknown' as any });
      await expect(VisionUtils.processCameraFrame(source)).rejects.toThrow(
        'Unsupported pixel format'
      );
    });
  });

  describe('convertYUVToRGB', () => {
    it('should convert YUV planes to RGB', async () => {
      mockConvertYUVToRGB.mockResolvedValue({
        data: Array(640 * 480 * 3).fill(128),
        width: 640,
        height: 480,
      });

      const result = await VisionUtils.convertYUVToRGB(
        'base64Y...',
        'base64U...',
        'base64V...',
        640,
        480,
        'yuv420'
      );

      expect(result.data).toBeDefined();
      expect(result.width).toBe(640);
      expect(result.height).toBe(480);
    });

    it('should reject for invalid dimensions', async () => {
      await expect(
        VisionUtils.convertYUVToRGB('y', 'u', 'v', 0, 480, 'yuv420')
      ).rejects.toThrow();

      await expect(
        VisionUtils.convertYUVToRGB('y', 'u', 'v', 640, -1, 'yuv420')
      ).rejects.toThrow();
    });

    it('should handle NV12 format', async () => {
      mockConvertYUVToRGB.mockResolvedValue({
        data: Array(640 * 480 * 3).fill(128),
        width: 640,
        height: 480,
      });

      const result = await VisionUtils.convertYUVToRGB(
        'base64Y...',
        'base64UV...',
        '',
        640,
        480,
        'nv12'
      );

      expect(result.data).toBeDefined();
    });
  });
});

describe('Camera Frame Processing - Edge Cases', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should handle very small frames (1x1)', async () => {
    mockProcessCameraFrame.mockResolvedValue({
      tensor: [0.5, 0.5, 0.5],
      shape: [1, 1, 3],
      width: 1,
      height: 1,
      processingTimeMs: 0.1,
    });

    const source = {
      width: 1,
      height: 1,
      pixelFormat: 'rgb' as CameraPixelFormat,
      bytesPerRow: 3,
      dataBase64: 'AQID', // 3 bytes: 1, 2, 3
    };

    const result = await VisionUtils.processCameraFrame(source);
    expect(result.shape).toEqual([1, 1, 3]);
  });

  it('should handle large frames (4K)', async () => {
    mockProcessCameraFrame.mockResolvedValue({
      tensor: [],
      shape: [2160, 3840, 3],
      width: 3840,
      height: 2160,
      processingTimeMs: 100,
    });

    const source = {
      width: 3840,
      height: 2160,
      pixelFormat: 'yuv420' as CameraPixelFormat,
      bytesPerRow: 3840,
      dataBase64: 'largeBinaryData...',
    };

    const result = await VisionUtils.processCameraFrame(source);
    expect(result.width).toBe(3840);
    expect(result.height).toBe(2160);
  });

  it('should handle non-standard aspect ratios', async () => {
    mockProcessCameraFrame.mockResolvedValue({
      tensor: [],
      shape: [1080, 1920, 3],
      width: 1920,
      height: 1080,
      processingTimeMs: 50,
    });

    const source = {
      width: 1920,
      height: 1080,
      pixelFormat: 'nv21' as CameraPixelFormat,
      bytesPerRow: 1920,
      dataBase64: 'data...',
    };

    const result = await VisionUtils.processCameraFrame(source);
    expect(result.width).toBe(1920);
    expect(result.height).toBe(1080);
  });

  it('should handle timestamp metadata', async () => {
    mockProcessCameraFrame.mockResolvedValue({
      tensor: [],
      shape: [480, 640, 3],
      width: 640,
      height: 480,
      processingTimeMs: 10,
    });

    const source = {
      width: 640,
      height: 480,
      pixelFormat: 'yuv420' as CameraPixelFormat,
      bytesPerRow: 640,
      dataBase64: 'data...',
      timestamp: Date.now(),
    };

    await VisionUtils.processCameraFrame(source);

    expect(mockProcessCameraFrame).toHaveBeenCalledWith(
      expect.objectContaining({ timestamp: expect.any(Number) }),
      expect.any(Object)
    );
  });

  it('should handle extreme normalization values', async () => {
    mockProcessCameraFrame.mockResolvedValue({
      tensor: [],
      shape: [224, 224, 3],
      width: 224,
      height: 224,
      processingTimeMs: 10,
    });

    const source = {
      width: 640,
      height: 480,
      pixelFormat: 'rgb' as CameraPixelFormat,
      bytesPerRow: 640 * 3,
      dataBase64: 'data...',
    };

    // Extreme normalization values
    await VisionUtils.processCameraFrame(source, {
      mean: [0, 0, 0],
      std: [0.001, 0.001, 0.001], // Very small std
    });

    expect(mockProcessCameraFrame).toHaveBeenCalled();
  });
});

describe('Camera Frame Processing - Performance Metadata', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should report processing time', async () => {
    mockProcessCameraFrame.mockResolvedValue({
      tensor: [],
      shape: [224, 224, 3],
      width: 224,
      height: 224,
      processingTimeMs: 15.5,
    });

    const source = {
      width: 640,
      height: 480,
      pixelFormat: 'yuv420' as CameraPixelFormat,
      bytesPerRow: 640,
      dataBase64: 'data...',
    };

    const result = await VisionUtils.processCameraFrame(source);

    expect(result.processingTimeMs).toBe(15.5);
    expect(typeof result.processingTimeMs).toBe('number');
  });

  it('should include correct shape dimensions', async () => {
    mockProcessCameraFrame.mockResolvedValue({
      tensor: Array(224 * 224 * 3).fill(0),
      shape: [224, 224, 3],
      width: 224,
      height: 224,
      processingTimeMs: 10,
    });

    const source = {
      width: 640,
      height: 480,
      pixelFormat: 'yuv420' as CameraPixelFormat,
      bytesPerRow: 640,
      dataBase64: 'data...',
    };

    const result = await VisionUtils.processCameraFrame(source, {
      outputWidth: 224,
      outputHeight: 224,
    });

    expect(result.shape).toEqual([224, 224, 3]);
    expect(result.tensor.length).toBe(224 * 224 * 3);
  });
});
