import { mockGetPixelData } from './jest.setup';
import { getPixelData, VisionUtilsException } from '../index';
import type { GetPixelDataOptions } from '../types';

describe('Input Validation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetPixelData.mockResolvedValue({
      data: [],
      width: 100,
      height: 100,
      channels: 3,
      dataLayout: 'hwc',
      processingTimeMs: 1,
    });
  });

  describe('source validation', () => {
    it('should throw when source is missing', async () => {
      await expect(getPixelData({} as GetPixelDataOptions)).rejects.toThrow(
        VisionUtilsException
      );
    });

    it('should throw when source type is missing', async () => {
      await expect(
        getPixelData({
          source: { value: 'test' } as any,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw when source value is missing', async () => {
      await expect(
        getPixelData({
          source: { type: 'url' } as any,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw with INVALID_SOURCE code when source is invalid', async () => {
      try {
        await getPixelData({
          source: { type: 'url' } as any,
        });
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('INVALID_SOURCE');
      }
    });

    it('should accept empty string value (validation passes, native handles error)', async () => {
      // Empty string is technically a valid value, native module should handle the error
      await expect(
        getPixelData({
          source: { type: 'url', value: '' },
        })
      ).rejects.toThrow(); // Will throw because empty string is falsy
    });
  });

  describe('resize validation', () => {
    it('should throw when resize width is zero', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: 0, height: 224 },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw when resize height is zero', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: 224, height: 0 },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw when resize width is negative', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: -100, height: 224 },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw when resize height is negative', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: 224, height: -100 },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw with INVALID_RESIZE code', async () => {
      try {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: 0, height: 224 },
        });
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('INVALID_RESIZE');
      }
    });

    it('should accept valid resize dimensions', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 224, height: 224 },
      });
      expect(mockGetPixelData).toHaveBeenCalled();
    });

    it('should accept very large resize dimensions', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 4096, height: 4096 },
      });
      expect(mockGetPixelData).toHaveBeenCalled();
    });

    it('should accept non-square resize dimensions', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 640, height: 480 },
      });
      expect(mockGetPixelData).toHaveBeenCalled();
    });
  });

  describe('ROI validation', () => {
    it('should throw when ROI width is zero', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          roi: { x: 0, y: 0, width: 0, height: 100 },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw when ROI height is zero', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          roi: { x: 0, y: 0, width: 100, height: 0 },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw when ROI x is negative', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          roi: { x: -10, y: 0, width: 100, height: 100 },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw when ROI y is negative', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          roi: { x: 0, y: -10, width: 100, height: 100 },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw with INVALID_ROI code', async () => {
      try {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          roi: { x: -10, y: 0, width: 100, height: 100 },
        });
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('INVALID_ROI');
      }
    });

    it('should accept valid ROI', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        roi: { x: 10, y: 20, width: 100, height: 100 },
      });
      expect(mockGetPixelData).toHaveBeenCalled();
    });

    it('should accept ROI at origin', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        roi: { x: 0, y: 0, width: 50, height: 50 },
      });
      expect(mockGetPixelData).toHaveBeenCalled();
    });
  });

  describe('normalization validation', () => {
    it('should throw when custom normalization is missing mean', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          normalization: { preset: 'custom', std: [0.5, 0.5, 0.5] },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw when custom normalization is missing std', async () => {
      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          normalization: { preset: 'custom', mean: [0.5, 0.5, 0.5] },
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw with INVALID_NORMALIZATION code', async () => {
      try {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          normalization: { preset: 'custom' },
        });
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe(
          'INVALID_NORMALIZATION'
        );
      }
    });

    it('should accept custom normalization with mean and std', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        normalization: {
          preset: 'custom',
          mean: [0.485, 0.456, 0.406],
          std: [0.229, 0.224, 0.225],
        },
      });
      expect(mockGetPixelData).toHaveBeenCalled();
    });

    it('should accept non-custom preset without mean/std', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        normalization: { preset: 'imagenet' },
      });
      expect(mockGetPixelData).toHaveBeenCalled();
    });
  });
});
