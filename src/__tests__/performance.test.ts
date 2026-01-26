import { mockGetPixelData, mockBatchGetPixelData } from './jest.setup';
import { getPixelData, batchGetPixelData } from '../index';
import type { PixelDataResult } from '../types';

describe('Performance & Timing', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('processingTimeMs', () => {
    it('should include processingTimeMs in result', async () => {
      mockGetPixelData.mockResolvedValue({
        data: [],
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'hwc',
        processingTimeMs: 15.5,
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
      });

      expect(result.processingTimeMs).toBeDefined();
      expect(typeof result.processingTimeMs).toBe('number');
    });

    it('should handle very small processing times', async () => {
      mockGetPixelData.mockResolvedValue({
        data: [],
        width: 10,
        height: 10,
        channels: 3,
        dataLayout: 'hwc',
        processingTimeMs: 0.123,
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/tiny.jpg' },
      });

      expect(result.processingTimeMs).toBeLessThan(1);
    });

    it('should handle large processing times', async () => {
      mockGetPixelData.mockResolvedValue({
        data: [],
        width: 4000,
        height: 4000,
        channels: 3,
        dataLayout: 'hwc',
        processingTimeMs: 5000,
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/huge.jpg' },
      });

      expect(result.processingTimeMs).toBeGreaterThanOrEqual(0);
    });
  });

  describe('batch totalTimeMs', () => {
    it('should include totalTimeMs in batch result', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [
          {
            data: [],
            width: 224,
            height: 224,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 10,
          },
          {
            data: [],
            width: 224,
            height: 224,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 12,
          },
        ],
        totalTimeMs: 15,
      });

      const result = await batchGetPixelData([
        { source: { type: 'url', value: 'https://example.com/image1.jpg' } },
        { source: { type: 'url', value: 'https://example.com/image2.jpg' } },
      ]);

      expect(result.totalTimeMs).toBeDefined();
      expect(typeof result.totalTimeMs).toBe('number');
    });

    it('should have totalTimeMs less than sum of individual times due to parallelism', async () => {
      const individualTime = 100;
      mockBatchGetPixelData.mockResolvedValue({
        results: [
          {
            data: [],
            width: 224,
            height: 224,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: individualTime,
          },
          {
            data: [],
            width: 224,
            height: 224,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: individualTime,
          },
          {
            data: [],
            width: 224,
            height: 224,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: individualTime,
          },
          {
            data: [],
            width: 224,
            height: 224,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: individualTime,
          },
        ],
        totalTimeMs: 150, // Faster due to parallelism
      });

      const result = await batchGetPixelData(
        [
          { source: { type: 'url', value: 'https://example.com/image1.jpg' } },
          { source: { type: 'url', value: 'https://example.com/image2.jpg' } },
          { source: { type: 'url', value: 'https://example.com/image3.jpg' } },
          { source: { type: 'url', value: 'https://example.com/image4.jpg' } },
        ],
        { concurrency: 4 }
      );

      const sumOfIndividual = result.results.reduce(
        (sum, r) => sum + ((r as PixelDataResult).processingTimeMs || 0),
        0
      );
      expect(result.totalTimeMs).toBeLessThan(sumOfIndividual);
    });
  });

  describe('concurrency impact', () => {
    it('should respect concurrency setting', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [],
        totalTimeMs: 100,
      });

      await batchGetPixelData(
        [
          { source: { type: 'url', value: 'https://example.com/1.jpg' } },
          { source: { type: 'url', value: 'https://example.com/2.jpg' } },
        ],
        { concurrency: 2 }
      );

      expect(mockBatchGetPixelData.mock.calls[0][1].concurrency).toBe(2);
    });

    it('should use default concurrency of 4', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [],
        totalTimeMs: 100,
      });

      await batchGetPixelData([
        { source: { type: 'url', value: 'https://example.com/image.jpg' } },
      ]);

      expect(mockBatchGetPixelData.mock.calls[0][1].concurrency).toBe(4);
    });
  });
});

describe('Result Shape Validation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('shape property', () => {
    it('should include shape in result when provided', async () => {
      mockGetPixelData.mockResolvedValue({
        data: new Array(224 * 224 * 3).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'hwc',
        processingTimeMs: 1,
        shape: [224, 224, 3],
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 224, height: 224 },
        dataLayout: 'hwc',
      });

      expect(result.shape).toEqual([224, 224, 3]);
    });

    it('should return correct shape for chw layout', async () => {
      mockGetPixelData.mockResolvedValue({
        data: new Array(3 * 224 * 224).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'chw',
        processingTimeMs: 1,
        shape: [3, 224, 224],
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 224, height: 224 },
        dataLayout: 'chw',
      });

      expect(result.shape).toEqual([3, 224, 224]);
    });

    it('should return correct shape for nhwc layout', async () => {
      mockGetPixelData.mockResolvedValue({
        data: new Array(1 * 224 * 224 * 3).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'nhwc',
        processingTimeMs: 1,
        shape: [1, 224, 224, 3],
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 224, height: 224 },
        dataLayout: 'nhwc',
      });

      expect(result.shape).toEqual([1, 224, 224, 3]);
    });

    it('should return correct shape for nchw layout', async () => {
      mockGetPixelData.mockResolvedValue({
        data: new Array(1 * 3 * 224 * 224).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'nchw',
        processingTimeMs: 1,
        shape: [1, 3, 224, 224],
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 224, height: 224 },
        dataLayout: 'nchw',
      });

      expect(result.shape).toEqual([1, 3, 224, 224]);
    });
  });

  describe('data array size validation', () => {
    it('data length should match product of shape dimensions', async () => {
      const shape = [224, 224, 3];
      const expectedLength = shape.reduce((a, b) => a * b, 1);

      mockGetPixelData.mockResolvedValue({
        data: new Array(expectedLength).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'hwc',
        processingTimeMs: 1,
        shape,
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 224, height: 224 },
      });

      expect(result.data.length).toBe(expectedLength);
    });

    it('data length should match width * height * channels', async () => {
      const width = 100;
      const height = 150;
      const channels = 4;
      const expectedLength = width * height * channels;

      mockGetPixelData.mockResolvedValue({
        data: new Array(expectedLength).fill(0),
        width,
        height,
        channels,
        dataLayout: 'hwc',
        processingTimeMs: 1,
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width, height },
        colorFormat: 'rgba',
      });

      expect(result.data.length).toBe(width * height * channels);
    });
  });
});
