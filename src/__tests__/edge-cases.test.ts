import { mockGetPixelData, mockBatchGetPixelData } from './jest.setup';
import { getPixelData, batchGetPixelData } from '../index';
import type { PixelDataResult } from '../types';

describe('Edge Cases', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getPixelData edge cases', () => {
    describe('minimal valid input', () => {
      it('should work with only required source field', async () => {
        mockGetPixelData.mockResolvedValue({
          data: [0.5],
          width: 1,
          height: 1,
          channels: 3,
          dataLayout: 'hwc',
          processingTimeMs: 1,
        });

        const result = await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
        });

        expect(result).toBeDefined();
        expect(mockGetPixelData).toHaveBeenCalledTimes(1);
      });
    });

    describe('special URL formats', () => {
      beforeEach(() => {
        mockGetPixelData.mockResolvedValue({
          data: [],
          width: 100,
          height: 100,
          channels: 3,
          dataLayout: 'hwc',
          processingTimeMs: 1,
        });
      });

      it('should handle URL with query parameters', async () => {
        await getPixelData({
          source: {
            type: 'url',
            value: 'https://example.com/image.jpg?width=100&height=100',
          },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.source.value).toBe(
          'https://example.com/image.jpg?width=100&height=100'
        );
      });

      it('should handle URL with hash fragment', async () => {
        await getPixelData({
          source: {
            type: 'url',
            value: 'https://example.com/image.jpg#section',
          },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.source.value).toContain('#section');
      });

      it('should handle URL with encoded characters', async () => {
        await getPixelData({
          source: {
            type: 'url',
            value: 'https://example.com/path%20with%20spaces/image.jpg',
          },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.source.value).toContain('%20');
      });

      it('should handle URL with authentication', async () => {
        await getPixelData({
          source: {
            type: 'url',
            value: 'https://user:pass@example.com/image.jpg',
          },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.source.value).toContain('user:pass@');
      });

      it('should handle URL with port number', async () => {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com:8080/image.jpg' },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.source.value).toContain(':8080');
      });

      it('should handle data URL with base64 source type', async () => {
        const dataUrl =
          'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
        await getPixelData({
          source: { type: 'base64', value: dataUrl },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.source.value).toBe(dataUrl);
      });
    });

    describe('file paths', () => {
      beforeEach(() => {
        mockGetPixelData.mockResolvedValue({
          data: [],
          width: 100,
          height: 100,
          channels: 3,
          dataLayout: 'hwc',
          processingTimeMs: 1,
        });
      });

      it('should handle absolute path', async () => {
        await getPixelData({
          source: { type: 'file', value: '/Users/test/Documents/image.jpg' },
        });

        expect(mockGetPixelData).toHaveBeenCalled();
      });

      it('should handle path with file:// prefix', async () => {
        await getPixelData({
          source: { type: 'file', value: 'file:///Users/test/image.jpg' },
        });

        expect(mockGetPixelData).toHaveBeenCalled();
      });

      it('should handle path with spaces', async () => {
        await getPixelData({
          source: { type: 'file', value: '/path/with spaces/image.jpg' },
        });

        expect(mockGetPixelData).toHaveBeenCalled();
      });

      it('should handle path with special characters', async () => {
        await getPixelData({
          source: {
            type: 'file',
            value: '/path/with-special_chars.123/image.jpg',
          },
        });

        expect(mockGetPixelData).toHaveBeenCalled();
      });
    });

    describe('base64 data', () => {
      beforeEach(() => {
        mockGetPixelData.mockResolvedValue({
          data: [],
          width: 100,
          height: 100,
          channels: 3,
          dataLayout: 'hwc',
          processingTimeMs: 1,
        });
      });

      it('should handle base64 with data URL prefix', async () => {
        await getPixelData({
          source: {
            type: 'base64',
            value: 'data:image/png;base64,iVBORw0KGgo=',
          },
        });

        expect(mockGetPixelData).toHaveBeenCalled();
      });

      it('should handle raw base64 without prefix', async () => {
        await getPixelData({
          source: { type: 'base64', value: 'iVBORw0KGgoAAAANSUhEUg==' },
        });

        expect(mockGetPixelData).toHaveBeenCalled();
      });

      it('should handle JPEG base64', async () => {
        await getPixelData({
          source: {
            type: 'base64',
            value: 'data:image/jpeg;base64,/9j/4AAQSkZJRg==',
          },
        });

        expect(mockGetPixelData).toHaveBeenCalled();
      });
    });

    describe('resize strategies', () => {
      beforeEach(() => {
        mockGetPixelData.mockResolvedValue({
          data: [],
          width: 224,
          height: 224,
          channels: 3,
          dataLayout: 'hwc',
          processingTimeMs: 1,
        });
      });

      it('should default to cover strategy', async () => {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: 224, height: 224 },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.resize.strategy).toBe('cover');
      });

      it('should pass contain strategy', async () => {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: 224, height: 224, strategy: 'contain' },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.resize.strategy).toBe('contain');
      });

      it('should pass stretch strategy', async () => {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: 224, height: 224, strategy: 'stretch' },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.resize.strategy).toBe('stretch');
      });
    });

    describe('extreme dimensions', () => {
      it('should handle 1x1 pixel output', async () => {
        mockGetPixelData.mockResolvedValue({
          data: [0.5, 0.5, 0.5],
          width: 1,
          height: 1,
          channels: 3,
          dataLayout: 'hwc',
          processingTimeMs: 1,
        });

        const result = await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: 1, height: 1 },
        });

        expect(result.width).toBe(1);
        expect(result.height).toBe(1);
      });

      it('should handle very large dimensions', async () => {
        const largeData = new Array(1000 * 1000 * 3).fill(0.5);
        mockGetPixelData.mockResolvedValue({
          data: largeData,
          width: 1000,
          height: 1000,
          channels: 3,
          dataLayout: 'hwc',
          processingTimeMs: 100,
        });

        const result = await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width: 1000, height: 1000 },
        });

        expect(result.data.length).toBe(1000 * 1000 * 3);
      });
    });

    describe('data layout combinations', () => {
      it('should return correct shape for hwc layout', async () => {
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
          dataLayout: 'hwc',
          resize: { width: 224, height: 224 },
        });

        expect(result.dataLayout).toBe('hwc');
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
          dataLayout: 'chw',
          resize: { width: 224, height: 224 },
        });

        expect(result.dataLayout).toBe('chw');
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
          dataLayout: 'nhwc',
          resize: { width: 224, height: 224 },
        });

        expect(result.dataLayout).toBe('nhwc');
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
          dataLayout: 'nchw',
          resize: { width: 224, height: 224 },
        });

        expect(result.dataLayout).toBe('nchw');
      });
    });

    describe('combined options', () => {
      it('should handle ROI + resize', async () => {
        mockGetPixelData.mockResolvedValue({
          data: [],
          width: 224,
          height: 224,
          channels: 3,
          dataLayout: 'hwc',
          processingTimeMs: 1,
        });

        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          roi: { x: 100, y: 100, width: 200, height: 200 },
          resize: { width: 224, height: 224 },
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.roi).toBeDefined();
        expect(calledOptions.resize).toBeDefined();
      });

      it('should handle all options together', async () => {
        mockGetPixelData.mockResolvedValue({
          data: [],
          width: 224,
          height: 224,
          channels: 3,
          dataLayout: 'nchw',
          processingTimeMs: 1,
        });

        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          colorFormat: 'bgr',
          normalization: {
            preset: 'custom',
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
          },
          resize: { width: 224, height: 224, strategy: 'contain' },
          roi: { x: 0, y: 0, width: 100, height: 100 },
          dataLayout: 'nchw',
        });

        const calledOptions = mockGetPixelData.mock.calls[0][0];
        expect(calledOptions.colorFormat).toBe('bgr');
        expect(calledOptions.normalization.preset).toBe('custom');
        expect(calledOptions.resize.strategy).toBe('contain');
        expect(calledOptions.roi).toBeDefined();
        expect(calledOptions.dataLayout).toBe('nchw');
      });
    });
  });

  describe('batchGetPixelData edge cases', () => {
    it('should handle empty array', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [],
        totalTimeMs: 0,
      });

      const result = await batchGetPixelData([]);
      expect(result.results).toHaveLength(0);
    });

    it('should handle single image in batch', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [
          {
            data: [0.5],
            width: 224,
            height: 224,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 10,
          },
        ],
        totalTimeMs: 10,
      });

      const result = await batchGetPixelData([
        { source: { type: 'url', value: 'https://example.com/image.jpg' } },
      ]);

      expect(result.results).toHaveLength(1);
    });

    it('should handle large batch', async () => {
      const batchSize = 100;
      const mockResults: PixelDataResult[] = Array(batchSize)
        .fill(null)
        .map((_, i) => ({
          data: [i],
          width: 224,
          height: 224,
          channels: 3,
          colorFormat: 'rgb' as const,
          dataLayout: 'hwc' as const,
          shape: [224, 224, 3],
          processingTimeMs: 1,
        }));

      mockBatchGetPixelData.mockResolvedValue({
        results: mockResults,
        totalTimeMs: 100,
      });

      const options = Array(batchSize)
        .fill(null)
        .map((_, i) => ({
          source: {
            type: 'url' as const,
            value: `https://example.com/image${i}.jpg`,
          },
        }));

      const result = await batchGetPixelData(options);
      expect(result.results).toHaveLength(batchSize);
    });

    it('should handle mixed source types', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [
          {
            data: [],
            width: 100,
            height: 100,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 1,
          },
          {
            data: [],
            width: 100,
            height: 100,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 1,
          },
          {
            data: [],
            width: 100,
            height: 100,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 1,
          },
        ],
        totalTimeMs: 5,
      });

      const result = await batchGetPixelData([
        { source: { type: 'url', value: 'https://example.com/image.jpg' } },
        { source: { type: 'file', value: '/path/to/image.png' } },
        { source: { type: 'base64', value: 'iVBORw0KGgo=' } },
      ]);

      expect(result.results).toHaveLength(3);
    });

    it('should handle different options per image', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [
          {
            data: [],
            width: 224,
            height: 224,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 1,
          },
          {
            data: [],
            width: 299,
            height: 299,
            channels: 3,
            dataLayout: 'chw',
            processingTimeMs: 1,
          },
        ],
        totalTimeMs: 5,
      });

      const result = await batchGetPixelData([
        {
          source: { type: 'url', value: 'https://example.com/image1.jpg' },
          resize: { width: 224, height: 224 },
          dataLayout: 'hwc',
        },
        {
          source: { type: 'url', value: 'https://example.com/image2.jpg' },
          resize: { width: 299, height: 299 },
          dataLayout: 'chw',
        },
      ]);

      expect(result.results).toHaveLength(2);
    });

    it('should respect concurrency limit', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [],
        totalTimeMs: 0,
      });

      await batchGetPixelData(
        [{ source: { type: 'url', value: 'https://example.com/image.jpg' } }],
        { concurrency: 1 }
      );

      const calledBatchOptions = mockBatchGetPixelData.mock.calls[0][1];
      expect(calledBatchOptions.concurrency).toBe(1);
    });

    it('should handle high concurrency', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [],
        totalTimeMs: 0,
      });

      await batchGetPixelData(
        [{ source: { type: 'url', value: 'https://example.com/image.jpg' } }],
        { concurrency: 16 }
      );

      const calledBatchOptions = mockBatchGetPixelData.mock.calls[0][1];
      expect(calledBatchOptions.concurrency).toBe(16);
    });
  });
});
