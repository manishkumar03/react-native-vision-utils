import { mockGetPixelData, mockBatchGetPixelData } from './jest.setup';
import { getPixelData, batchGetPixelData } from '../index';
import type {
  GetPixelDataOptions,
  PixelDataResult,
  BatchResult,
} from '../types';

describe('getPixelData', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('basic functionality', () => {
    it('should call native module with correct options', async () => {
      const mockResult: PixelDataResult = {
        data: new Array(224 * 224 * 3).fill(0.5),
        width: 224,
        height: 224,
        channels: 3,
        colorFormat: 'rgb',
        dataLayout: 'hwc',
        shape: [224, 224, 3],
        processingTimeMs: 10.5,
      };
      mockGetPixelData.mockResolvedValue(mockResult);

      const options: GetPixelDataOptions = {
        source: { type: 'url', value: 'https://example.com/image.jpg' },
      };

      const result = await getPixelData(options);

      expect(mockGetPixelData).toHaveBeenCalledTimes(1);
      expect(result).toEqual(mockResult);
    });

    it('should apply default options', async () => {
      const mockResult: PixelDataResult = {
        data: [],
        width: 100,
        height: 100,
        channels: 3,
        colorFormat: 'rgb',
        dataLayout: 'hwc',
        shape: [100, 100, 3],
        processingTimeMs: 5,
      };
      mockGetPixelData.mockResolvedValue(mockResult);

      await getPixelData({
        source: { type: 'file', value: '/path/to/image.png' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.colorFormat).toBe('rgb');
      expect(calledOptions.dataLayout).toBe('hwc');
      expect(calledOptions.outputFormat).toBe('array');
      expect(calledOptions.normalization.preset).toBe('scale');
    });
  });

  describe('image sources', () => {
    const mockResult: PixelDataResult = {
      data: [0.5],
      width: 1,
      height: 1,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [1, 1, 3],
      processingTimeMs: 1,
    };

    beforeEach(() => {
      mockGetPixelData.mockResolvedValue(mockResult);
    });

    it('should handle URL source', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.source.type).toBe('url');
      expect(calledOptions.source.value).toBe('https://example.com/image.jpg');
    });

    it('should handle file source', async () => {
      await getPixelData({
        source: { type: 'file', value: '/path/to/image.png' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.source.type).toBe('file');
    });

    it('should handle base64 source', async () => {
      await getPixelData({
        source: { type: 'base64', value: 'data:image/png;base64,iVBORw0KGgo=' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.source.type).toBe('base64');
    });

    it('should handle asset source', async () => {
      await getPixelData({
        source: { type: 'asset', value: 'images/test.png' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.source.type).toBe('asset');
    });

    it('should handle photo library source', async () => {
      await getPixelData({
        source: { type: 'photoLibrary', value: 'ph://ASSET123' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.source.type).toBe('photoLibrary');
    });
  });

  describe('color formats', () => {
    it('should pass rgb color format', async () => {
      mockGetPixelData.mockResolvedValue({
        data: [],
        width: 1,
        height: 1,
        channels: 3,
        dataLayout: 'hwc',
        processingTimeMs: 1,
      });

      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        colorFormat: 'rgb',
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.colorFormat).toBe('rgb');
    });

    it('should pass rgba color format', async () => {
      mockGetPixelData.mockResolvedValue({
        data: [],
        width: 1,
        height: 1,
        channels: 4,
        dataLayout: 'hwc',
        processingTimeMs: 1,
      });

      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        colorFormat: 'rgba',
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.colorFormat).toBe('rgba');
    });

    it('should pass grayscale color format', async () => {
      mockGetPixelData.mockResolvedValue({
        data: [],
        width: 1,
        height: 1,
        channels: 1,
        dataLayout: 'hwc',
        processingTimeMs: 1,
      });

      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        colorFormat: 'grayscale',
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.colorFormat).toBe('grayscale');
    });
  });

  describe('resize options', () => {
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

    it('should pass resize dimensions', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 224, height: 224 },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.resize.width).toBe(224);
      expect(calledOptions.resize.height).toBe(224);
    });

    it('should pass resize strategy', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width: 224, height: 224, strategy: 'contain' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.resize.strategy).toBe('contain');
    });
  });

  describe('normalization', () => {
    beforeEach(() => {
      mockGetPixelData.mockResolvedValue({
        data: [],
        width: 1,
        height: 1,
        channels: 3,
        dataLayout: 'hwc',
        processingTimeMs: 1,
      });
    });

    it('should pass ImageNet normalization preset', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        normalization: { preset: 'imagenet' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.normalization.preset).toBe('imagenet');
    });

    it('should pass TensorFlow normalization preset', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        normalization: { preset: 'tensorflow' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.normalization.preset).toBe('tensorflow');
    });

    it('should pass custom normalization values', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        normalization: {
          preset: 'custom',
          mean: [0.5, 0.5, 0.5],
          std: [0.5, 0.5, 0.5],
        },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.normalization.preset).toBe('custom');
      expect(calledOptions.normalization.mean).toEqual([0.5, 0.5, 0.5]);
      expect(calledOptions.normalization.std).toEqual([0.5, 0.5, 0.5]);
    });
  });

  describe('data layout', () => {
    beforeEach(() => {
      mockGetPixelData.mockResolvedValue({
        data: [],
        width: 1,
        height: 1,
        channels: 3,
        dataLayout: 'chw',
        processingTimeMs: 1,
      });
    });

    it('should pass chw data layout', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        dataLayout: 'chw',
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.dataLayout).toBe('chw');
    });

    it('should pass nchw data layout', async () => {
      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        dataLayout: 'nchw',
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.dataLayout).toBe('nchw');
    });
  });

  describe('output format conversion', () => {
    const mockNativeResult: PixelDataResult = {
      data: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
      width: 2,
      height: 1,
      channels: 3,
      colorFormat: 'rgb',
      dataLayout: 'hwc',
      shape: [1, 2, 3],
      processingTimeMs: 1,
    };

    beforeEach(() => {
      mockGetPixelData.mockResolvedValue({ ...mockNativeResult });
    });

    it('should return number[] for outputFormat array (default)', async () => {
      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        outputFormat: 'array',
      });

      expect(Array.isArray(result.data)).toBe(true);
      expect(result.data).toEqual([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    });

    it('should return Float32Array for outputFormat float32Array', async () => {
      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        outputFormat: 'float32Array',
      });

      expect(result.data).toBeInstanceOf(Float32Array);
      expect(Array.from(result.data as Float32Array)).toEqual([
        expect.closeTo(0.1, 5),
        expect.closeTo(0.2, 5),
        expect.closeTo(0.3, 5),
        expect.closeTo(0.4, 5),
        expect.closeTo(0.5, 5),
        expect.closeTo(0.6, 5),
      ]);
    });

    it('should return Uint8Array for outputFormat uint8Array', async () => {
      // Test with normalized values [0-1] - these get scaled to 0-255
      mockGetPixelData.mockResolvedValue({
        ...mockNativeResult,
        data: [0, 0.5, 1.0, 0.5, 1.0, 0],
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        outputFormat: 'uint8Array',
      });

      expect(result.data).toBeInstanceOf(Uint8Array);
      // Values should be scaled from [0-1] to [0-255], clamped and rounded
      expect(Array.from(result.data as Uint8Array)).toEqual([
        0, 128, 255, 128, 255, 0,
      ]);
    });

    it('should preserve other result properties when converting', async () => {
      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        outputFormat: 'float32Array',
      });

      expect(result.width).toBe(2);
      expect(result.height).toBe(1);
      expect(result.channels).toBe(3);
      expect(result.dataLayout).toBe('hwc');
      expect(result.processingTimeMs).toBe(1);
    });
  });

  describe('error handling', () => {
    it('should throw VisionUtilsException on native error', async () => {
      mockGetPixelData.mockRejectedValue({
        code: 'LOAD_ERROR',
        message: 'Failed to load image',
      });

      await expect(
        getPixelData({
          source: { type: 'url', value: 'https://invalid-url.com/image.jpg' },
        })
      ).rejects.toMatchObject({
        code: 'LOAD_ERROR',
        originalMessage: 'Failed to load image',
      });
    });
  });
});

describe('batchGetPixelData', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should process multiple images', async () => {
    const mockBatchResult: BatchResult = {
      results: [
        {
          data: [0.5],
          width: 224,
          height: 224,
          channels: 3,
          colorFormat: 'rgb',
          dataLayout: 'hwc',
          shape: [224, 224, 3],
          processingTimeMs: 10,
        },
        {
          data: [0.6],
          width: 224,
          height: 224,
          channels: 3,
          colorFormat: 'rgb',
          dataLayout: 'hwc',
          shape: [224, 224, 3],
          processingTimeMs: 12,
        },
      ],
      totalTimeMs: 15,
    };
    mockBatchGetPixelData.mockResolvedValue(mockBatchResult);

    const options: GetPixelDataOptions[] = [
      { source: { type: 'url', value: 'https://example.com/image1.jpg' } },
      { source: { type: 'url', value: 'https://example.com/image2.jpg' } },
    ];

    const result = await batchGetPixelData(options);

    expect(mockBatchGetPixelData).toHaveBeenCalledTimes(1);
    expect(result.results).toHaveLength(2);
    expect(result.totalTimeMs).toBe(15);
  });

  it('should pass concurrency option', async () => {
    mockBatchGetPixelData.mockResolvedValue({
      results: [],
      totalTimeMs: 0,
    });

    await batchGetPixelData(
      [{ source: { type: 'url', value: 'https://example.com/image.jpg' } }],
      { concurrency: 2 }
    );

    const calledBatchOptions = mockBatchGetPixelData.mock.calls[0][1];
    expect(calledBatchOptions.concurrency).toBe(2);
  });

  it('should use default concurrency of 4', async () => {
    mockBatchGetPixelData.mockResolvedValue({
      results: [],
      totalTimeMs: 0,
    });

    await batchGetPixelData([
      { source: { type: 'url', value: 'https://example.com/image.jpg' } },
    ]);

    const calledBatchOptions = mockBatchGetPixelData.mock.calls[0][1];
    expect(calledBatchOptions.concurrency).toBe(4);
  });

  describe('batch output format conversion', () => {
    it('should convert each result to its requested outputFormat', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [
          {
            data: [0.1, 0.2, 0.3],
            width: 1,
            height: 1,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 1,
          },
          {
            data: [0.4, 0.5, 0.6],
            width: 1,
            height: 1,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 1,
          },
        ],
        totalTimeMs: 2,
      });

      const result = await batchGetPixelData([
        {
          source: { type: 'url', value: 'https://example.com/image1.jpg' },
          outputFormat: 'array',
        },
        {
          source: { type: 'url', value: 'https://example.com/image2.jpg' },
          outputFormat: 'float32Array',
        },
      ]);

      // First result should be array
      expect(Array.isArray((result.results[0] as PixelDataResult).data)).toBe(
        true
      );
      // Second result should be Float32Array
      expect((result.results[1] as PixelDataResult).data).toBeInstanceOf(
        Float32Array
      );
    });

    it('should not convert error results in batch', async () => {
      const errorResult = {
        error: true as const,
        code: 'LOAD_ERROR',
        message: 'Failed to load',
        index: 1,
      };

      mockBatchGetPixelData.mockResolvedValue({
        results: [
          {
            data: [0.1, 0.2, 0.3],
            width: 1,
            height: 1,
            channels: 3,
            colorFormat: 'rgb',
            dataLayout: 'hwc',
            shape: [1, 1, 3],
            processingTimeMs: 1,
          },
          errorResult,
        ],
        totalTimeMs: 2,
      });

      const result = await batchGetPixelData([
        {
          source: { type: 'url', value: 'https://example.com/image1.jpg' },
          outputFormat: 'float32Array',
        },
        {
          source: { type: 'url', value: 'https://example.com/image2.jpg' },
          outputFormat: 'float32Array',
        },
      ]);

      // First result should be converted
      expect((result.results[0] as PixelDataResult).data).toBeInstanceOf(
        Float32Array
      );
      // Second result should still be an error
      expect('error' in result.results[1]!).toBe(true);
      expect((result.results[1] as { code: string }).code).toBe('LOAD_ERROR');
    });
  });
});
