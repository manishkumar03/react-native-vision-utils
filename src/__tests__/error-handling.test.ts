import { mockGetPixelData, mockBatchGetPixelData } from './jest.setup';
import {
  getPixelData,
  batchGetPixelData,
  VisionUtilsException,
} from '../index';

describe('Error Handling', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getPixelData errors', () => {
    describe('native module errors', () => {
      it('should handle LOAD_ERROR from native', async () => {
        const nativeError = new Error('Failed to load image from URL');
        (nativeError as any).code = 'LOAD_ERROR';
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: { type: 'url', value: 'https://example.com/not-found.jpg' },
          })
        ).rejects.toMatchObject({
          code: 'LOAD_ERROR',
        });
      });

      it('should handle DECODE_ERROR from native', async () => {
        const nativeError = new Error('Invalid image format');
        (nativeError as any).code = 'DECODE_ERROR';
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: { type: 'url', value: 'https://example.com/corrupted.jpg' },
          })
        ).rejects.toMatchObject({
          code: 'DECODE_ERROR',
        });
      });

      it('should handle INVALID_SOURCE from native', async () => {
        const nativeError = new Error('Invalid source type');
        (nativeError as any).code = 'INVALID_SOURCE';
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: { type: 'url', value: 'invalid-url' },
          })
        ).rejects.toMatchObject({
          code: 'INVALID_SOURCE',
        });
      });

      it('should handle TIMEOUT error from native', async () => {
        const nativeError = new Error('Request timed out');
        (nativeError as any).code = 'TIMEOUT';
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: { type: 'url', value: 'https://slow-server.com/image.jpg' },
          })
        ).rejects.toMatchObject({
          code: 'TIMEOUT',
        });
      });

      it('should handle NETWORK_ERROR from native', async () => {
        const nativeError = new Error('Network connection failed');
        (nativeError as any).code = 'NETWORK_ERROR';
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: { type: 'url', value: 'https://example.com/image.jpg' },
          })
        ).rejects.toMatchObject({
          code: 'NETWORK_ERROR',
        });
      });

      it('should handle PERMISSION_DENIED error from native', async () => {
        const nativeError = new Error('Permission denied');
        (nativeError as any).code = 'PERMISSION_DENIED';
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: { type: 'photoLibrary', value: 'ph://ASSET123' },
          })
        ).rejects.toMatchObject({
          code: 'PERMISSION_DENIED',
        });
      });

      it('should handle CANCELLED error from native', async () => {
        const nativeError = new Error('Operation was cancelled');
        (nativeError as any).code = 'CANCELLED';
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: { type: 'url', value: 'https://example.com/image.jpg' },
          })
        ).rejects.toMatchObject({
          code: 'CANCELLED',
        });
      });

      it('should handle MEMORY_ERROR from native', async () => {
        const nativeError = new Error('Out of memory');
        (nativeError as any).code = 'MEMORY_ERROR';
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: {
              type: 'url',
              value: 'https://example.com/huge-image.jpg',
            },
            resize: { width: 10000, height: 10000 },
          })
        ).rejects.toMatchObject({
          code: 'MEMORY_ERROR',
        });
      });
    });

    describe('error message preservation', () => {
      it('should preserve original error message from native', async () => {
        const nativeError = new Error('Specific error message from native');
        (nativeError as any).code = 'LOAD_ERROR';
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: { type: 'url', value: 'https://example.com/image.jpg' },
          })
        ).rejects.toMatchObject({
          originalMessage: 'Specific error message from native',
          code: 'LOAD_ERROR',
        });
      });

      it('should handle error without code property', async () => {
        const nativeError = new Error('Unknown error');
        mockGetPixelData.mockRejectedValue(nativeError);

        await expect(
          getPixelData({
            source: { type: 'url', value: 'https://example.com/image.jpg' },
          })
        ).rejects.toThrow();
      });

      it('should handle string rejection', async () => {
        mockGetPixelData.mockRejectedValue('String error');

        await expect(
          getPixelData({
            source: { type: 'url', value: 'https://example.com/image.jpg' },
          })
        ).rejects.toBeDefined();
      });
    });

    describe('validation errors', () => {
      it('should throw synchronously for missing source', async () => {
        await expect(getPixelData({} as any)).rejects.toBeInstanceOf(
          VisionUtilsException
        );
      });

      it('should not call native module when validation fails', async () => {
        try {
          await getPixelData({
            source: { type: 'url' } as any,
          });
        } catch {
          // Expected
        }

        expect(mockGetPixelData).not.toHaveBeenCalled();
      });
    });

    describe('error recovery', () => {
      it('should allow retry after error', async () => {
        mockGetPixelData
          .mockRejectedValueOnce(new Error('First attempt failed'))
          .mockResolvedValueOnce({
            data: [],
            width: 100,
            height: 100,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 1,
          });

        // First call fails
        await expect(
          getPixelData({
            source: { type: 'url', value: 'https://example.com/image.jpg' },
          })
        ).rejects.toThrow();

        // Retry succeeds
        const result = await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
        });

        expect(result).toBeDefined();
        expect(mockGetPixelData).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('batchGetPixelData errors', () => {
    it('should handle batch-level error', async () => {
      mockBatchGetPixelData.mockRejectedValue(
        new Error('Batch processing failed')
      );

      await expect(
        batchGetPixelData([
          { source: { type: 'url', value: 'https://example.com/image1.jpg' } },
          { source: { type: 'url', value: 'https://example.com/image2.jpg' } },
        ])
      ).rejects.toThrow();
    });

    it('should handle partial failure in results', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [
          {
            data: [0.5],
            width: 100,
            height: 100,
            channels: 3,
            dataLayout: 'hwc',
            processingTimeMs: 1,
          },
          {
            error: true,
            code: 'LOAD_ERROR',
            message: 'Failed to load second image',
            index: 1,
          },
        ],
        totalTimeMs: 10,
      });

      const result = await batchGetPixelData([
        { source: { type: 'url', value: 'https://example.com/image1.jpg' } },
        { source: { type: 'url', value: 'https://example.com/not-found.jpg' } },
      ]);

      expect(result.results).toHaveLength(2);
      expect((result.results[1] as any).error).toBe(true);
    });

    it('should handle all images failing', async () => {
      mockBatchGetPixelData.mockResolvedValue({
        results: [
          { error: true, code: 'LOAD_ERROR', message: 'Failed', index: 0 },
          { error: true, code: 'LOAD_ERROR', message: 'Failed', index: 1 },
        ],
        totalTimeMs: 5,
      });

      const result = await batchGetPixelData([
        { source: { type: 'url', value: 'https://example.com/bad1.jpg' } },
        { source: { type: 'url', value: 'https://example.com/bad2.jpg' } },
      ]);

      expect(result.results).toHaveLength(2);
      expect((result.results[0] as any).error).toBe(true);
      expect((result.results[1] as any).error).toBe(true);
    });

    it('should validate all options before calling native', async () => {
      await expect(
        batchGetPixelData([
          { source: { type: 'url', value: 'https://example.com/image.jpg' } },
          { source: { type: 'url', value: '' } }, // Invalid
        ])
      ).rejects.toThrow();

      expect(mockBatchGetPixelData).not.toHaveBeenCalled();
    });

    it('should include image index in validation error', async () => {
      try {
        await batchGetPixelData([
          { source: { type: 'url', value: 'https://example.com/image.jpg' } },
          { source: { type: 'url' } as any }, // Missing value
        ]);
        fail('Expected error');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).message).toContain('1'); // Index
      }
    });
  });
});
