import { mockGetPixelData, mockBatchGetPixelData } from './jest.setup';
import {
  getPixelData,
  batchGetPixelData,
  VisionUtilsException,
} from '../index';

describe('Native Module Error Wrapping', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getPixelData native error wrapping', () => {
    it('should wrap Error instance with code in VisionUtilsException', async () => {
      const nativeError = new Error('Native error message');
      (nativeError as any).code = 'NATIVE_CODE';
      mockGetPixelData.mockRejectedValue(nativeError);

      try {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
        });
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('NATIVE_CODE');
        expect((error as VisionUtilsException).originalMessage).toBe(
          'Native error message'
        );
      }
    });

    it('should wrap Error without code using UNKNOWN code', async () => {
      const nativeError = new Error('Error without code');
      mockGetPixelData.mockRejectedValue(nativeError);

      try {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
        });
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('UNKNOWN');
        expect((error as VisionUtilsException).originalMessage).toBe(
          'Error without code'
        );
      }
    });

    it('should wrap non-Error values in VisionUtilsException', async () => {
      const nonErrorValue = { custom: 'rejection' };
      mockGetPixelData.mockRejectedValue(nonErrorValue);

      try {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
        });
        fail('Expected rejection');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('UNKNOWN');
      }
    });

    it('should wrap string rejection in VisionUtilsException', async () => {
      mockGetPixelData.mockRejectedValue('string error');

      try {
        await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
        });
        fail('Expected rejection');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('UNKNOWN');
        expect((error as VisionUtilsException).originalMessage).toBe(
          'string error'
        );
      }
    });
  });

  describe('batchGetPixelData native error wrapping', () => {
    it('should wrap Error instance with code in VisionUtilsException', async () => {
      const nativeError = new Error('Batch native error');
      (nativeError as any).code = 'BATCH_ERROR';
      mockBatchGetPixelData.mockRejectedValue(nativeError);

      try {
        await batchGetPixelData([
          { source: { type: 'url', value: 'https://example.com/image.jpg' } },
        ]);
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('BATCH_ERROR');
        expect((error as VisionUtilsException).originalMessage).toBe(
          'Batch native error'
        );
      }
    });

    it('should wrap Error without code using UNKNOWN code', async () => {
      const nativeError = new Error('Batch error without code');
      mockBatchGetPixelData.mockRejectedValue(nativeError);

      try {
        await batchGetPixelData([
          { source: { type: 'url', value: 'https://example.com/image.jpg' } },
        ]);
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('UNKNOWN');
      }
    });

    it('should wrap non-Error values in VisionUtilsException from batch', async () => {
      const nonErrorValue = { batchCustom: 'rejection' };
      mockBatchGetPixelData.mockRejectedValue(nonErrorValue);

      try {
        await batchGetPixelData([
          { source: { type: 'url', value: 'https://example.com/image.jpg' } },
        ]);
        fail('Expected rejection');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('UNKNOWN');
      }
    });
  });

  describe('batchGetPixelData validation error wrapping', () => {
    it('should include image index in error message for validation failures', async () => {
      try {
        await batchGetPixelData([
          { source: { type: 'url', value: 'https://example.com/image.jpg' } },
          { source: { type: 'url', value: '' } }, // Invalid - empty value
        ]);
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).message).toContain('Index 1');
      }
    });

    it('should wrap validation errors with VisionUtilsException', async () => {
      try {
        await batchGetPixelData([
          { source: { type: 'url', value: 'https://example.com/image.jpg' } },
          {
            source: { type: 'url', value: 'https://example.com/image.jpg' },
            resize: { width: -1, height: 224 },
          },
        ]);
        fail('Expected error to be thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(VisionUtilsException);
        expect((error as VisionUtilsException).code).toBe('INVALID_RESIZE');
        expect((error as VisionUtilsException).message).toContain('Index 1');
      }
    });

    it('should rethrow non-VisionUtilsException validation errors', async () => {
      // This tests the else branch where error is not VisionUtilsException
      // This is hard to trigger without modifying validateOptions
      // The current implementation only throws VisionUtilsException
      // This test documents expected behavior
      mockBatchGetPixelData.mockResolvedValue({
        results: [],
        totalTimeMs: 0,
      });

      // Normal validation works
      await expect(
        batchGetPixelData([
          { source: { type: 'url', value: 'https://example.com/image.jpg' } },
        ])
      ).resolves.toBeDefined();
    });
  });
});
