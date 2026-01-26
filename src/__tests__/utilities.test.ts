import {
  getChannelCount,
  VisionUtilsException,
  isVisionUtilsError,
} from '../index';
import type { ColorFormat, PixelDataResult } from '../types';

// Type for batch result errors in utility tests
type BatchErrorResult = {
  error: true;
  code: string;
  message: string;
  index: number;
};

describe('Utility Functions', () => {
  describe('getChannelCount', () => {
    it('should return 3 for rgb', () => {
      expect(getChannelCount('rgb')).toBe(3);
    });

    it('should return 4 for rgba', () => {
      expect(getChannelCount('rgba')).toBe(4);
    });

    it('should return 3 for bgr', () => {
      expect(getChannelCount('bgr')).toBe(3);
    });

    it('should return 4 for bgra', () => {
      expect(getChannelCount('bgra')).toBe(4);
    });

    it('should return 1 for grayscale', () => {
      expect(getChannelCount('grayscale')).toBe(1);
    });

    it('should work with lowercase input via type assertion', () => {
      // This tests type safety - the function expects ColorFormat type
      const format: ColorFormat = 'rgb';
      expect(getChannelCount(format)).toBe(3);
    });
  });

  describe('VisionUtilsException', () => {
    it('should create exception with code and message', () => {
      const error = new VisionUtilsException('TEST_CODE', 'Test message');
      expect(error.code).toBe('TEST_CODE');
      expect(error.message).toBe('[TEST_CODE] Test message');
      expect(error.originalMessage).toBe('Test message');
    });

    it('should have name VisionUtilsException', () => {
      const error = new VisionUtilsException('TEST', 'Test');
      expect(error.name).toBe('VisionUtilsException');
    });

    it('should be instance of Error', () => {
      const error = new VisionUtilsException('TEST', 'Test');
      expect(error).toBeInstanceOf(Error);
    });

    it('should be instance of VisionUtilsException', () => {
      const error = new VisionUtilsException('TEST', 'Test');
      expect(error).toBeInstanceOf(VisionUtilsException);
    });

    it('should have stack trace', () => {
      const error = new VisionUtilsException('TEST', 'Test');
      expect(error.stack).toBeDefined();
    });

    it('should be catchable as Error', () => {
      try {
        throw new VisionUtilsException('TEST', 'Test');
      } catch (e) {
        expect(e).toBeInstanceOf(Error);
      }
    });

    it('should preserve code through catch', () => {
      try {
        throw new VisionUtilsException('PRESERVED_CODE', 'Test');
      } catch (e) {
        if (e instanceof VisionUtilsException) {
          expect(e.code).toBe('PRESERVED_CODE');
        } else {
          fail('Expected VisionUtilsException');
        }
      }
    });

    describe('common error codes', () => {
      it('should handle INVALID_SOURCE code', () => {
        const error = new VisionUtilsException(
          'INVALID_SOURCE',
          'Source is invalid'
        );
        expect(error.code).toBe('INVALID_SOURCE');
      });

      it('should handle LOAD_ERROR code', () => {
        const error = new VisionUtilsException('LOAD_ERROR', 'Failed to load');
        expect(error.code).toBe('LOAD_ERROR');
      });

      it('should handle DECODE_ERROR code', () => {
        const error = new VisionUtilsException(
          'DECODE_ERROR',
          'Failed to decode'
        );
        expect(error.code).toBe('DECODE_ERROR');
      });

      it('should handle INVALID_RESIZE code', () => {
        const error = new VisionUtilsException(
          'INVALID_RESIZE',
          'Invalid dimensions'
        );
        expect(error.code).toBe('INVALID_RESIZE');
      });

      it('should handle INVALID_ROI code', () => {
        const error = new VisionUtilsException('INVALID_ROI', 'Invalid ROI');
        expect(error.code).toBe('INVALID_ROI');
      });

      it('should handle INVALID_NORMALIZATION code', () => {
        const error = new VisionUtilsException(
          'INVALID_NORMALIZATION',
          'Invalid normalization'
        );
        expect(error.code).toBe('INVALID_NORMALIZATION');
      });

      it('should handle UNKNOWN code', () => {
        const error = new VisionUtilsException('UNKNOWN', 'Unknown error');
        expect(error.code).toBe('UNKNOWN');
      });

      it('should handle CANCELLED code', () => {
        const error = new VisionUtilsException(
          'CANCELLED',
          'Operation cancelled'
        );
        expect(error.code).toBe('CANCELLED');
      });
    });
  });

  describe('isVisionUtilsError', () => {
    // isVisionUtilsError is a type guard for batch results
    // It checks if a result has error: true property

    it('should return true for VisionUtilsError with error: true', () => {
      const error: BatchErrorResult = {
        error: true,
        code: 'TEST_CODE',
        message: 'Test message',
        index: 0,
      };
      expect(isVisionUtilsError(error)).toBe(true);
    });

    it('should return false for successful PixelDataResult', () => {
      const result: PixelDataResult = {
        data: [0.5],
        width: 100,
        height: 100,
        channels: 3,
        colorFormat: 'rgb',
        dataLayout: 'hwc',
        shape: [100, 100, 3],
        processingTimeMs: 10,
      };
      expect(isVisionUtilsError(result)).toBe(false);
    });

    it('should return true for error result in batch', () => {
      const batchError: BatchErrorResult = {
        error: true,
        code: 'LOAD_ERROR',
        message: 'Failed to load image',
        index: 2,
      };
      expect(isVisionUtilsError(batchError)).toBe(true);
    });

    it('should correctly identify errors in a batch array', () => {
      const results: (PixelDataResult | BatchErrorResult)[] = [
        {
          data: [],
          width: 100,
          height: 100,
          channels: 3,
          colorFormat: 'rgb',
          dataLayout: 'hwc',
          shape: [100, 100, 3],
          processingTimeMs: 1,
        },
        { error: true, code: 'LOAD_ERROR', message: 'Failed', index: 1 },
        {
          data: [],
          width: 200,
          height: 200,
          channels: 3,
          colorFormat: 'rgb',
          dataLayout: 'hwc',
          shape: [200, 200, 3],
          processingTimeMs: 2,
        },
      ];

      expect(isVisionUtilsError(results[0]!)).toBe(false);
      expect(isVisionUtilsError(results[1]!)).toBe(true);
      expect(isVisionUtilsError(results[2]!)).toBe(false);
    });

    it('should return false for object without error property', () => {
      const obj = {
        data: [],
        width: 100,
        height: 100,
        channels: 3,
        colorFormat: 'rgb' as const,
        dataLayout: 'hwc' as const,
        shape: [100, 100, 3],
        processingTimeMs: 1,
      } as PixelDataResult;
      expect(isVisionUtilsError(obj)).toBe(false);
    });

    it('should work with filter to separate errors from successes', () => {
      const results: (PixelDataResult | BatchErrorResult)[] = [
        {
          data: [],
          width: 100,
          height: 100,
          channels: 3,
          colorFormat: 'rgb',
          dataLayout: 'hwc',
          shape: [100, 100, 3],
          processingTimeMs: 1,
        },
        { error: true, code: 'ERROR', message: 'Failed', index: 1 },
        {
          data: [],
          width: 100,
          height: 100,
          channels: 3,
          colorFormat: 'rgb',
          dataLayout: 'hwc',
          shape: [100, 100, 3],
          processingTimeMs: 1,
        },
        { error: true, code: 'ERROR', message: 'Failed', index: 3 },
      ];

      const errors = results.filter(isVisionUtilsError);
      const successes = results.filter((r) => !isVisionUtilsError(r));

      expect(errors).toHaveLength(2);
      expect(successes).toHaveLength(2);
    });
  });
});
