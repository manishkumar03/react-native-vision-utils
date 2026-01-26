import type {
  ColorFormat,
  DataLayout,
  NormalizationPreset,
  ResizeStrategy,
} from '../types';
import { VisionUtilsException } from '../index';

describe('Types', () => {
  describe('ColorFormat', () => {
    it('should have all expected values', () => {
      const formats: ColorFormat[] = [
        'rgb',
        'rgba',
        'bgr',
        'bgra',
        'grayscale',
        'hsv',
        'hsl',
        'lab',
        'yuv',
        'ycbcr',
      ];
      expect(formats).toHaveLength(10);
    });
  });

  describe('DataLayout', () => {
    it('should have all expected values', () => {
      const layouts: DataLayout[] = ['hwc', 'chw', 'nhwc', 'nchw'];
      expect(layouts).toHaveLength(4);
    });
  });

  describe('NormalizationPreset', () => {
    it('should have all expected values', () => {
      const presets: NormalizationPreset[] = [
        'imagenet',
        'tensorflow',
        'scale',
        'raw',
        'custom',
      ];
      expect(presets).toHaveLength(5);
    });
  });

  describe('ResizeStrategy', () => {
    it('should have all expected values', () => {
      const strategies: ResizeStrategy[] = [
        'cover',
        'contain',
        'stretch',
        'letterbox',
      ];
      expect(strategies).toHaveLength(4);
    });
  });

  describe('VisionUtilsException', () => {
    it('should create exception with code and message', () => {
      const error = new VisionUtilsException(
        'LOAD_ERROR',
        'Failed to load image'
      );
      expect(error.code).toBe('LOAD_ERROR');
      expect(error.message).toBe('[LOAD_ERROR] Failed to load image');
      expect(error.originalMessage).toBe('Failed to load image');
      expect(error.name).toBe('VisionUtilsException');
    });

    it('should be instance of Error', () => {
      const error = new VisionUtilsException('TEST', 'Test error');
      expect(error).toBeInstanceOf(Error);
    });
  });
});
