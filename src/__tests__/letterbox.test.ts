import { VisionUtilsException } from '../index';
import type { BoundingBox, ImageSource, BoxFormat } from '../types';
import { mockLetterbox, mockReverseLetterbox } from './testSetup';

const VisionUtils = require('react-native').NativeModules.VisionUtils;

// Test-local type for letterbox results (simplified for testing)
interface TestLetterboxResult {
  imageUri: string;
  scale: number;
  padX: number;
  padY: number;
  originalWidth: number;
  originalHeight: number;
  processingTimeMs: number;
}

// Helper functions with validation logic
async function letterbox(
  source: ImageSource,
  options: {
    targetWidth: number;
    targetHeight: number;
    fillColor?: [number, number, number];
    saveFormat?: 'jpg' | 'png';
  }
): Promise<TestLetterboxResult> {
  if (!source || typeof source !== 'object' || !source.type || !source.value) {
    throw new VisionUtilsException(
      'INVALID_SOURCE',
      'Source must be a valid ImageSource object'
    );
  }
  if (options.targetWidth <= 0 || options.targetHeight <= 0) {
    throw new VisionUtilsException(
      'INVALID_DIMENSIONS',
      'Target dimensions must be positive'
    );
  }
  if (options.fillColor) {
    if (options.fillColor.length !== 3) {
      throw new VisionUtilsException(
        'INVALID_COLOR',
        'Fill color must have 3 components'
      );
    }
    for (const c of options.fillColor) {
      if (c < 0 || c > 255) {
        throw new VisionUtilsException(
          'INVALID_COLOR',
          'Color components must be 0-255'
        );
      }
    }
  }
  const opts = {
    targetWidth: options.targetWidth,
    targetHeight: options.targetHeight,
    fillColor: options.fillColor ?? [0, 0, 0],
    saveFormat: options.saveFormat ?? 'jpg',
  };
  return VisionUtils.letterbox(source.value, opts);
}

async function reverseLetterbox(
  boxes: BoundingBox[],
  letterboxResult: TestLetterboxResult,
  options?: { format?: BoxFormat; clip?: boolean }
) {
  if (!boxes || !Array.isArray(boxes)) {
    throw new VisionUtilsException('INVALID_INPUT', 'Boxes must be an array');
  }
  if (!letterboxResult || typeof letterboxResult !== 'object') {
    throw new VisionUtilsException(
      'INVALID_INPUT',
      'LetterboxResult must be an object'
    );
  }
  if (typeof letterboxResult.scale !== 'number') {
    throw new VisionUtilsException(
      'INVALID_INPUT',
      'LetterboxResult must have a scale property'
    );
  }
  const format = options?.format ?? 'xyxy';
  const clip = options?.clip ?? false;
  return VisionUtils.reverseLetterbox(
    boxes,
    letterboxResult.scale,
    letterboxResult.padX,
    letterboxResult.padY,
    format,
    clip
  );
}

describe('Letterbox Utilities', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('letterbox', () => {
    it('should letterbox an image to target size', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const expectedResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        scale: 0.5,
        padX: 80,
        padY: 0,
        originalWidth: 1920,
        originalHeight: 1080,
        processingTimeMs: 15.5,
      };

      mockLetterbox.mockResolvedValue(expectedResult);

      const result = await letterbox(source, {
        targetWidth: 640,
        targetHeight: 640,
      });

      expect(mockLetterbox).toHaveBeenCalledWith('/path/to/image.jpg', {
        targetWidth: 640,
        targetHeight: 640,
        fillColor: [0, 0, 0],
        saveFormat: 'jpg',
      });
      expect(result.scale).toBe(0.5);
      expect(result.padX).toBe(80);
      expect(result.padY).toBe(0);
    });

    it('should letterbox with custom fill color', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const expectedResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        scale: 0.5,
        padX: 80,
        padY: 0,
        originalWidth: 1920,
        originalHeight: 1080,
        processingTimeMs: 15.5,
      };

      mockLetterbox.mockResolvedValue(expectedResult);

      await letterbox(source, {
        targetWidth: 640,
        targetHeight: 640,
        fillColor: [114, 114, 114],
      });

      expect(mockLetterbox).toHaveBeenCalledWith('/path/to/image.jpg', {
        targetWidth: 640,
        targetHeight: 640,
        fillColor: [114, 114, 114],
        saveFormat: 'jpg',
      });
    });

    it('should letterbox and save as PNG', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const expectedResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.png',
        scale: 0.5,
        padX: 0,
        padY: 60,
        originalWidth: 640,
        originalHeight: 480,
        processingTimeMs: 20.0,
      };

      mockLetterbox.mockResolvedValue(expectedResult);

      const result = await letterbox(source, {
        targetWidth: 640,
        targetHeight: 640,
        saveFormat: 'png',
      });

      expect(mockLetterbox).toHaveBeenCalledWith('/path/to/image.jpg', {
        targetWidth: 640,
        targetHeight: 640,
        fillColor: [0, 0, 0],
        saveFormat: 'png',
      });
      expect(result.imageUri).toContain('.png');
    });

    it('should handle square images without padding', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/square.jpg',
      };
      const expectedResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        scale: 1.0,
        padX: 0,
        padY: 0,
        originalWidth: 640,
        originalHeight: 640,
        processingTimeMs: 5.0,
      };

      mockLetterbox.mockResolvedValue(expectedResult);

      const result = await letterbox(source, {
        targetWidth: 640,
        targetHeight: 640,
      });

      expect(result.padX).toBe(0);
      expect(result.padY).toBe(0);
    });

    it('should throw error for invalid source', async () => {
      await expect(
        letterbox('' as unknown as ImageSource, {
          targetWidth: 640,
          targetHeight: 640,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid target dimensions', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };

      await expect(
        letterbox(source, {
          targetWidth: -1,
          targetHeight: 640,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for zero target dimensions', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };

      await expect(
        letterbox(source, {
          targetWidth: 0,
          targetHeight: 640,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid fill color', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };

      await expect(
        letterbox(source, {
          targetWidth: 640,
          targetHeight: 640,
          fillColor: [256, 0, 0],
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for fill color with wrong length', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };

      await expect(
        letterbox(source, {
          targetWidth: 640,
          targetHeight: 640,
          fillColor: [0, 0] as unknown as [number, number, number],
        })
      ).rejects.toThrow(VisionUtilsException);
    });
  });

  describe('reverseLetterbox', () => {
    it('should reverse letterbox transformation on boxes', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];
      const letterboxResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        scale: 0.5,
        padX: 80,
        padY: 0,
        originalWidth: 1920,
        originalHeight: 1080,
        processingTimeMs: 15.5,
      };

      const expectedResult = {
        boxes: [[40, 200, 240, 400]],
        format: 'xyxy',
        processingTimeMs: 0.5,
      };

      mockReverseLetterbox.mockResolvedValue(expectedResult);

      const result = await reverseLetterbox(boxes, letterboxResult, {
        format: 'xyxy',
      });

      expect(mockReverseLetterbox).toHaveBeenCalledWith(
        boxes,
        letterboxResult.scale,
        letterboxResult.padX,
        letterboxResult.padY,
        'xyxy',
        false
      );
      expect(result.boxes).toEqual([[40, 200, 240, 400]]);
    });

    it('should reverse letterbox with clipping', async () => {
      const boxes: BoundingBox[] = [[-10, -10, 700, 500]];
      const letterboxResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        scale: 0.5,
        padX: 80,
        padY: 0,
        originalWidth: 640,
        originalHeight: 480,
        processingTimeMs: 15.5,
      };

      const expectedResult = {
        boxes: [[0, 0, 640, 480]],
        format: 'xyxy',
        processingTimeMs: 0.5,
      };

      mockReverseLetterbox.mockResolvedValue(expectedResult);

      const result = await reverseLetterbox(boxes, letterboxResult, {
        format: 'xyxy',
        clip: true,
      });

      expect(mockReverseLetterbox).toHaveBeenCalledWith(
        boxes,
        letterboxResult.scale,
        letterboxResult.padX,
        letterboxResult.padY,
        'xyxy',
        true
      );
      expect(result.boxes).toEqual([[0, 0, 640, 480]]);
    });

    it('should use default format', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];
      const letterboxResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        scale: 1.0,
        padX: 0,
        padY: 60,
        originalWidth: 640,
        originalHeight: 480,
        processingTimeMs: 10.0,
      };

      const expectedResult = {
        boxes: [[100, 40, 200, 140]],
        format: 'xyxy',
        processingTimeMs: 0.5,
      };

      mockReverseLetterbox.mockResolvedValue(expectedResult);

      await reverseLetterbox(boxes, letterboxResult);

      expect(mockReverseLetterbox).toHaveBeenCalledWith(
        boxes,
        letterboxResult.scale,
        letterboxResult.padX,
        letterboxResult.padY,
        'xyxy',
        false
      );
    });

    it('should handle multiple boxes', async () => {
      const boxes: BoundingBox[] = [
        [100, 100, 200, 200],
        [300, 300, 400, 400],
      ];
      const letterboxResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        scale: 0.5,
        padX: 80,
        padY: 0,
        originalWidth: 1920,
        originalHeight: 1080,
        processingTimeMs: 15.5,
      };

      const expectedResult = {
        boxes: [
          [40, 200, 240, 400],
          [440, 600, 640, 800],
        ],
        format: 'xyxy',
        processingTimeMs: 0.5,
      };

      mockReverseLetterbox.mockResolvedValue(expectedResult);

      const result = await reverseLetterbox(boxes, letterboxResult);

      expect(result.boxes.length).toBe(2);
    });

    it('should throw error for invalid boxes', async () => {
      const letterboxResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        scale: 0.5,
        padX: 80,
        padY: 0,
        originalWidth: 1920,
        originalHeight: 1080,
        processingTimeMs: 15.5,
      };

      await expect(
        reverseLetterbox(null as unknown as BoundingBox[], letterboxResult)
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid letterbox result', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];

      await expect(
        reverseLetterbox(boxes, null as unknown as TestLetterboxResult)
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for letterbox result missing scale', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];
      const invalidLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        padX: 80,
        padY: 0,
      } as unknown as TestLetterboxResult;

      await expect(
        reverseLetterbox(boxes, invalidLetterboxResult)
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should handle xywh format', async () => {
      const boxes: BoundingBox[] = [[100, 100, 100, 100]];
      const letterboxResult: TestLetterboxResult = {
        imageUri: 'file:///path/to/letterboxed.jpg',
        scale: 0.5,
        padX: 80,
        padY: 0,
        originalWidth: 1920,
        originalHeight: 1080,
        processingTimeMs: 15.5,
      };

      const expectedResult = {
        boxes: [[40, 200, 200, 200]],
        format: 'xywh',
        processingTimeMs: 0.5,
      };

      mockReverseLetterbox.mockResolvedValue(expectedResult);

      const result = await reverseLetterbox(boxes, letterboxResult, {
        format: 'xywh',
      });

      expect(result.format).toBe('xywh');
    });
  });
});
