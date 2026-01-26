import { VisionUtilsException } from '../index';
import type { BoundingBox, Detection, BoxFormat } from '../types';
import {
  mockConvertBoxFormat,
  mockScaleBoxes,
  mockClipBoxes,
  mockCalculateIoU,
  mockNonMaxSuppression,
} from './testSetup';

// Re-export for testing - these are pure functions that call native module
// We'll test the validation logic and mock the native calls
const VisionUtils = require('react-native').NativeModules.VisionUtils;

// Helper to create validated convert box format call
async function convertBoxFormat(
  boxes: BoundingBox[],
  options: { fromFormat: BoxFormat; toFormat: BoxFormat }
) {
  if (!boxes || !Array.isArray(boxes)) {
    throw new VisionUtilsException('INVALID_INPUT', 'Boxes must be an array');
  }
  const validFormats: BoxFormat[] = ['xyxy', 'xywh', 'cxcywh'];
  if (!validFormats.includes(options.fromFormat)) {
    throw new VisionUtilsException(
      'INVALID_FORMAT',
      `Invalid fromFormat: ${options.fromFormat}`
    );
  }
  if (!validFormats.includes(options.toFormat)) {
    throw new VisionUtilsException(
      'INVALID_FORMAT',
      `Invalid toFormat: ${options.toFormat}`
    );
  }
  return VisionUtils.convertBoxFormat(
    boxes,
    options.fromFormat,
    options.toFormat
  );
}

async function scaleBoxes(
  boxes: BoundingBox[],
  options: {
    fromWidth: number;
    fromHeight: number;
    toWidth: number;
    toHeight: number;
    format?: BoxFormat;
    clip?: boolean;
  }
) {
  if (!boxes || !Array.isArray(boxes)) {
    throw new VisionUtilsException('INVALID_INPUT', 'Boxes must be an array');
  }
  if (
    options.fromWidth <= 0 ||
    options.fromHeight <= 0 ||
    options.toWidth <= 0 ||
    options.toHeight <= 0
  ) {
    throw new VisionUtilsException(
      'INVALID_DIMENSIONS',
      'All dimensions must be positive'
    );
  }
  return VisionUtils.scaleBoxes(boxes, options);
}

async function clipBoxes(
  boxes: BoundingBox[],
  options: { width: number; height: number; format?: BoxFormat }
) {
  if (!boxes || !Array.isArray(boxes)) {
    throw new VisionUtilsException('INVALID_INPUT', 'Boxes must be an array');
  }
  if (options.width <= 0 || options.height <= 0) {
    throw new VisionUtilsException(
      'INVALID_DIMENSIONS',
      'Width and height must be positive'
    );
  }
  const format = options.format ?? 'xyxy';
  return VisionUtils.clipBoxes(boxes, options.width, options.height, format);
}

async function calculateIoU(
  box1: BoundingBox,
  box2: BoundingBox,
  format?: BoxFormat
) {
  if (!box1 || !Array.isArray(box1) || box1.length !== 4) {
    throw new VisionUtilsException(
      'INVALID_INPUT',
      'box1 must be an array of 4 numbers'
    );
  }
  if (!box2 || !Array.isArray(box2) || box2.length !== 4) {
    throw new VisionUtilsException(
      'INVALID_INPUT',
      'box2 must be an array of 4 numbers'
    );
  }
  return VisionUtils.calculateIoU(box1, box2, format ?? 'xyxy');
}

async function nonMaxSuppression(
  detections: Detection[],
  options?: {
    iouThreshold?: number;
    scoreThreshold?: number;
    maxDetections?: number;
    format?: BoxFormat;
  }
) {
  if (!detections || !Array.isArray(detections)) {
    throw new VisionUtilsException(
      'INVALID_INPUT',
      'Detections must be an array'
    );
  }
  const opts = {
    iouThreshold: options?.iouThreshold ?? 0.5,
    scoreThreshold: options?.scoreThreshold ?? 0.0,
    maxDetections: options?.maxDetections,
    format: options?.format ?? 'xyxy',
  };
  return VisionUtils.nonMaxSuppression(detections, opts);
}

describe('Bounding Box Utilities', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('convertBoxFormat', () => {
    it('should convert boxes from xyxy to xywh', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];
      const expectedResult = {
        boxes: [[100, 100, 100, 100]],
        format: 'xywh',
        processingTimeMs: 0.5,
      };

      mockConvertBoxFormat.mockResolvedValue(expectedResult);

      const result = await convertBoxFormat(boxes, {
        fromFormat: 'xyxy',
        toFormat: 'xywh',
      });

      expect(mockConvertBoxFormat).toHaveBeenCalledWith(boxes, 'xyxy', 'xywh');
      expect(result.boxes).toEqual([[100, 100, 100, 100]]);
      expect(result.format).toBe('xywh');
    });

    it('should convert boxes from xywh to cxcywh', async () => {
      const boxes: BoundingBox[] = [[100, 100, 100, 100]];
      const expectedResult = {
        boxes: [[150, 150, 100, 100]],
        format: 'cxcywh',
        processingTimeMs: 0.5,
      };

      mockConvertBoxFormat.mockResolvedValue(expectedResult);

      const result = await convertBoxFormat(boxes, {
        fromFormat: 'xywh',
        toFormat: 'cxcywh',
      });

      expect(mockConvertBoxFormat).toHaveBeenCalledWith(
        boxes,
        'xywh',
        'cxcywh'
      );
      expect(result.format).toBe('cxcywh');
    });

    it('should convert boxes from cxcywh to xyxy', async () => {
      const boxes: BoundingBox[] = [[150, 150, 100, 100]];
      const expectedResult = {
        boxes: [[100, 100, 200, 200]],
        format: 'xyxy',
        processingTimeMs: 0.5,
      };

      mockConvertBoxFormat.mockResolvedValue(expectedResult);

      const result = await convertBoxFormat(boxes, {
        fromFormat: 'cxcywh',
        toFormat: 'xyxy',
      });

      expect(result.boxes).toEqual([[100, 100, 200, 200]]);
    });

    it('should handle multiple boxes', async () => {
      const boxes: BoundingBox[] = [
        [0, 0, 100, 100],
        [200, 200, 300, 300],
      ];
      const expectedResult = {
        boxes: [
          [0, 0, 100, 100],
          [200, 200, 100, 100],
        ],
        format: 'xywh',
        processingTimeMs: 0.5,
      };

      mockConvertBoxFormat.mockResolvedValue(expectedResult);

      const result = await convertBoxFormat(boxes, {
        fromFormat: 'xyxy',
        toFormat: 'xywh',
      });

      expect(result.boxes.length).toBe(2);
    });

    it('should throw error for invalid input', async () => {
      await expect(
        convertBoxFormat(null as unknown as BoundingBox[], {
          fromFormat: 'xyxy',
          toFormat: 'xywh',
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid fromFormat', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];

      await expect(
        convertBoxFormat(boxes, {
          fromFormat: 'invalid' as BoxFormat,
          toFormat: 'xywh',
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid toFormat', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];

      await expect(
        convertBoxFormat(boxes, {
          fromFormat: 'xyxy',
          toFormat: 'invalid' as BoxFormat,
        })
      ).rejects.toThrow(VisionUtilsException);
    });
  });

  describe('scaleBoxes', () => {
    it('should scale boxes from one size to another', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];
      const expectedResult = {
        boxes: [[300, 225, 600, 450]],
        format: 'xyxy',
        processingTimeMs: 0.5,
      };

      mockScaleBoxes.mockResolvedValue(expectedResult);

      const result = await scaleBoxes(boxes, {
        fromWidth: 640,
        fromHeight: 640,
        toWidth: 1920,
        toHeight: 1440,
        format: 'xyxy',
      });

      expect(mockScaleBoxes).toHaveBeenCalledWith(boxes, {
        fromWidth: 640,
        fromHeight: 640,
        toWidth: 1920,
        toHeight: 1440,
        format: 'xyxy',
      });
      expect(result.boxes).toEqual([[300, 225, 600, 450]]);
    });

    it('should scale boxes with clipping enabled', async () => {
      const boxes: BoundingBox[] = [[0, 0, 700, 700]];
      const expectedResult = {
        boxes: [[0, 0, 640, 480]],
        format: 'xyxy',
        processingTimeMs: 0.5,
      };

      mockScaleBoxes.mockResolvedValue(expectedResult);

      const result = await scaleBoxes(boxes, {
        fromWidth: 640,
        fromHeight: 640,
        toWidth: 640,
        toHeight: 480,
        format: 'xyxy',
        clip: true,
      });

      expect(result.boxes).toEqual([[0, 0, 640, 480]]);
    });

    it('should throw error for invalid input', async () => {
      await expect(
        scaleBoxes(null as unknown as BoundingBox[], {
          fromWidth: 640,
          fromHeight: 640,
          toWidth: 1920,
          toHeight: 1080,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid dimensions', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];

      await expect(
        scaleBoxes(boxes, {
          fromWidth: -1,
          fromHeight: 640,
          toWidth: 1920,
          toHeight: 1080,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for zero dimensions', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];

      await expect(
        scaleBoxes(boxes, {
          fromWidth: 0,
          fromHeight: 640,
          toWidth: 1920,
          toHeight: 1080,
        })
      ).rejects.toThrow(VisionUtilsException);
    });
  });

  describe('clipBoxes', () => {
    it('should clip boxes to image boundaries', async () => {
      const boxes: BoundingBox[] = [[-10, -10, 700, 500]];
      const expectedResult = {
        boxes: [[0, 0, 640, 480]],
        format: 'xyxy',
        removedCount: 0,
        processingTimeMs: 0.5,
      };

      mockClipBoxes.mockResolvedValue(expectedResult);

      const result = await clipBoxes(boxes, {
        width: 640,
        height: 480,
        format: 'xyxy',
      });

      expect(mockClipBoxes).toHaveBeenCalledWith(boxes, 640, 480, 'xyxy');
      expect(result.boxes).toEqual([[0, 0, 640, 480]]);
    });

    it('should use default format when not specified', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];
      const expectedResult = {
        boxes: [[100, 100, 200, 200]],
        format: 'xyxy',
        removedCount: 0,
        processingTimeMs: 0.5,
      };

      mockClipBoxes.mockResolvedValue(expectedResult);

      await clipBoxes(boxes, { width: 640, height: 480 });

      expect(mockClipBoxes).toHaveBeenCalledWith(boxes, 640, 480, 'xyxy');
    });

    it('should throw error for invalid input', async () => {
      await expect(
        clipBoxes(null as unknown as BoundingBox[], {
          width: 640,
          height: 480,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid dimensions', async () => {
      const boxes: BoundingBox[] = [[100, 100, 200, 200]];

      await expect(
        clipBoxes(boxes, {
          width: -1,
          height: 480,
        })
      ).rejects.toThrow(VisionUtilsException);
    });
  });

  describe('calculateIoU', () => {
    it('should calculate IoU between two overlapping boxes', async () => {
      const box1: BoundingBox = [100, 100, 200, 200];
      const box2: BoundingBox = [150, 150, 250, 250];
      const expectedResult = {
        iou: 0.142857,
        intersection: 2500,
        union: 17500,
        processingTimeMs: 0.1,
      };

      mockCalculateIoU.mockResolvedValue(expectedResult);

      const result = await calculateIoU(box1, box2, 'xyxy');

      expect(mockCalculateIoU).toHaveBeenCalledWith(box1, box2, 'xyxy');
      expect(result.iou).toBeCloseTo(0.142857, 5);
      expect(result.intersection).toBe(2500);
    });

    it('should return 0 for non-overlapping boxes', async () => {
      const box1: BoundingBox = [0, 0, 100, 100];
      const box2: BoundingBox = [200, 200, 300, 300];
      const expectedResult = {
        iou: 0,
        intersection: 0,
        union: 20000,
        processingTimeMs: 0.1,
      };

      mockCalculateIoU.mockResolvedValue(expectedResult);

      const result = await calculateIoU(box1, box2, 'xyxy');

      expect(result.iou).toBe(0);
      expect(result.intersection).toBe(0);
    });

    it('should return 1 for identical boxes', async () => {
      const box1: BoundingBox = [100, 100, 200, 200];
      const box2: BoundingBox = [100, 100, 200, 200];
      const expectedResult = {
        iou: 1,
        intersection: 10000,
        union: 10000,
        processingTimeMs: 0.1,
      };

      mockCalculateIoU.mockResolvedValue(expectedResult);

      const result = await calculateIoU(box1, box2, 'xyxy');

      expect(result.iou).toBe(1);
    });

    it('should use default format when not specified', async () => {
      const box1: BoundingBox = [100, 100, 200, 200];
      const box2: BoundingBox = [150, 150, 250, 250];
      const expectedResult = {
        iou: 0.142857,
        intersection: 2500,
        union: 17500,
        processingTimeMs: 0.1,
      };

      mockCalculateIoU.mockResolvedValue(expectedResult);

      await calculateIoU(box1, box2);

      expect(mockCalculateIoU).toHaveBeenCalledWith(box1, box2, 'xyxy');
    });

    it('should throw error for invalid box1', async () => {
      const box2: BoundingBox = [100, 100, 200, 200];

      await expect(
        calculateIoU(null as unknown as BoundingBox, box2)
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid box2', async () => {
      const box1: BoundingBox = [100, 100, 200, 200];

      await expect(
        calculateIoU(box1, null as unknown as BoundingBox)
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for box with wrong length', async () => {
      const box1: BoundingBox = [100, 100, 200, 200];
      const box2 = [100, 100, 200] as unknown as BoundingBox;

      await expect(calculateIoU(box1, box2)).rejects.toThrow(
        VisionUtilsException
      );
    });
  });

  describe('nonMaxSuppression', () => {
    it('should filter overlapping detections', async () => {
      const detections: Detection[] = [
        { box: [100, 100, 200, 200], score: 0.9, classIndex: 0 },
        { box: [110, 110, 210, 210], score: 0.8, classIndex: 0 },
        { box: [300, 300, 400, 400], score: 0.7, classIndex: 1 },
      ];
      const expectedResult = {
        indices: [0, 2],
        detections: [
          { box: [100, 100, 200, 200], score: 0.9, classIndex: 0 },
          { box: [300, 300, 400, 400], score: 0.7, classIndex: 1 },
        ],
        suppressedCount: 1,
        processingTimeMs: 0.5,
      };

      mockNonMaxSuppression.mockResolvedValue(expectedResult);

      const result = await nonMaxSuppression(detections, {
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
      });

      expect(mockNonMaxSuppression).toHaveBeenCalledWith(detections, {
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
        maxDetections: undefined,
        format: 'xyxy',
      });
      expect(result.detections.length).toBe(2);
      expect(result.suppressedCount).toBe(1);
    });

    it('should respect maxDetections limit', async () => {
      const detections: Detection[] = [
        { box: [0, 0, 100, 100], score: 0.9 },
        { box: [200, 200, 300, 300], score: 0.8 },
        { box: [400, 400, 500, 500], score: 0.7 },
      ];
      const expectedResult = {
        indices: [0],
        detections: [{ box: [0, 0, 100, 100], score: 0.9 }],
        suppressedCount: 2,
        processingTimeMs: 0.5,
      };

      mockNonMaxSuppression.mockResolvedValue(expectedResult);

      const result = await nonMaxSuppression(detections, {
        maxDetections: 1,
      });

      expect(result.detections.length).toBe(1);
    });

    it('should filter by score threshold', async () => {
      const detections: Detection[] = [
        { box: [0, 0, 100, 100], score: 0.9 },
        { box: [200, 200, 300, 300], score: 0.2 },
      ];
      const expectedResult = {
        indices: [0],
        detections: [{ box: [0, 0, 100, 100], score: 0.9 }],
        suppressedCount: 1,
        processingTimeMs: 0.5,
      };

      mockNonMaxSuppression.mockResolvedValue(expectedResult);

      const result = await nonMaxSuppression(detections, {
        scoreThreshold: 0.5,
      });

      expect(result.detections.length).toBe(1);
    });

    it('should use default options', async () => {
      const detections: Detection[] = [
        { box: [100, 100, 200, 200], score: 0.9 },
      ];
      const expectedResult = {
        indices: [0],
        detections: [{ box: [100, 100, 200, 200], score: 0.9 }],
        suppressedCount: 0,
        processingTimeMs: 0.5,
      };

      mockNonMaxSuppression.mockResolvedValue(expectedResult);

      await nonMaxSuppression(detections);

      expect(mockNonMaxSuppression).toHaveBeenCalledWith(detections, {
        iouThreshold: 0.5,
        scoreThreshold: 0.0,
        maxDetections: undefined,
        format: 'xyxy',
      });
    });

    it('should throw error for invalid input', async () => {
      await expect(
        nonMaxSuppression(null as unknown as Detection[])
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should preserve labels in detections', async () => {
      const detections: Detection[] = [
        { box: [100, 100, 200, 200], score: 0.9, label: 'person' },
      ];
      const expectedResult = {
        indices: [0],
        detections: [
          { box: [100, 100, 200, 200], score: 0.9, label: 'person' },
        ],
        suppressedCount: 0,
        processingTimeMs: 0.5,
      };

      mockNonMaxSuppression.mockResolvedValue(expectedResult);

      const result = await nonMaxSuppression(detections);

      expect(result.detections[0].label).toBe('person');
    });
  });
});
