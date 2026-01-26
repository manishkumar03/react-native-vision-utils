import { VisionUtilsException } from '../index';
import type { Detection, Keypoint, ImageSource, BoxFormat } from '../types';
import {
  mockDrawBoxes,
  mockDrawKeypoints,
  mockOverlayMask,
  mockOverlayHeatmap,
} from './testSetup';

const VisionUtils = require('react-native').NativeModules.VisionUtils;

// Test-local types for drawing options (different from exported types for testing purposes)
interface TestDrawBoxesOptions {
  format?: BoxFormat;
  lineWidth?: number;
  showLabels?: boolean;
  showScores?: boolean;
  fontSize?: number;
  saveFormat?: 'jpg' | 'png';
  colors?: [number, number, number][];
}

// Helper functions with validation logic
async function drawBoxes(
  source: ImageSource,
  detections: Detection[],
  options?: TestDrawBoxesOptions
) {
  if (!source || typeof source !== 'object' || !source.type || !source.value) {
    throw new VisionUtilsException(
      'INVALID_SOURCE',
      'Source must be a valid ImageSource object'
    );
  }
  if (!detections || !Array.isArray(detections)) {
    throw new VisionUtilsException(
      'INVALID_INPUT',
      'Detections must be an array'
    );
  }
  const opts = {
    format: (options?.format ?? 'xyxy') as BoxFormat,
    lineWidth: options?.lineWidth ?? 2,
    showLabels: options?.showLabels ?? true,
    showScores: options?.showScores ?? true,
    fontSize: options?.fontSize ?? 14,
    saveFormat: options?.saveFormat ?? 'jpg',
    ...(options?.colors && { colors: options.colors }),
  };
  return VisionUtils.drawBoxes(source.value, detections, opts);
}

async function drawKeypoints(
  source: ImageSource,
  keypoints: Keypoint[][],
  options?: {
    radius?: number;
    showConfidence?: boolean;
    skeleton?: number[][];
    colors?: [number, number, number][];
    saveFormat?: 'jpg' | 'png';
  }
) {
  if (!source || typeof source !== 'object' || !source.type || !source.value) {
    throw new VisionUtilsException(
      'INVALID_SOURCE',
      'Source must be a valid ImageSource object'
    );
  }
  if (!keypoints || !Array.isArray(keypoints)) {
    throw new VisionUtilsException(
      'INVALID_INPUT',
      'Keypoints must be an array'
    );
  }
  const opts = {
    radius: options?.radius ?? 4,
    showConfidence: options?.showConfidence ?? false,
    saveFormat: options?.saveFormat ?? 'jpg',
    ...(options?.skeleton && { skeleton: options.skeleton }),
    ...(options?.colors && { colors: options.colors }),
  };
  return VisionUtils.drawKeypoints(source.value, keypoints, opts);
}

async function overlayMask(
  source: ImageSource,
  maskData: Uint8Array | number[],
  options: {
    width: number;
    height: number;
    alpha?: number;
    colorMap?: [number, number, number][];
    saveFormat?: 'jpg' | 'png';
  }
) {
  if (!source || typeof source !== 'object' || !source.type || !source.value) {
    throw new VisionUtilsException(
      'INVALID_SOURCE',
      'Source must be a valid ImageSource object'
    );
  }
  if (!maskData) {
    throw new VisionUtilsException('INVALID_INPUT', 'Mask data is required');
  }
  if (options.width <= 0 || options.height <= 0) {
    throw new VisionUtilsException(
      'INVALID_DIMENSIONS',
      'Width and height must be positive'
    );
  }
  const alpha = options.alpha ?? 0.5;
  if (alpha < 0 || alpha > 1) {
    throw new VisionUtilsException(
      'INVALID_ALPHA',
      'Alpha must be between 0 and 1'
    );
  }
  const opts = {
    width: options.width,
    height: options.height,
    alpha,
    saveFormat: options.saveFormat ?? 'jpg',
    ...(options.colorMap && { colorMap: options.colorMap }),
  };
  const dataArray = Array.isArray(maskData) ? maskData : Array.from(maskData);
  return VisionUtils.overlayMask(source.value, dataArray, opts);
}

async function overlayHeatmap(
  source: ImageSource,
  heatmapData: Float32Array | number[],
  options: {
    width: number;
    height: number;
    alpha?: number;
    colorScheme?: 'jet' | 'viridis' | 'hot' | 'cool';
    minValue?: number;
    maxValue?: number;
    saveFormat?: 'jpg' | 'png';
  }
) {
  if (!source || typeof source !== 'object' || !source.type || !source.value) {
    throw new VisionUtilsException(
      'INVALID_SOURCE',
      'Source must be a valid ImageSource object'
    );
  }
  if (!heatmapData) {
    throw new VisionUtilsException('INVALID_INPUT', 'Heatmap data is required');
  }
  if (options.width <= 0 || options.height <= 0) {
    throw new VisionUtilsException(
      'INVALID_DIMENSIONS',
      'Width and height must be positive'
    );
  }
  const alpha = options.alpha ?? 0.5;
  if (alpha < 0 || alpha > 1) {
    throw new VisionUtilsException(
      'INVALID_ALPHA',
      'Alpha must be between 0 and 1'
    );
  }
  const validSchemes = ['jet', 'viridis', 'hot', 'cool'];
  const scheme = options.colorScheme ?? 'jet';
  if (!validSchemes.includes(scheme)) {
    throw new VisionUtilsException(
      'INVALID_COLOR_SCHEME',
      `Invalid color scheme: ${scheme}`
    );
  }
  const opts = {
    width: options.width,
    height: options.height,
    alpha,
    colorScheme: scheme,
    saveFormat: options.saveFormat ?? 'jpg',
    ...(options.minValue !== undefined && { minValue: options.minValue }),
    ...(options.maxValue !== undefined && { maxValue: options.maxValue }),
  };
  const dataArray = Array.isArray(heatmapData)
    ? heatmapData
    : Array.from(heatmapData);
  return VisionUtils.overlayHeatmap(source.value, dataArray, opts);
}

describe('Drawing Utilities', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('drawBoxes', () => {
    it('should draw boxes on an image', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const detections: Detection[] = [
        { box: [100, 100, 200, 200], score: 0.9, label: 'person' },
        { box: [300, 300, 400, 400], score: 0.8, label: 'car' },
      ];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        boxesDrawn: 2,
        processingTimeMs: 25.5,
      };

      mockDrawBoxes.mockResolvedValue(expectedResult);

      const result = await drawBoxes(source, detections);

      expect(mockDrawBoxes).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        detections,
        {
          format: 'xyxy',
          lineWidth: 2,
          showLabels: true,
          showScores: true,
          fontSize: 14,
          saveFormat: 'jpg',
        }
      );
      expect(result.boxesDrawn).toBe(2);
    });

    it('should draw boxes with custom options', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const detections: Detection[] = [
        { box: [100, 100, 200, 200], score: 0.9, label: 'person' },
      ];
      const options: TestDrawBoxesOptions = {
        lineWidth: 4,
        showLabels: false,
        showScores: false,
        saveFormat: 'png',
      };
      const expectedResult = {
        imageUri: 'file:///path/to/result.png',
        boxesDrawn: 1,
        processingTimeMs: 20.0,
      };

      mockDrawBoxes.mockResolvedValue(expectedResult);

      const result = await drawBoxes(source, detections, options);

      expect(mockDrawBoxes).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        detections,
        {
          format: 'xyxy',
          lineWidth: 4,
          showLabels: false,
          showScores: false,
          fontSize: 14,
          saveFormat: 'png',
        }
      );
      expect(result.imageUri).toContain('.png');
    });

    it('should draw boxes with custom colors', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const detections: Detection[] = [
        { box: [100, 100, 200, 200], score: 0.9, label: 'person' },
      ];
      const options: TestDrawBoxesOptions = {
        colors: [
          [255, 0, 0],
          [0, 255, 0],
        ],
      };
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        boxesDrawn: 1,
        processingTimeMs: 25.5,
      };

      mockDrawBoxes.mockResolvedValue(expectedResult);

      await drawBoxes(source, detections, options);

      expect(mockDrawBoxes).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        detections,
        {
          format: 'xyxy',
          lineWidth: 2,
          showLabels: true,
          showScores: true,
          fontSize: 14,
          saveFormat: 'jpg',
          colors: [
            [255, 0, 0],
            [0, 255, 0],
          ],
        }
      );
    });

    it('should draw boxes with custom font size', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const detections: Detection[] = [
        { box: [100, 100, 200, 200], score: 0.9, label: 'person' },
      ];
      const options: TestDrawBoxesOptions = {
        fontSize: 20,
      };
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        boxesDrawn: 1,
        processingTimeMs: 25.5,
      };

      mockDrawBoxes.mockResolvedValue(expectedResult);

      await drawBoxes(source, detections, options);

      expect(mockDrawBoxes).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        detections,
        {
          format: 'xyxy',
          lineWidth: 2,
          showLabels: true,
          showScores: true,
          fontSize: 20,
          saveFormat: 'jpg',
        }
      );
    });

    it('should throw error for invalid source', async () => {
      const detections: Detection[] = [
        { box: [100, 100, 200, 200], score: 0.9 },
      ];

      await expect(
        drawBoxes('' as unknown as ImageSource, detections)
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid detections', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };

      await expect(
        drawBoxes(source, null as unknown as Detection[])
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should handle empty detections array', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const detections: Detection[] = [];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        boxesDrawn: 0,
        processingTimeMs: 5.0,
      };

      mockDrawBoxes.mockResolvedValue(expectedResult);

      const result = await drawBoxes(source, detections);

      expect(result.boxesDrawn).toBe(0);
    });

    it('should handle xywh format', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const detections: Detection[] = [
        { box: [100, 100, 100, 100], score: 0.9 },
      ];
      const options: TestDrawBoxesOptions = {
        format: 'xywh',
      };
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        boxesDrawn: 1,
        processingTimeMs: 25.5,
      };

      mockDrawBoxes.mockResolvedValue(expectedResult);

      await drawBoxes(source, detections, options);

      expect(mockDrawBoxes).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        detections,
        {
          format: 'xywh',
          lineWidth: 2,
          showLabels: true,
          showScores: true,
          fontSize: 14,
          saveFormat: 'jpg',
        }
      );
    });
  });

  describe('drawKeypoints', () => {
    it('should draw keypoints on an image', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const keypoints: Keypoint[][] = [
        [
          { x: 100, y: 100, confidence: 0.9, name: 'nose' },
          { x: 150, y: 120, confidence: 0.85, name: 'left_eye' },
          { x: 80, y: 120, confidence: 0.85, name: 'right_eye' },
        ],
      ];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        keypointsDrawn: 3,
        processingTimeMs: 20.0,
      };

      mockDrawKeypoints.mockResolvedValue(expectedResult);

      const result = await drawKeypoints(source, keypoints);

      expect(mockDrawKeypoints).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        keypoints,
        {
          radius: 4,
          showConfidence: false,
          saveFormat: 'jpg',
        }
      );
      expect(result.keypointsDrawn).toBe(3);
    });

    it('should draw keypoints with custom radius', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const keypoints: Keypoint[][] = [[{ x: 100, y: 100, confidence: 0.9 }]];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        keypointsDrawn: 1,
        processingTimeMs: 15.0,
      };

      mockDrawKeypoints.mockResolvedValue(expectedResult);

      await drawKeypoints(source, keypoints, { radius: 8 });

      expect(mockDrawKeypoints).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        keypoints,
        {
          radius: 8,
          showConfidence: false,
          saveFormat: 'jpg',
        }
      );
    });

    it('should draw keypoints with confidence shown', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const keypoints: Keypoint[][] = [
        [{ x: 100, y: 100, confidence: 0.9, name: 'nose' }],
      ];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        keypointsDrawn: 1,
        processingTimeMs: 18.0,
      };

      mockDrawKeypoints.mockResolvedValue(expectedResult);

      await drawKeypoints(source, keypoints, { showConfidence: true });

      expect(mockDrawKeypoints).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        keypoints,
        {
          radius: 4,
          showConfidence: true,
          saveFormat: 'jpg',
        }
      );
    });

    it('should draw keypoints with skeleton connections', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const keypoints: Keypoint[][] = [
        [
          { x: 100, y: 100, confidence: 0.9 },
          { x: 150, y: 120, confidence: 0.85 },
        ],
      ];
      const skeleton = [[0, 1]];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        keypointsDrawn: 2,
        processingTimeMs: 22.0,
      };

      mockDrawKeypoints.mockResolvedValue(expectedResult);

      await drawKeypoints(source, keypoints, { skeleton });

      expect(mockDrawKeypoints).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        keypoints,
        {
          radius: 4,
          showConfidence: false,
          saveFormat: 'jpg',
          skeleton,
        }
      );
    });

    it('should draw keypoints with custom colors', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const keypoints: Keypoint[][] = [[{ x: 100, y: 100, confidence: 0.9 }]];
      const colors: [number, number, number][] = [[255, 0, 0]];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        keypointsDrawn: 1,
        processingTimeMs: 15.0,
      };

      mockDrawKeypoints.mockResolvedValue(expectedResult);

      await drawKeypoints(source, keypoints, { colors });

      expect(mockDrawKeypoints).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        keypoints,
        {
          radius: 4,
          showConfidence: false,
          saveFormat: 'jpg',
          colors,
        }
      );
    });

    it('should throw error for invalid source', async () => {
      const keypoints: Keypoint[][] = [[{ x: 100, y: 100, confidence: 0.9 }]];

      await expect(
        drawKeypoints('' as unknown as ImageSource, keypoints)
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid keypoints', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };

      await expect(
        drawKeypoints(source, null as unknown as Keypoint[][])
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should handle multiple people keypoints', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const keypoints: Keypoint[][] = [
        [{ x: 100, y: 100, confidence: 0.9 }],
        [{ x: 300, y: 300, confidence: 0.85 }],
      ];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        keypointsDrawn: 2,
        processingTimeMs: 18.0,
      };

      mockDrawKeypoints.mockResolvedValue(expectedResult);

      const result = await drawKeypoints(source, keypoints);

      expect(result.keypointsDrawn).toBe(2);
    });
  });

  describe('overlayMask', () => {
    it('should overlay a segmentation mask on an image', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const maskData = new Uint8Array([0, 1, 1, 0, 2, 2]);
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        classesFound: [1, 2],
        processingTimeMs: 30.0,
      };

      mockOverlayMask.mockResolvedValue(expectedResult);

      const result = await overlayMask(source, maskData, {
        width: 3,
        height: 2,
      });

      expect(mockOverlayMask).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        Array.from(maskData),
        {
          width: 3,
          height: 2,
          alpha: 0.5,
          saveFormat: 'jpg',
        }
      );
      expect(result.classesFound).toContain(1);
      expect(result.classesFound).toContain(2);
    });

    it('should overlay mask with custom alpha', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const maskData = new Uint8Array([0, 1, 1, 0]);
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        classesFound: [1],
        processingTimeMs: 25.0,
      };

      mockOverlayMask.mockResolvedValue(expectedResult);

      await overlayMask(source, maskData, {
        width: 2,
        height: 2,
        alpha: 0.7,
      });

      expect(mockOverlayMask).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        Array.from(maskData),
        {
          width: 2,
          height: 2,
          alpha: 0.7,
          saveFormat: 'jpg',
        }
      );
    });

    it('should overlay mask with custom color map', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const maskData = new Uint8Array([0, 1, 1, 0]);
      const colorMap: [number, number, number][] = [
        [0, 0, 0],
        [255, 0, 0],
      ];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        classesFound: [1],
        processingTimeMs: 25.0,
      };

      mockOverlayMask.mockResolvedValue(expectedResult);

      await overlayMask(source, maskData, {
        width: 2,
        height: 2,
        colorMap,
      });

      expect(mockOverlayMask).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        Array.from(maskData),
        {
          width: 2,
          height: 2,
          alpha: 0.5,
          saveFormat: 'jpg',
          colorMap,
        }
      );
    });

    it('should throw error for invalid source', async () => {
      const maskData = new Uint8Array([0, 1, 1, 0]);

      await expect(
        overlayMask('' as unknown as ImageSource, maskData, {
          width: 2,
          height: 2,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid mask data', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };

      await expect(
        overlayMask(source, null as unknown as Uint8Array, {
          width: 2,
          height: 2,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid dimensions', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const maskData = new Uint8Array([0, 1, 1, 0]);

      await expect(
        overlayMask(source, maskData, { width: -1, height: 2 })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for alpha out of range', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const maskData = new Uint8Array([0, 1, 1, 0]);

      await expect(
        overlayMask(source, maskData, { width: 2, height: 2, alpha: 1.5 })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should handle regular array as mask data', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const maskData = [0, 1, 1, 0];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        classesFound: [1],
        processingTimeMs: 25.0,
      };

      mockOverlayMask.mockResolvedValue(expectedResult);

      await overlayMask(source, maskData as unknown as Uint8Array, {
        width: 2,
        height: 2,
      });

      expect(mockOverlayMask).toHaveBeenCalled();
    });
  });

  describe('overlayHeatmap', () => {
    it('should overlay a heatmap on an image', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const heatmapData = new Float32Array([0.0, 0.5, 0.8, 1.0]);
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        minValue: 0.0,
        maxValue: 1.0,
        processingTimeMs: 35.0,
      };

      mockOverlayHeatmap.mockResolvedValue(expectedResult);

      const result = await overlayHeatmap(source, heatmapData, {
        width: 2,
        height: 2,
      });

      expect(mockOverlayHeatmap).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        Array.from(heatmapData),
        {
          width: 2,
          height: 2,
          alpha: 0.5,
          colorScheme: 'jet',
          saveFormat: 'jpg',
        }
      );
      expect(result.minValue).toBe(0.0);
      expect(result.maxValue).toBe(1.0);
    });

    it('should overlay heatmap with custom alpha', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const heatmapData = new Float32Array([0.0, 0.5, 0.8, 1.0]);
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        minValue: 0.0,
        maxValue: 1.0,
        processingTimeMs: 35.0,
      };

      mockOverlayHeatmap.mockResolvedValue(expectedResult);

      await overlayHeatmap(source, heatmapData, {
        width: 2,
        height: 2,
        alpha: 0.3,
      });

      expect(mockOverlayHeatmap).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        Array.from(heatmapData),
        {
          width: 2,
          height: 2,
          alpha: 0.3,
          colorScheme: 'jet',
          saveFormat: 'jpg',
        }
      );
    });

    it('should overlay heatmap with viridis color scheme', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const heatmapData = new Float32Array([0.0, 0.5, 0.8, 1.0]);
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        minValue: 0.0,
        maxValue: 1.0,
        processingTimeMs: 35.0,
      };

      mockOverlayHeatmap.mockResolvedValue(expectedResult);

      await overlayHeatmap(source, heatmapData, {
        width: 2,
        height: 2,
        colorScheme: 'viridis',
      });

      expect(mockOverlayHeatmap).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        Array.from(heatmapData),
        {
          width: 2,
          height: 2,
          alpha: 0.5,
          colorScheme: 'viridis',
          saveFormat: 'jpg',
        }
      );
    });

    it('should overlay heatmap with custom value range', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const heatmapData = new Float32Array([0.0, 128.0, 200.0, 255.0]);
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        minValue: 0.0,
        maxValue: 255.0,
        processingTimeMs: 35.0,
      };

      mockOverlayHeatmap.mockResolvedValue(expectedResult);

      await overlayHeatmap(source, heatmapData, {
        width: 2,
        height: 2,
        minValue: 0.0,
        maxValue: 255.0,
      });

      expect(mockOverlayHeatmap).toHaveBeenCalledWith(
        '/path/to/image.jpg',
        Array.from(heatmapData),
        {
          width: 2,
          height: 2,
          alpha: 0.5,
          colorScheme: 'jet',
          saveFormat: 'jpg',
          minValue: 0.0,
          maxValue: 255.0,
        }
      );
    });

    it('should throw error for invalid source', async () => {
      const heatmapData = new Float32Array([0.0, 0.5, 0.8, 1.0]);

      await expect(
        overlayHeatmap('' as unknown as ImageSource, heatmapData, {
          width: 2,
          height: 2,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid heatmap data', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };

      await expect(
        overlayHeatmap(source, null as unknown as Float32Array, {
          width: 2,
          height: 2,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid dimensions', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const heatmapData = new Float32Array([0.0, 0.5, 0.8, 1.0]);

      await expect(
        overlayHeatmap(source, heatmapData, { width: 0, height: 2 })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for alpha out of range', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const heatmapData = new Float32Array([0.0, 0.5, 0.8, 1.0]);

      await expect(
        overlayHeatmap(source, heatmapData, {
          width: 2,
          height: 2,
          alpha: -0.1,
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should throw error for invalid color scheme', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const heatmapData = new Float32Array([0.0, 0.5, 0.8, 1.0]);

      await expect(
        overlayHeatmap(source, heatmapData, {
          width: 2,
          height: 2,
          colorScheme: 'invalid' as 'jet',
        })
      ).rejects.toThrow(VisionUtilsException);
    });

    it('should handle regular array as heatmap data', async () => {
      const source: ImageSource = {
        type: 'file',
        value: '/path/to/image.jpg',
      };
      const heatmapData = [0.0, 0.5, 0.8, 1.0];
      const expectedResult = {
        imageUri: 'file:///path/to/result.jpg',
        minValue: 0.0,
        maxValue: 1.0,
        processingTimeMs: 35.0,
      };

      mockOverlayHeatmap.mockResolvedValue(expectedResult);

      await overlayHeatmap(source, heatmapData as unknown as Float32Array, {
        width: 2,
        height: 2,
      });

      expect(mockOverlayHeatmap).toHaveBeenCalled();
    });
  });
});
