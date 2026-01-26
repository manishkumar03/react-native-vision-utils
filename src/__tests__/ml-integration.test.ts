import { mockGetPixelData } from './jest.setup';
import { getPixelData } from '../index';
import type { PixelDataResult } from '../types';

describe('ML Framework Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('PyTorch compatibility', () => {
    it('should provide ImageNet normalized data for PyTorch', async () => {
      const mockResult: PixelDataResult = {
        data: new Array(3 * 224 * 224).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        colorFormat: 'rgb',
        dataLayout: 'nchw',
        processingTimeMs: 10,
        shape: [1, 3, 224, 224],
      };
      mockGetPixelData.mockResolvedValue(mockResult);

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224, strategy: 'cover' },
        normalization: { preset: 'imagenet' },
        dataLayout: 'nchw',
      });

      expect(result.dataLayout).toBe('nchw');
      expect(result.channels).toBe(3);
      expect(result.width).toBe(224);
      expect(result.height).toBe(224);
    });

    it('should provide chw layout for PyTorch models', async () => {
      mockGetPixelData.mockResolvedValue({
        data: new Array(3 * 224 * 224).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'chw',
        processingTimeMs: 10,
        shape: [3, 224, 224],
      });

      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        dataLayout: 'chw',
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.dataLayout).toBe('chw');
    });

    it('should support bgr format for OpenCV models', async () => {
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
        colorFormat: 'bgr',
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.colorFormat).toBe('bgr');
    });
  });

  describe('TensorFlow compatibility', () => {
    it('should provide TensorFlow normalized data', async () => {
      mockGetPixelData.mockResolvedValue({
        data: new Array(224 * 224 * 3).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'nhwc',
        processingTimeMs: 10,
        shape: [1, 224, 224, 3],
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224 },
        normalization: { preset: 'tensorflow' },
        dataLayout: 'nhwc',
      });

      expect(result.dataLayout).toBe('nhwc');
    });

    it('should provide hwc layout for TensorFlow.js', async () => {
      mockGetPixelData.mockResolvedValue({
        data: new Array(224 * 224 * 3).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'hwc',
        processingTimeMs: 10,
      });

      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        dataLayout: 'hwc',
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.dataLayout).toBe('hwc');
    });
  });

  describe('ONNX Runtime compatibility', () => {
    it('should support nchw layout for ONNX models', async () => {
      mockGetPixelData.mockResolvedValue({
        data: new Array(1 * 3 * 224 * 224).fill(0),
        width: 224,
        height: 224,
        channels: 3,
        dataLayout: 'nchw',
        processingTimeMs: 10,
      });

      await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        dataLayout: 'nchw',
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.dataLayout).toBe('nchw');
    });
  });

  describe('Common model input sizes', () => {
    const commonSizes = [
      { name: 'MobileNet/ResNet', width: 224, height: 224 },
      { name: 'InceptionV3', width: 299, height: 299 },
      { name: 'EfficientNet-B0', width: 224, height: 224 },
      { name: 'EfficientNet-B7', width: 600, height: 600 },
      { name: 'YOLO', width: 416, height: 416 },
      { name: 'YOLOv5', width: 640, height: 640 },
      { name: 'ViT-Base', width: 224, height: 224 },
      { name: 'ViT-Large', width: 384, height: 384 },
    ];

    commonSizes.forEach(({ name, width, height }) => {
      it(`should support ${name} input size (${width}x${height})`, async () => {
        mockGetPixelData.mockResolvedValue({
          data: new Array(width * height * 3).fill(0),
          width,
          height,
          channels: 3,
          dataLayout: 'hwc',
          processingTimeMs: 10,
        });

        const result = await getPixelData({
          source: { type: 'url', value: 'https://example.com/image.jpg' },
          resize: { width, height },
        });

        expect(result.width).toBe(width);
        expect(result.height).toBe(height);
      });
    });
  });

  describe('Normalization presets', () => {
    it('should apply ImageNet normalization values', async () => {
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
        normalization: { preset: 'imagenet' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.normalization.preset).toBe('imagenet');
    });

    it('should apply custom normalization for specific models', async () => {
      const customMean = [0.5, 0.5, 0.5];
      const customStd = [0.5, 0.5, 0.5];

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
        normalization: {
          preset: 'custom',
          mean: customMean,
          std: customStd,
        },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.normalization.mean).toEqual(customMean);
      expect(calledOptions.normalization.std).toEqual(customStd);
    });

    it('should apply scale normalization (0-1 range)', async () => {
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
        normalization: { preset: 'scale' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.normalization.preset).toBe('scale');
    });

    it('should apply raw normalization (0-255 range)', async () => {
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
        normalization: { preset: 'raw' },
      });

      const calledOptions = mockGetPixelData.mock.calls[0][0];
      expect(calledOptions.normalization.preset).toBe('raw');
    });
  });

  describe('Grayscale models', () => {
    it('should support grayscale input for MNIST-like models', async () => {
      mockGetPixelData.mockResolvedValue({
        data: new Array(28 * 28 * 1).fill(0),
        width: 28,
        height: 28,
        channels: 1,
        dataLayout: 'hwc',
        processingTimeMs: 1,
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/digit.png' },
        colorFormat: 'grayscale',
        resize: { width: 28, height: 28 },
      });

      expect(result.channels).toBe(1);
    });
  });

  describe('Data validation helpers', () => {
    it('should return expected data size for hwc rgb', async () => {
      const width = 224;
      const height = 224;
      const channels = 3;
      const expectedSize = width * height * channels;

      mockGetPixelData.mockResolvedValue({
        data: new Array(expectedSize).fill(0),
        width,
        height,
        channels,
        dataLayout: 'hwc',
        processingTimeMs: 1,
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width, height },
        colorFormat: 'rgb',
      });

      expect(result.data.length).toBe(expectedSize);
    });

    it('should return expected data size for rgba', async () => {
      const width = 224;
      const height = 224;
      const channels = 4;
      const expectedSize = width * height * channels;

      mockGetPixelData.mockResolvedValue({
        data: new Array(expectedSize).fill(0),
        width,
        height,
        channels,
        dataLayout: 'hwc',
        processingTimeMs: 1,
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.png' },
        resize: { width, height },
        colorFormat: 'rgba',
      });

      expect(result.data.length).toBe(expectedSize);
    });

    it('should return expected data size for grayscale', async () => {
      const width = 224;
      const height = 224;
      const channels = 1;
      const expectedSize = width * height * channels;

      mockGetPixelData.mockResolvedValue({
        data: new Array(expectedSize).fill(0),
        width,
        height,
        channels,
        dataLayout: 'hwc',
        processingTimeMs: 1,
      });

      const result = await getPixelData({
        source: { type: 'url', value: 'https://example.com/image.jpg' },
        resize: { width, height },
        colorFormat: 'grayscale',
      });

      expect(result.data.length).toBe(expectedSize);
    });
  });
});
