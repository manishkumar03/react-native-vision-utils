import React, { useState, useCallback } from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ActivityIndicator,
  Alert,
  Image,
  Dimensions,
} from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import {
  getPixelData,
  batchGetPixelData,
  getImageMetadata,
  validateImage,
  getImageStatistics,
  applyAugmentations,
  extractChannel,
  tensorToImage,
  fiveCrop,
  clearCache,
  getCacheStats,
  quantize,
  dequantize,
  calculateQuantizationParams,
  getLabel,
  getTopLabels,
  getDatasetInfo,
  getAvailableDatasets,
  processCameraFrame,
  convertBoxFormat,
  scaleBoxes,
  clipBoxes,
  calculateIoU,
  nonMaxSuppression,
  letterbox,
  reverseLetterbox,
  drawBoxes,
  drawKeypoints,
  overlayHeatmap,
  detectBlur,
  extractVideoFrames,
  extractGrid,
  randomCrop,
  validateTensor,
  assembleBatch,
  type PixelDataResult,
  type ColorFormat,
  type DataLayout,
  type NormalizationPreset,
  type LabelDataset,
  type BoundingBox,
  type Detection,
  type DrawableBox,
} from 'react-native-vision-utils';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const IMAGE_SIZE = SCREEN_WIDTH - 40;

// Sample image URLs for testing (using stable public test images with seed for consistency)
const SAMPLE_IMAGES = [
  'https://picsum.photos/seed/vision1/400/400', // Consistent square image
  'https://picsum.photos/seed/vision2/600/300', // Consistent wide image
  'https://picsum.photos/seed/vision3/300/600', // Consistent tall image
];

interface ResultData {
  label: string;
  result: PixelDataResult | null;
  error: string | null;
}

const App: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('Processing...');
  const [results, setResults] = useState<ResultData[]>([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [processedImageUri, setProcessedImageUri] = useState<string | null>(
    null
  );

  const currentImage = SAMPLE_IMAGES[currentImageIndex]!;

  // Cycle through sample images
  const cycleImage = useCallback(() => {
    setCurrentImageIndex((prev) => (prev + 1) % SAMPLE_IMAGES.length);
    setProcessedImageUri(null);
  }, []);

  // Test basic URL loading with RGB
  const testBasicRGB = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getPixelData({
        source: { type: 'url', value: currentImage },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224, strategy: 'cover' },
        normalization: { preset: 'scale' },
        dataLayout: 'hwc',
      });
      setResults((prev) => [
        ...prev,
        { label: 'Basic RGB (224x224)', result, error: null },
      ]);
      Alert.alert(
        'Success',
        `Processed ${result.width}x${result.height} image with ${
          result.channels
        } channels in ${result.processingTimeMs.toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setResults((prev) => [
        ...prev,
        { label: 'Basic RGB', result: null, error: errorMessage },
      ]);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test ImageNet normalization (PyTorch style)
  const testImageNet = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getPixelData({
        source: { type: 'url', value: currentImage },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224, strategy: 'cover' },
        normalization: { preset: 'imagenet' },
        dataLayout: 'chw', // PyTorch format
      });
      setResults((prev) => [
        ...prev,
        { label: 'ImageNet CHW (PyTorch)', result, error: null },
      ]);

      // Sample some normalized values
      const dataArray = Array.from(result.data as ArrayLike<number>);
      const sampleValues = dataArray.slice(0, 5).map((v) => v.toFixed(4));
      Alert.alert(
        'ImageNet Normalized',
        `Shape: [${result.channels}, ${result.height}, ${
          result.width
        }]\nSample values: [${sampleValues.join(', ')}]`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setResults((prev) => [
        ...prev,
        { label: 'ImageNet CHW', result: null, error: errorMessage },
      ]);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test TensorFlow normalization
  const testTensorFlow = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getPixelData({
        source: { type: 'url', value: currentImage },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224, strategy: 'cover' },
        normalization: { preset: 'tensorflow' },
        dataLayout: 'nhwc', // TensorFlow format
      });
      setResults((prev) => [
        ...prev,
        { label: 'TensorFlow NHWC', result, error: null },
      ]);

      const dataArray = Array.from(result.data as ArrayLike<number>);
      const sampleValues = dataArray.slice(0, 5).map((v) => v.toFixed(4));
      Alert.alert(
        'TensorFlow Normalized',
        `Shape: [1, ${result.height}, ${result.width}, ${
          result.channels
        }]\nSample values (range -1 to 1): [${sampleValues.join(', ')}]`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setResults((prev) => [
        ...prev,
        { label: 'TensorFlow NHWC', result: null, error: errorMessage },
      ]);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test grayscale conversion
  const testGrayscale = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getPixelData({
        source: { type: 'url', value: currentImage },
        colorFormat: 'grayscale',
        resize: { width: 28, height: 28, strategy: 'cover' },
        normalization: { preset: 'scale' },
        dataLayout: 'hwc',
      });
      setResults((prev) => [
        ...prev,
        { label: 'Grayscale 28x28', result, error: null },
      ]);
      Alert.alert(
        'Grayscale',
        `${result.width}x${result.height} with ${result.channels} channel(s)\nTotal pixels: ${result.data.length}`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setResults((prev) => [
        ...prev,
        { label: 'Grayscale', result: null, error: errorMessage },
      ]);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test batch processing
  const testBatchProcessing = useCallback(async () => {
    setLoading(true);
    try {
      const batchResult = await batchGetPixelData(
        SAMPLE_IMAGES.map((url) => ({
          source: { type: 'url' as const, value: url },
          colorFormat: 'rgb' as ColorFormat,
          resize: { width: 224, height: 224, strategy: 'cover' as const },
          normalization: { preset: 'scale' as NormalizationPreset },
          dataLayout: 'hwc' as DataLayout,
        })),
        { concurrency: 2 }
      );

      const successCount = batchResult.results.filter(
        (r) => !('error' in r)
      ).length;

      Alert.alert(
        'Batch Processing Complete',
        `Processed ${successCount}/${
          batchResult.results.length
        } images\nTotal time: ${batchResult.totalTimeMs.toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Batch Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test different resize strategies
  const testResizeStrategies = useCallback(async () => {
    setLoading(true);
    try {
      const strategies = ['cover', 'contain', 'stretch', 'letterbox'] as const;

      for (const strategy of strategies) {
        const result = await getPixelData({
          source: { type: 'url', value: currentImage },
          colorFormat: 'rgb',
          resize: {
            width: 224,
            height: 224,
            strategy,
            padColor: strategy === 'contain' ? [128, 128, 128, 255] : undefined,
            letterboxColor:
              strategy === 'letterbox' ? [128, 128, 128] : undefined,
          },
          normalization: { preset: 'scale' },
          dataLayout: 'hwc',
        });

        setResults((prev) => [
          ...prev,
          { label: `Resize: ${strategy}`, result, error: null },
        ]);
      }

      Alert.alert(
        'Resize Strategies',
        'All resize strategies tested successfully!\nIncluding letterbox with gray padding.'
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test image metadata
  const testImageMetadata = useCallback(async () => {
    setLoading(true);
    try {
      const metadata = await getImageMetadata({
        type: 'url',
        value: currentImage,
      });
      Alert.alert(
        'Image Metadata',
        `Size: ${metadata.width}x${metadata.height}\nFormat: ${
          metadata.format
        }\nColor Space: ${metadata.colorSpace}\nHas Alpha: ${
          metadata.hasAlpha
        }\nBits: ${metadata.bitsPerComponent}\nFile Size: ${
          metadata.fileSize
            ? `${(metadata.fileSize / 1024).toFixed(1)}KB`
            : 'N/A'
        }`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test image validation
  const testValidation = useCallback(async () => {
    setLoading(true);
    try {
      const validation = await validateImage({
        type: 'url',
        value: currentImage,
      });
      // Handle different response formats - native returns width/height at top level
      const width =
        validation.metadata?.width ??
        (validation as unknown as { width: number }).width;
      const height =
        validation.metadata?.height ??
        (validation as unknown as { height: number }).height;
      Alert.alert(
        'Image Validation',
        `Valid: ${validation.isValid}\nErrors: ${
          validation.errors.length > 0 ? validation.errors.join(', ') : 'None'
        }\nSize: ${width}x${height}`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test image statistics
  const testStatistics = useCallback(async () => {
    setLoading(true);
    try {
      const stats = await getImageStatistics({
        type: 'url',
        value: currentImage,
      });
      // Handle both array and single number for mean/std (native returns single number)
      const meanStr = Array.isArray(stats.mean)
        ? stats.mean.map((v) => v.toFixed(2)).join(', ')
        : (stats.mean as number).toFixed(2);
      const stdStr = Array.isArray(stats.std)
        ? stats.std.map((v) => v.toFixed(2)).join(', ')
        : (stats.std as number).toFixed(2);
      Alert.alert(
        'Image Statistics',
        `Mean: ${meanStr}\nStd: ${stdStr}\nMin: ${stats.min}\nMax: ${stats.max}`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test augmentations
  const testAugmentations = useCallback(async () => {
    setLoading(true);
    try {
      const result = await applyAugmentations(
        { type: 'url', value: currentImage },
        {
          horizontalFlip: true,
          rotation: 15,
          brightness: 0.2,
          contrast: 1.1,
          saturation: 1.3,
        }
      );
      if (result.base64) {
        setProcessedImageUri(`data:image/png;base64,${result.base64}`);
      }
      const timeMs =
        result && typeof result.processingTimeMs === 'number'
          ? result.processingTimeMs
          : 0;
      Alert.alert(
        'Augmentations Applied',
        `Processed in ${timeMs.toFixed(2)}ms\nOutput: ${
          result.base64 ? 'Generated' : 'N/A'
        }`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test five crop extraction
  const testFiveCrop = useCallback(async () => {
    setLoading(true);
    try {
      const result = await fiveCrop(
        {
          source: { type: 'url', value: currentImage },
          colorFormat: 'rgb',
          normalization: { preset: 'scale' },
          dataLayout: 'hwc',
        },
        { width: 100, height: 100 }
      );
      const crops = (result as any).crops || (result as any).results || [];
      const totalTime =
        result && typeof result.totalTimeMs === 'number'
          ? result.totalTimeMs
          : 0;
      Alert.alert(
        'Five Crop Extract',
        `Extracted ${crops.length} crops\nTotal time: ${totalTime.toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test extract channel
  const testExtractChannel = useCallback(async () => {
    setLoading(true);
    try {
      // First get RGB pixel data
      const pixelData = await getPixelData({
        source: { type: 'url', value: currentImage },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224, strategy: 'cover' },
        normalization: { preset: 'scale' },
        dataLayout: 'hwc',
      });

      // Extract the red channel (index 0)
      const redChannel = await extractChannel(pixelData, 0);
      Alert.alert(
        'Extract Channel',
        `Extracted red channel from ${pixelData.channels}-channel image\nResult: ${redChannel.width}x${redChannel.height}x${redChannel.channels}\nData length: ${redChannel.data.length}`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test quantization (per-tensor uint8)
  const testQuantization = useCallback(async () => {
    setLoading(true);
    try {
      // Get normalized float pixel data
      const pixelData = await getPixelData({
        source: { type: 'url', value: currentImage },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224, strategy: 'cover' },
        normalization: { preset: 'scale' }, // 0-1 range
        dataLayout: 'hwc',
      });

      // Calculate optimal quantization params
      const params = await calculateQuantizationParams(
        Array.from(pixelData.data),
        {
          mode: 'per-tensor',
          dtype: 'uint8',
        }
      );

      // Quantize to uint8
      const quantized = await quantize(Array.from(pixelData.data), {
        mode: 'per-tensor',
        dtype: 'uint8',
        scale: params.scale as number,
        zeroPoint: params.zeroPoint as number,
      });

      // Sample some quantized values
      const sampleValues = Array.from(quantized.data.slice(0, 5));
      const minVal = Array.isArray(params.min) ? params.min[0] : params.min;
      const maxVal = Array.isArray(params.max) ? params.max[0] : params.max;

      Alert.alert(
        'Quantization (Per-Tensor)',
        `Input: ${pixelData.data.length} float32 values\n` +
          `Output: ${quantized.data.length} uint8 values\n` +
          `Scale: ${(params.scale as number).toFixed(6)}\n` +
          `ZeroPoint: ${params.zeroPoint}\n` +
          `Sample values: [${sampleValues.join(', ')}]\n` +
          `Data range: [${minVal?.toFixed(3)}, ${maxVal?.toFixed(3)}]`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test per-channel quantization (int8)
  const testPerChannelQuantization = useCallback(async () => {
    setLoading(true);
    try {
      // Get normalized float pixel data in CHW layout
      const pixelData = await getPixelData({
        source: { type: 'url', value: currentImage },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224, strategy: 'cover' },
        normalization: { preset: 'imagenet' }, // ImageNet normalization
        dataLayout: 'chw', // CHW for per-channel
      });

      // Calculate per-channel quantization params
      const params = await calculateQuantizationParams(
        Array.from(pixelData.data),
        {
          mode: 'per-channel',
          dtype: 'int8',
          channels: 3,
          dataLayout: 'chw',
        }
      );

      // Quantize with per-channel params
      const quantized = await quantize(Array.from(pixelData.data), {
        mode: 'per-channel',
        dtype: 'int8',
        scale: params.scale as number[],
        zeroPoint: params.zeroPoint as number[],
        channels: 3,
        dataLayout: 'chw',
      });

      const scales = (params.scale as number[]).map((s) => s.toFixed(4));

      Alert.alert(
        'Quantization (Per-Channel)',
        `Mode: per-channel int8\n` +
          `Channels: 3 (RGB)\n` +
          `Per-channel scales: [${scales.join(', ')}]\n` +
          `Per-channel zeroPoints: [${(params.zeroPoint as number[]).join(
            ', '
          )}]\n` +
          `Output: ${quantized.data.length} int8 values`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test dequantization roundtrip
  const testDequantization = useCallback(async () => {
    setLoading(true);
    try {
      // Get normalized float pixel data
      const pixelData = await getPixelData({
        source: { type: 'url', value: currentImage },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224, strategy: 'cover' },
        normalization: { preset: 'scale' },
        dataLayout: 'hwc',
      });

      const originalData = Array.from(pixelData.data);
      const originalSample = originalData.slice(0, 3).map((v) => v.toFixed(4));

      // Quantize
      const quantized = await quantize(originalData, {
        mode: 'per-tensor',
        dtype: 'uint8',
        scale: 0.00392157, // 1/255
        zeroPoint: 0,
      });

      // Dequantize back to float
      const dequantized = await dequantize(Array.from(quantized.data), {
        mode: 'per-tensor',
        dtype: 'uint8',
        scale: 0.00392157,
        zeroPoint: 0,
      });

      const dequantizedSample = Array.from(dequantized.data.slice(0, 3)).map(
        (v) => v.toFixed(4)
      );

      // Calculate error
      let totalError = 0;
      for (let i = 0; i < Math.min(1000, originalData.length); i++) {
        totalError += Math.abs(originalData[i]! - dequantized.data[i]!);
      }
      const avgError = totalError / Math.min(1000, originalData.length);

      Alert.alert(
        'Dequantization Roundtrip',
        `Original (first 3): [${originalSample.join(', ')}]\n` +
          `Dequantized (first 3): [${dequantizedSample.join(', ')}]\n` +
          `Avg reconstruction error: ${avgError.toFixed(6)}\n` +
          `(Lower is better, ~0.002 expected for uint8)`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test tensor to image conversion
  const testTensorToImage = useCallback(async () => {
    setLoading(true);
    try {
      // First get pixel data, then convert back to image
      const pixelData = await getPixelData({
        source: { type: 'url', value: currentImage },
        colorFormat: 'rgb',
        resize: { width: 224, height: 224, strategy: 'cover' },
        normalization: { preset: 'scale' },
        dataLayout: 'hwc',
      });

      const imageResult = await tensorToImage(pixelData, {
        format: 'png',
      });

      if (imageResult.base64) {
        setProcessedImageUri(`data:image/png;base64,${imageResult.base64}`);
      }
      Alert.alert(
        'Tensor to Image',
        `Converted tensor back to image!\nSize: ${pixelData.width}x${pixelData.height}`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test cache statistics
  const testCacheStats = useCallback(async () => {
    try {
      const stats = await getCacheStats();
      Alert.alert(
        'Cache Statistics',
        `Hits: ${stats.hitCount}\nMisses: ${stats.missCount}\nSize: ${stats.size}/${stats.maxSize}`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    }
  }, []);

  // Clear cache
  const testClearCache = useCallback(async () => {
    try {
      await clearCache();
      Alert.alert(
        'Cache Cleared',
        'Image cache has been cleared successfully.'
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    }
  }, []);

  // Test label database - get single label
  const testGetLabel = useCallback(async () => {
    setLoading(true);
    try {
      // Get label with metadata
      const labelInfo = await getLabel(0, 'coco', true);

      const info = labelInfo as {
        index: number;
        name: string;
        displayName: string;
        supercategory?: string;
      };

      Alert.alert(
        'Label Database - Single Label',
        `Index 0 in COCO:\n` +
          `Name: ${info.name}\n` +
          `Display: ${info.displayName}\n` +
          `Category: ${info.supercategory || 'N/A'}`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test label database - get top labels from mock scores
  const testTopLabels = useCallback(async () => {
    setLoading(true);
    try {
      // Create mock prediction scores (80 classes for COCO)
      const mockScores = Array(80).fill(0.01);
      mockScores[0] = 0.85; // person
      mockScores[15] = 0.72; // cat
      mockScores[16] = 0.45; // dog
      mockScores[2] = 0.38; // car

      const topLabels = await getTopLabels(mockScores, {
        dataset: 'coco',
        k: 5,
        minConfidence: 0.1,
        includeMetadata: true,
      });

      const labelStrings = topLabels
        .map(
          (l: { label: string; confidence: number }) =>
            `${l.label}: ${(l.confidence * 100).toFixed(1)}%`
        )
        .join('\n');

      Alert.alert(
        'Label Database - Top Predictions',
        `Top ${topLabels.length} from mock scores:\n\n${labelStrings}`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test available datasets
  const testAvailableDatasets = useCallback(async () => {
    setLoading(true);
    try {
      const datasets = await getAvailableDatasets();
      const infos: string[] = [];

      for (const dataset of datasets.slice(0, 4)) {
        const info = await getDatasetInfo(dataset as LabelDataset);
        infos.push(`${info.name}: ${info.numClasses} classes`);
      }

      Alert.alert(
        'Available Datasets',
        `Found ${datasets.length} datasets:\n\n${infos.join(
          '\n'
        )}\n\n...and more`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test camera frame processing (simulated with base64 data)
  const testCameraFrameProcessing = useCallback(async () => {
    setLoading(true);
    try {
      // Create a small simulated RGB frame (8x8 pixels)
      const width = 8;
      const height = 8;
      const rgbData = new Uint8Array(width * height * 3);

      // Fill with a gradient pattern
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 3;
          rgbData[idx] = Math.floor((x / width) * 255); // R
          rgbData[idx + 1] = Math.floor((y / height) * 255); // G
          rgbData[idx + 2] = 128; // B
        }
      }

      // Convert to base64
      const binaryStr = String.fromCharCode(...rgbData);
      const base64Data = btoa(binaryStr);

      const result = await processCameraFrame(
        {
          width,
          height,
          pixelFormat: 'rgb',
          bytesPerRow: width * 3,
          dataBase64: base64Data,
        },
        {
          outputWidth: 4,
          outputHeight: 4,
          normalize: true,
          outputFormat: 'rgb',
          mean: [0.485, 0.456, 0.406],
          std: [0.229, 0.224, 0.225],
        }
      );

      const sampleValues = result.tensor.slice(0, 3).map((v) => v.toFixed(3));

      Alert.alert(
        'Camera Frame Processing',
        `Input: ${width}x${height} RGB\n` +
          `Output: ${result.width}x${result.height}\n` +
          `Shape: [${result.shape.join(', ')}]\n` +
          `Time: ${result.processingTimeMs.toFixed(2)}ms\n` +
          `Sample (normalized): [${sampleValues.join(', ')}]`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test bounding box format conversion
  const testBoxFormatConversion = useCallback(async () => {
    setLoading(true);
    try {
      // Convert YOLO format (center x, center y, width, height) to corners (xyxy)
      const cxcywhBoxes: BoundingBox[] = [
        [320, 240, 100, 80], // center at (320, 240), size 100x80
        [150, 150, 60, 60], // center at (150, 150), size 60x60
      ];

      const result = await convertBoxFormat(cxcywhBoxes, {
        fromFormat: 'cxcywh',
        toFormat: 'xyxy',
      });

      const boxStr = result.boxes
        .map((b) => `[${b.map((v) => Number(v).toFixed(0)).join(', ')}]`)
        .join('\n');

      Alert.alert(
        'Box Format Conversion',
        `Input (cxcywh):\n${cxcywhBoxes
          .map((b) => `[${b.join(', ')}]`)
          .join('\n')}\n\n` +
          `Output (xyxy):\n${boxStr}\n\n` +
          `Time: ${Number(result.processingTimeMs).toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test bounding box scaling
  const testBoxScaling = useCallback(async () => {
    setLoading(true);
    try {
      // Scale boxes from 640x640 model input to 1920x1080 original
      const boxes: BoundingBox[] = [
        [100, 100, 200, 200],
        [300, 200, 400, 350],
      ];

      const result = await scaleBoxes(boxes, {
        fromWidth: 640,
        fromHeight: 640,
        toWidth: 1920,
        toHeight: 1080,
        format: 'xyxy',
      });

      const originalStr = boxes.map((b) => `[${b.join(', ')}]`).join('\n');
      const scaledStr = result.boxes
        .map((b) => `[${b.map((v) => Number(v).toFixed(0)).join(', ')}]`)
        .join('\n');

      Alert.alert(
        'Box Scaling',
        `640x640 → 1920x1080\n\n` +
          `Original:\n${originalStr}\n\n` +
          `Scaled:\n${scaledStr}\n\n` +
          `Time: ${Number(result.processingTimeMs).toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test IoU calculation
  const testIoUCalculation = useCallback(async () => {
    setLoading(true);
    try {
      const box1: BoundingBox = [100, 100, 200, 200];
      const box2: BoundingBox = [150, 150, 250, 250];

      const result = await calculateIoU(box1, box2, 'xyxy');

      Alert.alert(
        'IoU Calculation',
        `Box 1: [${box1.join(', ')}]\n` +
          `Box 2: [${box2.join(', ')}]\n\n` +
          `IoU: ${(Number(result.iou) * 100).toFixed(1)}%\n` +
          `Intersection: ${Number(result.intersection).toFixed(0)} px²\n` +
          `Union: ${Number(result.union).toFixed(0)} px²\n\n` +
          `Time: ${Number(result.processingTimeMs).toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test Non-Maximum Suppression
  const testNMS = useCallback(async () => {
    setLoading(true);
    try {
      const detections: Detection[] = [
        { box: [100, 100, 200, 200], score: 0.9, classIndex: 0 },
        { box: [110, 110, 210, 210], score: 0.8, classIndex: 0 }, // Overlaps with first
        { box: [300, 300, 400, 400], score: 0.7, classIndex: 1 },
        { box: [305, 305, 405, 405], score: 0.6, classIndex: 1 }, // Overlaps with third
        { box: [500, 100, 600, 200], score: 0.5, classIndex: 0 },
      ];

      const result = await nonMaxSuppression(detections, {
        iouThreshold: 0.5,
        scoreThreshold: 0.3,
        maxDetections: 100,
      });

      Alert.alert(
        'Non-Maximum Suppression',
        `Input: ${detections.length} detections\n` +
          `Output: ${result.detections.length} detections\n\n` +
          `Kept indices: [${result.indices.join(', ')}]\n` +
          `Scores: [${result.detections
            .map((d) => Number(d.score).toFixed(2))
            .join(', ')}]\n\n` +
          `Time: ${Number(result.processingTimeMs).toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test letterbox padding
  const testLetterbox = useCallback(async () => {
    setLoading(true);
    try {
      const result = await letterbox(
        { type: 'url', value: currentImage },
        {
          targetWidth: 640,
          targetHeight: 640,
          fillColor: [114, 114, 114], // YOLO gray
        }
      );

      if (result.imageBase64) {
        setProcessedImageUri(`data:image/jpeg;base64,${result.imageBase64}`);
      }

      const info = result.letterboxInfo;
      Alert.alert(
        'Letterbox Padding',
        `Original: ${info.originalSize[0]}x${info.originalSize[1]}\n` +
          `Letterboxed: ${info.letterboxedSize[0]}x${info.letterboxedSize[1]}\n` +
          `Scale: ${Number(info.scale).toFixed(4)}\n` +
          `Offset: [${info.offset
            .map((o) => Number(o).toFixed(1))
            .join(', ')}]\n\n` +
          `Time: ${
            result.processingTimeMs
              ? Number(result.processingTimeMs).toFixed(2)
              : 'N/A'
          }ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test reverse letterbox
  const testReverseLetterbox = useCallback(async () => {
    setLoading(true);
    try {
      // First letterbox the image
      const lb = await letterbox(
        { type: 'url', value: currentImage },
        { targetWidth: 640, targetHeight: 640 }
      );

      // Simulate detections in letterboxed space
      const boxesInLetterbox: BoundingBox[] = [
        [100, 150, 200, 250],
        [400, 300, 500, 450],
      ];

      // Reverse transform
      const result = await reverseLetterbox(boxesInLetterbox, {
        scale: lb.letterboxInfo.scale,
        offset: lb.letterboxInfo.offset,
        originalSize: lb.letterboxInfo.originalSize,
        format: 'xyxy',
      });

      const letterboxStr = boxesInLetterbox
        .map((b) => `[${b.join(', ')}]`)
        .join('\n');
      const originalStr = result.boxes
        .map((b) => `[${b.map((v) => Number(v).toFixed(0)).join(', ')}]`)
        .join('\n');

      // Show more precision for fast operations
      const timeStr =
        result.processingTimeMs != null
          ? result.processingTimeMs < 0.01
            ? `${(result.processingTimeMs * 1000).toFixed(2)}µs`
            : `${Number(result.processingTimeMs).toFixed(2)}ms`
          : 'N/A';

      Alert.alert(
        'Reverse Letterbox',
        `Boxes in 640x640 space:\n${letterboxStr}\n\n` +
          `Boxes in original space:\n${originalStr}\n\n` +
          `Time: ${timeStr}`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test drawing boxes
  const testDrawBoxes = useCallback(async () => {
    setLoading(true);
    try {
      // First get the image as base64
      const metadata = await getImageMetadata({
        type: 'url',
        value: currentImage,
      });

      // Draw some sample detections
      const boxes: DrawableBox[] = [
        {
          box: [50, 50, 150, 150],
          label: 'person',
          score: 0.95,
          classIndex: 0,
        },
        {
          box: [180, 80, 280, 200],
          label: 'dog',
          score: 0.87,
          classIndex: 16,
        },
        {
          box: [300, 100, 380, 180],
          label: 'cat',
          score: 0.72,
          classIndex: 15,
          color: [255, 100, 100], // Custom color
        },
      ];

      const result = await drawBoxes(
        { type: 'url', value: currentImage },
        boxes,
        {
          lineWidth: 3,
          fontSize: 14,
          drawLabels: true,
          labelBackgroundAlpha: 0.7,
        }
      );

      if (result.imageBase64) {
        setProcessedImageUri(`data:image/jpeg;base64,${result.imageBase64}`);
      }

      Alert.alert(
        'Draw Boxes',
        `Drew ${boxes.length} boxes on ${metadata.width}x${metadata.height} image\n\n` +
          `Time: ${Number(result.processingTimeMs).toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test drawing keypoints
  const testDrawKeypoints = useCallback(async () => {
    setLoading(true);
    try {
      // Sample keypoints (simplified pose)
      const keypoints = [
        { x: 200, y: 80, confidence: 0.95 }, // 0: nose
        { x: 190, y: 70, confidence: 0.9 }, // 1: left_eye
        { x: 210, y: 70, confidence: 0.9 }, // 2: right_eye
        { x: 180, y: 75, confidence: 0.85 }, // 3: left_ear
        { x: 220, y: 75, confidence: 0.85 }, // 4: right_ear
        { x: 160, y: 120, confidence: 0.88 }, // 5: left_shoulder
        { x: 240, y: 120, confidence: 0.88 }, // 6: right_shoulder
        { x: 140, y: 180, confidence: 0.82 }, // 7: left_elbow
        { x: 260, y: 180, confidence: 0.82 }, // 8: right_elbow
        { x: 130, y: 240, confidence: 0.75 }, // 9: left_wrist
        { x: 270, y: 240, confidence: 0.75 }, // 10: right_wrist
        { x: 180, y: 200, confidence: 0.9 }, // 11: left_hip
        { x: 220, y: 200, confidence: 0.9 }, // 12: right_hip
      ];

      // COCO skeleton connections
      const skeleton = [
        { from: 0, to: 1 }, // nose to left_eye
        { from: 0, to: 2 }, // nose to right_eye
        { from: 1, to: 3 }, // left_eye to left_ear
        { from: 2, to: 4 }, // right_eye to right_ear
        { from: 5, to: 6 }, // left_shoulder to right_shoulder
        { from: 5, to: 7 }, // left_shoulder to left_elbow
        { from: 6, to: 8 }, // right_shoulder to right_elbow
        { from: 7, to: 9 }, // left_elbow to left_wrist
        { from: 8, to: 10 }, // right_elbow to right_wrist
        { from: 5, to: 11 }, // left_shoulder to left_hip
        { from: 6, to: 12 }, // right_shoulder to right_hip
        { from: 11, to: 12 }, // left_hip to right_hip
      ];

      const result = await drawKeypoints(
        { type: 'url', value: currentImage },
        keypoints,
        {
          pointRadius: 6,
          lineWidth: 2,
          minConfidence: 0.5,
          skeleton,
        }
      );

      if (result.imageBase64) {
        setProcessedImageUri(`data:image/jpeg;base64,${result.imageBase64}`);
      }

      Alert.alert(
        'Draw Keypoints',
        `Drew ${result.pointsDrawn} keypoints and ${result.connectionsDrawn} connections\n\n` +
          `Time: ${Number(result.processingTimeMs).toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test heatmap overlay
  const testHeatmapOverlay = useCallback(async () => {
    setLoading(true);
    try {
      // Create a sample 14x14 attention heatmap (like ViT patches)
      const heatmapWidth = 14;
      const heatmapHeight = 14;
      const heatmap = new Array(heatmapWidth * heatmapHeight).fill(0);

      // Create a gaussian-like attention pattern in the center
      const centerX = heatmapWidth / 2;
      const centerY = heatmapHeight / 2;
      for (let y = 0; y < heatmapHeight; y++) {
        for (let x = 0; x < heatmapWidth; x++) {
          const dx = x - centerX;
          const dy = y - centerY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          heatmap[y * heatmapWidth + x] = Math.exp(-(dist * dist) / 20);
        }
      }

      const result = await overlayHeatmap(
        { type: 'url', value: currentImage },
        heatmap,
        {
          heatmapWidth,
          heatmapHeight,
          alpha: 0.5,
          colorScheme: 'jet',
        }
      );

      if (result.imageBase64) {
        setProcessedImageUri(`data:image/jpeg;base64,${result.imageBase64}`);
      }

      Alert.alert(
        'Heatmap Overlay',
        `Overlaid ${heatmapWidth}x${heatmapHeight} heatmap with jet colorscheme\n\n` +
          `Time: ${Number(result.processingTimeMs).toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test clip boxes
  const testClipBoxes = useCallback(async () => {
    setLoading(true);
    try {
      // Boxes that extend beyond boundaries
      const boxes: BoundingBox[] = [
        [-20, 50, 150, 200], // Extends left
        [100, -10, 250, 150], // Extends top
        [500, 300, 700, 500], // Extends right/bottom (assuming 640x480)
      ];

      const result = await clipBoxes(boxes, {
        width: 640,
        height: 480,
        format: 'xyxy',
      });

      const originalStr = boxes.map((b) => `[${b.join(', ')}]`).join('\n');
      const clippedStr = result.boxes
        .map((b) => `[${b.map((v) => Number(v).toFixed(0)).join(', ')}]`)
        .join('\n');

      Alert.alert(
        'Clip Boxes',
        `Image: 640x480\n\n` +
          `Original (out of bounds):\n${originalStr}\n\n` +
          `Clipped:\n${clippedStr}\n\n` +
          `Time: ${Number(result.processingTimeMs).toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  // Test blur detection
  const testBlurDetection = useCallback(async () => {
    setLoading(true);
    try {
      const result = await detectBlur(
        { type: 'url', value: currentImage },
        { threshold: 100, downsampleSize: 500 }
      );

      Alert.alert(
        'Blur Detection',
        `Is Blurry: ${result.isBlurry ? 'YES' : 'NO'}\n` +
          `Score: ${result.score.toFixed(2)}\n` +
          `Threshold: ${result.threshold}\n` +
          `(Higher score = sharper image)\n\n` +
          `Time: ${result.processingTimeMs.toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test video frame extraction (uses a sample video URL)
  const testVideoFrameExtraction = useCallback(async () => {
    setLoadingMessage('Extracting video frames...\nThis may take a moment');
    setLoading(true);
    try {
      // Use a sample video URL (Big Buck Bunny - public domain)
      const videoUrl =
        'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4';

      const result = await extractVideoFrames(
        { type: 'url', value: videoUrl },
        {
          count: 5, // Extract 5 evenly-spaced frames
          resize: { width: 224, height: 224 },
          outputFormat: 'base64',
          quality: 70, // JPEG quality 0-100
        }
      );

      // Show first frame as processed image if available
      if (result.frames.length > 0 && result.frames[0]!.base64) {
        const firstFrame = result.frames[0]!;
        setProcessedImageUri(`data:image/jpeg;base64,${firstFrame.base64}`);
      }

      Alert.alert(
        'Video Frame Extraction',
        `Extracted ${result.frameCount} frames\n` +
          `Video Duration: ${result.videoDuration.toFixed(2)}s\n` +
          `Video Size: ${result.videoWidth}x${result.videoHeight}\n` +
          `Frame Timestamps: ${result.frames
            .map((f) => f.timestamp.toFixed(2) + 's')
            .join(', ')}\n\n` +
          `Time: ${result.processingTimeMs.toFixed(2)}ms\n\n` +
          `First frame shown in preview`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', `Video extraction failed: ${errorMessage}`);
    } finally {
      setLoading(false);
      setLoadingMessage('Processing...');
    }
  }, []);

  // Test grid extraction
  const testGridExtraction = useCallback(async () => {
    setLoading(true);
    try {
      const result = await extractGrid(
        { type: 'url', value: currentImage },
        {
          columns: 3,
          rows: 3,
          overlap: 0,
          includePartial: false,
        },
        {
          colorFormat: 'rgb',
          normalization: { preset: 'scale' },
        }
      );

      Alert.alert(
        'Grid Extraction',
        `Extracted ${result.patchCount} patches (${result.columns}x${result.rows})\n` +
          `Patch size: ${result.patchWidth}x${result.patchHeight}\n` +
          `Original size: ${result.originalWidth}x${result.originalHeight}\n\n` +
          `Sample patch positions:\n` +
          result.patches
            .slice(0, 4)
            .map(
              (p) =>
                `  [${p.row},${p.column}]: (${p.x},${p.y}) ${p.width}x${p.height}`
            )
            .join('\n') +
          `\n\nTime: ${result.processingTimeMs.toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test random crop
  const testRandomCrop = useCallback(async () => {
    setLoading(true);
    try {
      const result = await randomCrop(
        { type: 'url', value: currentImage },
        {
          width: 64,
          height: 64,
          count: 5,
          seed: 42, // For reproducibility
        },
        {
          colorFormat: 'rgb',
          normalization: { preset: 'scale' },
        }
      );

      Alert.alert(
        'Random Crop',
        `Extracted ${result.cropCount} random crops\n` +
          `Crop size: ${result.crops[0]?.width ?? 64}x${
            result.crops[0]?.height ?? 64
          }\n` +
          `Original size: ${result.originalWidth}x${result.originalHeight}\n` +
          `Seed used: ${result.crops[0]?.seed ?? 'N/A'}\n\n` +
          `Crop positions:\n` +
          result.crops.map((c, i) => `  ${i + 1}: (${c.x},${c.y})`).join('\n') +
          `\n\nTime: ${(result.processingTimeMs ?? 0).toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test tensor validation
  const testTensorValidation = useCallback(async () => {
    setLoading(true);
    try {
      // First get some pixel data
      const pixelResult = await getPixelData({
        source: { type: 'url', value: currentImage },
        resize: { width: 224, height: 224, strategy: 'cover' },
        colorFormat: 'rgb',
        normalization: { preset: 'scale' },
        dataLayout: 'chw',
      });

      // Convert to number[] for validateTensor
      const dataArray = Array.from(pixelResult.data as ArrayLike<number>);

      // Validate against expected spec
      const validation = validateTensor(dataArray, [3, 224, 224], {
        shape: [3, 224, 224],
        dtype: 'float32',
        minValue: 0,
        maxValue: 1,
      });

      Alert.alert(
        'Tensor Validation',
        `Valid: ${validation.isValid ? 'YES ✓' : 'NO ✗'}\n` +
          `Actual shape: [${validation.actualShape.join(', ')}]\n\n` +
          `Statistics:\n` +
          `  Min: ${validation.actualMin.toFixed(4)}\n` +
          `  Max: ${validation.actualMax.toFixed(4)}\n` +
          `  Mean: ${validation.actualMean.toFixed(4)}\n\n` +
          (validation.issues.length > 0
            ? `Issues: ${validation.issues.join(', ')}`
            : 'No issues')
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [currentImage]);

  // Test batch assembly
  const testBatchAssembly = useCallback(async () => {
    setLoading(true);
    try {
      // Process multiple images
      const batchResults = await batchGetPixelData(
        SAMPLE_IMAGES.map((url) => ({
          source: { type: 'url' as const, value: url },
          resize: { width: 224, height: 224, strategy: 'cover' as const },
          colorFormat: 'rgb' as const,
          normalization: { preset: 'imagenet' as const },
          dataLayout: 'chw' as const,
        })),
        { concurrency: 3 }
      );

      // Filter successful results
      const successfulResults = batchResults.results.filter(
        (r): r is PixelDataResult => !('error' in r)
      );

      if (successfulResults.length === 0) {
        throw new Error('No images processed successfully');
      }

      // Assemble into batch
      const batch = assembleBatch(successfulResults, {
        layout: 'nchw',
      });

      Alert.alert(
        'Batch Assembly',
        `Assembled ${batch.batchSize} images into batch\n` +
          `Shape: [${batch.shape.join(', ')}]\n` +
          `Layout: NCHW\n` +
          `Total elements: ${batch.data.length.toLocaleString()}\n\n` +
          `Time: ${batchResults.totalTimeMs.toFixed(2)}ms`
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
  }, []);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <Text style={styles.title}>react-native-vision-utils</Text>
          <Text style={styles.subtitle}>Example App</Text>

          {/* Image Preview Section */}
          <TouchableOpacity onPress={cycleImage} activeOpacity={0.8}>
            <View style={styles.imageContainer}>
              <Image
                source={{ uri: processedImageUri || currentImage }}
                style={styles.previewImage}
                resizeMode="contain"
              />
              <Text style={styles.imageHint}>
                {processedImageUri
                  ? 'Processed Image'
                  : `Image ${currentImageIndex + 1}/${SAMPLE_IMAGES.length}`}
                {'\n'}Tap to cycle images
              </Text>
            </View>
          </TouchableOpacity>

          {processedImageUri && (
            <TouchableOpacity
              style={[styles.button, styles.resetButton]}
              onPress={() => setProcessedImageUri(null)}
            >
              <Text style={styles.buttonText}>Show Original</Text>
            </TouchableOpacity>
          )}

          {/* Basic Operations */}
          <Text style={styles.sectionTitle}>📊 Basic Operations</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[styles.button, loading && styles.buttonDisabled]}
              onPress={testBasicRGB}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Test Basic RGB</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.button, loading && styles.buttonDisabled]}
              onPress={testImageNet}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Test ImageNet (PyTorch)</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.button, loading && styles.buttonDisabled]}
              onPress={testTensorFlow}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Test TensorFlow</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.button, loading && styles.buttonDisabled]}
              onPress={testGrayscale}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Test Grayscale</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.button, loading && styles.buttonDisabled]}
              onPress={testResizeStrategies}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Test Resize Strategies</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.batchButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testBatchProcessing}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Test Batch Processing</Text>
            </TouchableOpacity>
          </View>

          {/* Image Analysis */}
          <Text style={styles.sectionTitle}>🔍 Image Analysis</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.analysisButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testStatistics}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Get Statistics</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.analysisButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testImageMetadata}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Get Metadata</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.analysisButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testValidation}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Validate Source</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.analysisButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testBlurDetection}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Detect Blur</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.analysisButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testVideoFrameExtraction}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Extract Video Frames</Text>
            </TouchableOpacity>
          </View>

          {/* Augmentations & Transforms */}
          <Text style={styles.sectionTitle}>🎨 Augmentations & Transforms</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.augmentButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testAugmentations}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Apply Augmentations</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.augmentButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testFiveCrop}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Five Crop Extract</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.augmentButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testTensorToImage}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Tensor → Image</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.augmentButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testGridExtraction}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Extract Grid</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.augmentButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testRandomCrop}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Random Crop</Text>
            </TouchableOpacity>
          </View>

          {/* Tensor Operations */}
          <Text style={styles.sectionTitle}>🧮 Tensor Operations</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.tensorButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testExtractChannel}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Extract Channel</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.tensorButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testTensorValidation}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Validate Tensor</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.tensorButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testBatchAssembly}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Assemble Batch</Text>
            </TouchableOpacity>
          </View>

          {/* Quantization */}
          <Text style={styles.sectionTitle}>🎯 Quantization (TFLite)</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.quantizeButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testQuantization}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Per-Tensor (uint8)</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.quantizeButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testPerChannelQuantization}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Per-Channel (int8)</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.quantizeButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testDequantization}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Dequantize Roundtrip</Text>
            </TouchableOpacity>
          </View>

          {/* Label Database */}
          <Text style={styles.sectionTitle}>🏷️ Label Database</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.labelButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testGetLabel}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Get Single Label</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.labelButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testTopLabels}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Top Predictions</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.labelButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testAvailableDatasets}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Available Datasets</Text>
            </TouchableOpacity>
          </View>

          {/* Camera Frame Processing */}
          <Text style={styles.sectionTitle}>📹 Camera Frame Utils</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.cameraButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testCameraFrameProcessing}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Process Camera Frame</Text>
            </TouchableOpacity>
          </View>

          {/* Bounding Box Utilities */}
          <Text style={styles.sectionTitle}>📦 Bounding Box Utilities</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.boxButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testBoxFormatConversion}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Format Conversion</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.boxButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testBoxScaling}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Scale Boxes</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.boxButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testClipBoxes}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Clip Boxes</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.boxButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testIoUCalculation}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Calculate IoU</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.boxButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testNMS}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Non-Max Suppression</Text>
            </TouchableOpacity>
          </View>

          {/* Letterbox Padding */}
          <Text style={styles.sectionTitle}>🖼️ Letterbox Padding</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.letterboxButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testLetterbox}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Apply Letterbox</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.letterboxButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testReverseLetterbox}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Reverse Letterbox</Text>
            </TouchableOpacity>
          </View>

          {/* Drawing & Visualization */}
          <Text style={styles.sectionTitle}>🎨 Drawing & Visualization</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.drawButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testDrawBoxes}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Draw Boxes</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.drawButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testDrawKeypoints}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Draw Keypoints</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.drawButton,
                loading && styles.buttonDisabled,
              ]}
              onPress={testHeatmapOverlay}
              disabled={loading}
            >
              <Text style={styles.buttonText}>Heatmap Overlay</Text>
            </TouchableOpacity>
          </View>

          {/* Cache Management */}
          <Text style={styles.sectionTitle}>💾 Cache Management</Text>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[styles.button, styles.cacheButton]}
              onPress={testCacheStats}
            >
              <Text style={styles.buttonText}>Cache Statistics</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.button, styles.clearButton]}
              onPress={testClearCache}
            >
              <Text style={styles.buttonText}>Clear Cache</Text>
            </TouchableOpacity>

            {results.length > 0 && (
              <TouchableOpacity
                style={[styles.button, styles.clearButton]}
                onPress={clearResults}
              >
                <Text style={styles.buttonText}>Clear Results</Text>
              </TouchableOpacity>
            )}
          </View>

          {results.length > 0 && (
            <View style={styles.resultsContainer}>
              <Text style={styles.resultsTitle}>Results:</Text>
              {results.map((item, index) => (
                <View key={index} style={styles.resultItem}>
                  <Text style={styles.resultLabel}>{item.label}</Text>
                  {item.result ? (
                    <Text style={styles.resultValue}>
                      {item.result.width}x{item.result.height}x
                      {item.result.channels} ({item.result.dataLayout}){'\n'}
                      Time: {item.result.processingTimeMs.toFixed(2)}ms
                      {'\n'}Data length: {item.result.data.length}
                    </Text>
                  ) : (
                    <Text style={styles.resultError}>Error: {item.error}</Text>
                  )}
                </View>
              ))}
            </View>
          )}
        </ScrollView>

        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#FFFFFF" />
            <Text style={styles.loadingText}>{loadingMessage}</Text>
          </View>
        )}
      </SafeAreaView>
    </SafeAreaProvider>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContent: {
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
    color: '#666',
    marginBottom: 20,
  },
  imageContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 10,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  previewImage: {
    width: IMAGE_SIZE - 20,
    height: IMAGE_SIZE - 20,
    borderRadius: 8,
    backgroundColor: '#e0e0e0',
  },
  imageHint: {
    textAlign: 'center',
    color: '#888',
    fontSize: 12,
    marginTop: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginTop: 20,
    marginBottom: 12,
  },
  loadingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  loadingText: {
    marginTop: 16,
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
    textAlign: 'center',
  },
  buttonContainer: {
    gap: 12,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 16,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  batchButton: {
    backgroundColor: '#5856D6',
  },
  analysisButton: {
    backgroundColor: '#34C759',
  },
  augmentButton: {
    backgroundColor: '#FF9500',
  },
  tensorButton: {
    backgroundColor: '#AF52DE',
  },
  quantizeButton: {
    backgroundColor: '#FF2D55',
  },
  labelButton: {
    backgroundColor: '#30D158',
  },
  cameraButton: {
    backgroundColor: '#64D2FF',
  },
  boxButton: {
    backgroundColor: '#FF6B35',
  },
  letterboxButton: {
    backgroundColor: '#7B68EE',
  },
  drawButton: {
    backgroundColor: '#20B2AA',
  },
  cacheButton: {
    backgroundColor: '#5AC8FA',
  },
  clearButton: {
    backgroundColor: '#FF3B30',
  },
  resetButton: {
    backgroundColor: '#8E8E93',
    marginBottom: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  resultsContainer: {
    marginTop: 24,
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 16,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#333',
  },
  resultItem: {
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    paddingVertical: 12,
  },
  resultLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#007AFF',
    marginBottom: 4,
  },
  resultValue: {
    fontSize: 12,
    color: '#666',
    fontFamily: 'monospace',
  },
  resultError: {
    fontSize: 12,
    color: '#FF3B30',
  },
});

export default App;
