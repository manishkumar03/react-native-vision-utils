import React, { useState, useCallback } from 'react';
import {
  SafeAreaView,
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
  type PixelDataResult,
  type ColorFormat,
  type DataLayout,
  type NormalizationPreset,
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
      Alert.alert(
        'Image Validation',
        `Valid: ${validation.isValid}\nErrors: ${
          validation.errors.length > 0 ? validation.errors.join(', ') : 'None'
        }\nSize: ${validation.metadata.width}x${validation.metadata.height}`
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
          brightness: 1.2,
          contrast: 1.1,
          saturation: 1.3,
        }
      );
      if (result.base64) {
        setProcessedImageUri(`data:image/png;base64,${result.base64}`);
      }
      Alert.alert(
        'Augmentations Applied',
        `Processed in ${result.processingTimeMs.toFixed(2)}ms\nOutput: ${
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
      Alert.alert(
        'Five Crop Extract',
        `Extracted ${
          result.crops.length
        } crops\nTotal time: ${result.totalTimeMs.toFixed(2)}ms`
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

  const clearResults = useCallback(() => {
    setResults([]);
  }, []);

  return (
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

        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.loadingText}>Processing...</Text>
          </View>
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
                    {item.result.channels} ({item.result.dataLayout}){'\n'}Time:{' '}
                    {item.result.processingTimeMs.toFixed(2)}ms
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
    </SafeAreaView>
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
    alignItems: 'center',
    marginVertical: 20,
  },
  loadingText: {
    marginTop: 10,
    color: '#666',
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
