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
} from 'react-native';
import {
  getPixelData,
  batchGetPixelData,
  type PixelDataResult,
  type ColorFormat,
  type DataLayout,
  type NormalizationPreset,
} from 'react-native-vision-utils';

// Sample image URLs for testing (using stable public test images)
const SAMPLE_IMAGES = [
  'https://picsum.photos/400/400', // Random square image
  'https://picsum.photos/600/300', // Random wide image
  'https://picsum.photos/300/600', // Random tall image
];

interface ResultData {
  label: string;
  result: PixelDataResult | null;
  error: string | null;
}

const App: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ResultData[]>([]);

  // Test basic URL loading with RGB
  const testBasicRGB = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getPixelData({
        source: { type: 'url', value: SAMPLE_IMAGES[0]! },
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
  }, []);

  // Test ImageNet normalization (PyTorch style)
  const testImageNet = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getPixelData({
        source: { type: 'url', value: SAMPLE_IMAGES[0]! },
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
  }, []);

  // Test TensorFlow normalization
  const testTensorFlow = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getPixelData({
        source: { type: 'url', value: SAMPLE_IMAGES[0]! },
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
  }, []);

  // Test grayscale conversion
  const testGrayscale = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getPixelData({
        source: { type: 'url', value: SAMPLE_IMAGES[0]! },
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
  }, []);

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
      const strategies = ['cover', 'contain', 'stretch'] as const;

      for (const strategy of strategies) {
        const result = await getPixelData({
          source: { type: 'url', value: SAMPLE_IMAGES[1]! }, // Non-square image
          colorFormat: 'rgb',
          resize: { width: 224, height: 224, strategy },
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
        'All resize strategies tested successfully!'
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
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.title}>react-native-vision-utils</Text>
        <Text style={styles.subtitle}>Example App</Text>

        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.loadingText}>Processing...</Text>
          </View>
        )}

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
  clearButton: {
    backgroundColor: '#FF3B30',
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
