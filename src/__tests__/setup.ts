import { NativeModules } from 'react-native';

// Mock the native module
const mockGetPixelData = jest.fn();
const mockBatchGetPixelData = jest.fn();
const mockGetImageStatistics = jest.fn();
const mockGetImageMetadata = jest.fn();
const mockValidateImage = jest.fn();
const mockTensorToImage = jest.fn();
const mockFiveCrop = jest.fn();
const mockTenCrop = jest.fn();
const mockExtractChannel = jest.fn();
const mockExtractPatch = jest.fn();
const mockConcatenateToBatch = jest.fn();
const mockPermute = jest.fn();
const mockApplyAugmentations = jest.fn();
const mockClearCache = jest.fn();
const mockGetCacheStats = jest.fn();
const mockQuantize = jest.fn();
const mockDequantize = jest.fn();
const mockCalculateQuantizationParams = jest.fn();
const mockGetLabel = jest.fn();
const mockGetTopLabels = jest.fn();
const mockGetAllLabels = jest.fn();
const mockGetDatasetInfo = jest.fn();
const mockGetAvailableDatasets = jest.fn();
const mockProcessCameraFrame = jest.fn();
const mockConvertYUVToRGB = jest.fn();

NativeModules.VisionUtils = {
  getPixelData: mockGetPixelData,
  batchGetPixelData: mockBatchGetPixelData,
  getImageStatistics: mockGetImageStatistics,
  getImageMetadata: mockGetImageMetadata,
  validateImage: mockValidateImage,
  tensorToImage: mockTensorToImage,
  fiveCrop: mockFiveCrop,
  tenCrop: mockTenCrop,
  extractChannel: mockExtractChannel,
  extractPatch: mockExtractPatch,
  concatenateToBatch: mockConcatenateToBatch,
  permute: mockPermute,
  applyAugmentations: mockApplyAugmentations,
  clearCache: mockClearCache,
  getCacheStats: mockGetCacheStats,
  quantize: mockQuantize,
  dequantize: mockDequantize,
  calculateQuantizationParams: mockCalculateQuantizationParams,
  getLabel: mockGetLabel,
  getTopLabels: mockGetTopLabels,
  getAllLabels: mockGetAllLabels,
  getDatasetInfo: mockGetDatasetInfo,
  getAvailableDatasets: mockGetAvailableDatasets,
  processCameraFrame: mockProcessCameraFrame,
  convertYUVToRGB: mockConvertYUVToRGB,
};

export {
  mockGetPixelData,
  mockBatchGetPixelData,
  mockGetImageStatistics,
  mockGetImageMetadata,
  mockValidateImage,
  mockTensorToImage,
  mockFiveCrop,
  mockTenCrop,
  mockExtractChannel,
  mockExtractPatch,
  mockConcatenateToBatch,
  mockPermute,
  mockApplyAugmentations,
  mockClearCache,
  mockGetCacheStats,
  mockQuantize,
  mockDequantize,
  mockCalculateQuantizationParams,
  mockGetLabel,
  mockGetTopLabels,
  mockGetAllLabels,
  mockGetDatasetInfo,
  mockGetAvailableDatasets,
  mockProcessCameraFrame,
  mockConvertYUVToRGB,
};
