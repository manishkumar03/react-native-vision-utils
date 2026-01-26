# react-native-vision-utils

A high-performance React Native library for image preprocessing optimized for ML/AI inference pipelines. Provides comprehensive tools for pixel data extraction, tensor manipulation, image augmentation, and model-specific preprocessing.

## Features

- 🚀 **High Performance**: Native implementations in Swift (iOS) and Kotlin (Android)
- 🎨 **Multiple Color Formats**: RGB, RGBA, BGR, BGRA, Grayscale, HSV, HSL, LAB, YUV, YCbCr
- 📐 **Flexible Resizing**: Cover, contain, stretch, and letterbox strategies
- 🔢 **ML-Ready Normalization**: ImageNet, TensorFlow, custom presets
- 📊 **Multiple Data Layouts**: HWC, CHW, NHWC, NCHW (PyTorch/TensorFlow compatible)
- 📦 **Batch Processing**: Process multiple images with concurrency control
- 🖼️ **Multiple Sources**: URL, file, base64, assets, photo library
- 🤖 **Model Presets**: Pre-configured settings for YOLO, MobileNet, EfficientNet, ResNet, ViT, CLIP, SAM, DINO, DETR
- 🔄 **Image Augmentation**: Rotation, flip, brightness, contrast, saturation, blur
- 📈 **Image Analysis**: Statistics, metadata, validation
- 🧮 **Tensor Operations**: Channel extraction, patch extraction, permutation, batch concatenation
- 🔙 **Tensor to Image**: Convert processed tensors back to images

## Installation

```sh
npm install react-native-vision-utils
```

For iOS, run:

```sh
cd ios && pod install
```

## Quick Start

### Basic Usage

```typescript
import { getPixelData } from 'react-native-vision-utils';

const result = await getPixelData({
  source: { type: 'url', value: 'https://example.com/image.jpg' },
});

console.log(result.data); // Float array of pixel values
console.log(result.width); // Image width
console.log(result.height); // Image height
console.log(result.shape); // [height, width, channels]
```

### Using Model Presets

```typescript
import { getPixelData, MODEL_PRESETS } from 'react-native-vision-utils';

// Use pre-configured YOLO settings
const result = await getPixelData({
  source: { type: 'file', value: '/path/to/image.jpg' },
  ...MODEL_PRESETS.yolov8,
});
// Automatically configured: 640x640, letterbox resize, RGB, scale normalization, NCHW layout

// Or MobileNet
const mobileNetResult = await getPixelData({
  source: { type: 'file', value: '/path/to/image.jpg' },
  ...MODEL_PRESETS.mobilenet,
});
// Configured: 224x224, cover resize, RGB, ImageNet normalization, NCHW layout
```

### Available Model Presets

| Preset            | Size      | Resize    | Normalization | Layout |
| ----------------- | --------- | --------- | ------------- | ------ |
| `yolo` / `yolov8` | 640×640   | letterbox | scale         | NCHW   |
| `mobilenet`       | 224×224   | cover     | ImageNet      | NCHW   |
| `efficientnet_b0` | 224×224   | cover     | ImageNet      | NCHW   |
| `efficientnet_b7` | 600×600   | cover     | ImageNet      | NCHW   |
| `resnet`          | 224×224   | cover     | ImageNet      | NCHW   |
| `vit`             | 224×224   | cover     | ImageNet      | NCHW   |
| `vit_large`       | 384×384   | cover     | ImageNet      | NCHW   |
| `clip`            | 224×224   | cover     | CLIP-specific | NCHW   |
| `sam`             | 1024×1024 | letterbox | scale         | NCHW   |
| `dino`            | 224×224   | cover     | ImageNet      | NCHW   |
| `detr`            | 800×800   | contain   | ImageNet      | NCHW   |

## API Reference

### Core Functions

#### `getPixelData(options)`

Extract pixel data from a single image.

```typescript
const result = await getPixelData({
  source: { type: 'file', value: '/path/to/image.jpg' },
  resize: { width: 224, height: 224, strategy: 'cover' },
  colorFormat: 'rgb',
  normalization: { preset: 'imagenet' },
  dataLayout: 'nchw',
  outputFormat: 'float32Array',
});
```

##### Options

| Property        | Type            | Default               | Description                   |
| --------------- | --------------- | --------------------- | ----------------------------- |
| `source`        | `ImageSource`   | _required_            | Image source specification    |
| `colorFormat`   | `ColorFormat`   | `'rgb'`               | Output color format           |
| `resize`        | `ResizeOptions` | -                     | Resize options                |
| `roi`           | `Roi`           | -                     | Region of interest to extract |
| `normalization` | `Normalization` | `{ preset: 'scale' }` | Normalization settings        |
| `dataLayout`    | `DataLayout`    | `'hwc'`               | Data layout format            |
| `outputFormat`  | `OutputFormat`  | `'array'`             | Output format                 |

##### Result

```typescript
interface PixelDataResult {
  data: number[] | Float32Array | Uint8Array;
  width: number;
  height: number;
  channels: number;
  dataLayout: DataLayout;
  shape: number[];
  processingTimeMs: number;
}
```

#### `batchGetPixelData(optionsArray, batchOptions)`

Process multiple images with concurrency control.

```typescript
const results = await batchGetPixelData(
  [
    {
      source: { type: 'url', value: 'https://example.com/1.jpg' },
      ...MODEL_PRESETS.mobilenet,
    },
    {
      source: { type: 'url', value: 'https://example.com/2.jpg' },
      ...MODEL_PRESETS.mobilenet,
    },
    {
      source: { type: 'url', value: 'https://example.com/3.jpg' },
      ...MODEL_PRESETS.mobilenet,
    },
  ],
  { concurrency: 4 }
);

console.log(results.totalTimeMs);
results.results.forEach((result, index) => {
  if ('error' in result) {
    console.log(`Image ${index} failed: ${result.message}`);
  } else {
    console.log(`Image ${index}: ${result.width}x${result.height}`);
  }
});
```

### Image Analysis

#### `getImageStatistics(source)`

Calculate image statistics for analysis and preprocessing decisions.

```typescript
import { getImageStatistics } from 'react-native-vision-utils';

const stats = await getImageStatistics({
  type: 'file',
  value: '/path/to/image.jpg',
});

console.log(stats.mean); // [r, g, b] mean values (0-1)
console.log(stats.std); // [r, g, b] standard deviations
console.log(stats.min); // [r, g, b] minimum values
console.log(stats.max); // [r, g, b] maximum values
console.log(stats.histogram); // { r: number[], g: number[], b: number[] }
```

#### `getImageMetadata(source)`

Get image metadata without loading full pixel data.

```typescript
import { getImageMetadata } from 'react-native-vision-utils';

const metadata = await getImageMetadata({
  type: 'file',
  value: '/path/to/image.jpg',
});

console.log(metadata.width); // Image width
console.log(metadata.height); // Image height
console.log(metadata.channels); // Number of channels
console.log(metadata.colorSpace); // Color space (sRGB, etc.)
console.log(metadata.hasAlpha); // Has alpha channel
console.log(metadata.aspectRatio); // Width / height ratio
```

#### `validateImage(source, options)`

Validate an image against specified criteria.

```typescript
import { validateImage } from 'react-native-vision-utils';

const validation = await validateImage(
  { type: 'file', value: '/path/to/image.jpg' },
  {
    minWidth: 224,
    minHeight: 224,
    maxWidth: 4096,
    maxHeight: 4096,
    requiredAspectRatio: 1.0,
    aspectRatioTolerance: 0.1,
  }
);

if (validation.isValid) {
  console.log('Image meets all requirements');
} else {
  console.log('Issues:', validation.issues);
}
```

### Image Augmentation

#### `applyAugmentations(source, augmentations)`

Apply image augmentations for data augmentation pipelines.

```typescript
import { applyAugmentations } from 'react-native-vision-utils';

const augmented = await applyAugmentations(
  { type: 'file', value: '/path/to/image.jpg' },
  {
    rotation: 15, // Degrees
    flipHorizontal: true,
    flipVertical: false,
    brightness: 1.2, // 1.0 = no change
    contrast: 1.1, // 1.0 = no change
    saturation: 0.9, // 1.0 = no change
    blur: 2.0, // Blur radius
  }
);

console.log(augmented.base64); // Base64 encoded result
console.log(augmented.width);
console.log(augmented.height);
```

### Multi-Crop Operations

#### `fiveCrop(source, options, pixelOptions)`

Extract 5 crops: 4 corners + center. Useful for test-time augmentation.

```typescript
import { fiveCrop, MODEL_PRESETS } from 'react-native-vision-utils';

const crops = await fiveCrop(
  { type: 'file', value: '/path/to/image.jpg' },
  { width: 224, height: 224 },
  MODEL_PRESETS.mobilenet
);

console.log(crops.cropCount); // 5
crops.results.forEach((result, i) => {
  console.log(`Crop ${i}: ${result.width}x${result.height}`);
});
```

#### `tenCrop(source, options, pixelOptions)`

Extract 10 crops: 5 crops + their horizontal flips.

```typescript
import { tenCrop, MODEL_PRESETS } from 'react-native-vision-utils';

const crops = await tenCrop(
  { type: 'file', value: '/path/to/image.jpg' },
  { width: 224, height: 224 },
  MODEL_PRESETS.mobilenet
);

console.log(crops.cropCount); // 10
```

### Tensor Operations

#### `extractChannel(data, width, height, channels, channelIndex, dataLayout)`

Extract a single channel from pixel data.

```typescript
import { getPixelData, extractChannel } from 'react-native-vision-utils';

const result = await getPixelData({
  source: { type: 'file', value: '/path/to/image.jpg' },
  colorFormat: 'rgb',
});

// Extract red channel
const redChannel = await extractChannel(
  result.data,
  result.width,
  result.height,
  result.channels,
  0, // channel index (0=R, 1=G, 2=B)
  result.dataLayout
);
```

#### `extractPatch(data, width, height, channels, patchOptions, dataLayout)`

Extract a rectangular patch from pixel data.

```typescript
import { getPixelData, extractPatch } from 'react-native-vision-utils';

const result = await getPixelData({
  source: { type: 'file', value: '/path/to/image.jpg' },
});

const patch = await extractPatch(
  result.data,
  result.width,
  result.height,
  result.channels,
  { x: 100, y: 100, width: 64, height: 64 },
  result.dataLayout
);
```

#### `concatenateToBatch(results)`

Combine multiple results into a batch tensor.

```typescript
import {
  batchGetPixelData,
  concatenateToBatch,
  MODEL_PRESETS,
} from 'react-native-vision-utils';

const batch = await batchGetPixelData(
  [
    {
      source: { type: 'file', value: '/path/to/1.jpg' },
      ...MODEL_PRESETS.mobilenet,
    },
    {
      source: { type: 'file', value: '/path/to/2.jpg' },
      ...MODEL_PRESETS.mobilenet,
    },
  ],
  { concurrency: 2 }
);

const batchTensor = await concatenateToBatch(batch.results);
console.log(batchTensor.shape); // [2, 3, 224, 224]
console.log(batchTensor.batchSize); // 2
```

#### `permute(data, shape, order)`

Transpose/permute tensor dimensions.

```typescript
import { getPixelData, permute } from 'react-native-vision-utils';

const result = await getPixelData({
  source: { type: 'file', value: '/path/to/image.jpg' },
  dataLayout: 'hwc', // [H, W, C]
});

// Convert HWC to CHW
const permuted = await permute(
  result.data,
  result.shape,
  [2, 0, 1] // new order: C, H, W
);
console.log(permuted.shape); // [channels, height, width]
```

### Tensor to Image

#### `tensorToImage(data, width, height, options)`

Convert tensor data back to an image.

```typescript
import {
  getPixelData,
  tensorToImage,
  MODEL_PRESETS,
} from 'react-native-vision-utils';

// Process image
const result = await getPixelData({
  source: { type: 'file', value: '/path/to/image.jpg' },
  ...MODEL_PRESETS.mobilenet,
});

// Convert back to image (with denormalization)
const image = await tensorToImage(result.data, result.width, result.height, {
  channels: 3,
  dataLayout: 'chw',
  format: 'png',
  denormalize: true,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
});

console.log(image.base64); // Base64 encoded PNG
```

### Cache Management

#### `clearCache()`

Clear the internal image cache.

```typescript
import { clearCache } from 'react-native-vision-utils';

await clearCache();
```

#### `getCacheStats()`

Get cache statistics.

```typescript
import { getCacheStats } from 'react-native-vision-utils';

const stats = await getCacheStats();
console.log(stats.hitCount);
console.log(stats.missCount);
console.log(stats.size);
console.log(stats.maxSize);
```

## Type Reference

### Image Source Types

```typescript
type ImageSourceType = 'url' | 'file' | 'base64' | 'asset' | 'photoLibrary';

interface ImageSource {
  type: ImageSourceType;
  value: string;
}

// Examples:
{ type: 'url', value: 'https://example.com/image.jpg' }
{ type: 'file', value: '/path/to/image.jpg' }
{ type: 'base64', value: 'data:image/png;base64,...' }
{ type: 'asset', value: 'image_name' }
{ type: 'photoLibrary', value: 'identifier' }
```

### Color Formats

```typescript
type ColorFormat =
  | 'rgb' // 3 channels
  | 'rgba' // 4 channels
  | 'bgr' // 3 channels (OpenCV style)
  | 'bgra' // 4 channels
  | 'grayscale' // 1 channel
  | 'hsv' // 3 channels (Hue, Saturation, Value)
  | 'hsl' // 3 channels (Hue, Saturation, Lightness)
  | 'lab' // 3 channels (CIE LAB)
  | 'yuv' // 3 channels
  | 'ycbcr'; // 3 channels
```

### Resize Strategies

```typescript
type ResizeStrategy =
  | 'cover' // Fill target, crop excess (default)
  | 'contain' // Fit within target, black letterbox
  | 'stretch' // Stretch to fill (may distort)
  | 'letterbox'; // Fit within target, gray padding (YOLO-style)

interface ResizeOptions {
  width: number;
  height: number;
  strategy?: ResizeStrategy;
}
```

### Normalization

```typescript
type NormalizationPreset =
  | 'scale' // [0, 1] range
  | 'imagenet' // ImageNet mean/std
  | 'tensorflow' // [-1, 1] range
  | 'raw' // No normalization (0-255)
  | 'custom'; // Custom mean/std

interface Normalization {
  preset?: NormalizationPreset;
  mean?: number[]; // Per-channel mean
  std?: number[]; // Per-channel std
}
```

### Data Layouts

```typescript
type DataLayout =
  | 'hwc' // Height × Width × Channels (default)
  | 'chw' // Channels × Height × Width (PyTorch)
  | 'nhwc' // Batch × Height × Width × Channels (TensorFlow)
  | 'nchw'; // Batch × Channels × Height × Width (PyTorch batched)
```

### Utility Functions

```typescript
// Get channel count for a color format
getChannelCount('rgb'); // 3
getChannelCount('rgba'); // 4
getChannelCount('grayscale'); // 1

// Check if error is a VisionUtilsError
isVisionUtilsError(error); // boolean
```

## Error Handling

All functions throw `VisionUtilsError` on failure:

```typescript
import {
  getPixelData,
  isVisionUtilsError,
  VisionUtilsErrorCode,
} from 'react-native-vision-utils';

try {
  const result = await getPixelData({
    source: { type: 'file', value: '/nonexistent.jpg' },
  });
} catch (error) {
  if (isVisionUtilsError(error)) {
    console.log(error.code); // 'LOAD_FAILED'
    console.log(error.message); // Human-readable message
  }
}
```

### Error Codes

| Code                 | Description                |
| -------------------- | -------------------------- |
| `INVALID_SOURCE`     | Invalid image source       |
| `LOAD_FAILED`        | Failed to load image       |
| `INVALID_ROI`        | Invalid region of interest |
| `PROCESSING_FAILED`  | Processing error           |
| `INVALID_OPTIONS`    | Invalid options provided   |
| `INVALID_CHANNEL`    | Invalid channel index      |
| `INVALID_PATCH`      | Invalid patch dimensions   |
| `DIMENSION_MISMATCH` | Tensor dimension mismatch  |
| `EMPTY_BATCH`        | Empty batch provided       |
| `UNKNOWN`            | Unknown error              |

## Performance Tips

1. **Use appropriate resize strategies**: `letterbox` for YOLO, `cover` for classification models
2. **Batch processing**: Use `batchGetPixelData` with appropriate concurrency for multiple images
3. **Cache management**: Call `clearCache()` when memory is constrained
4. **Use model presets**: Pre-configured settings are optimized for each model
5. **Avoid unnecessary conversions**: Choose the right `dataLayout` upfront

## Contributing

- [Development workflow](CONTRIBUTING.md#development-workflow)
- [Sending a pull request](CONTRIBUTING.md#sending-a-pull-request)
- [Code of conduct](CODE_OF_CONDUCT.md)

## License

MIT

---

Made with [create-react-native-library](https://github.com/callstack/react-native-builder-bob)
