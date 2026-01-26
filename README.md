<p align="center">
  <h1 align="center">react-native-vision-utils</h1>
</p>

<p align="center">
  <strong>High-performance React Native library for image preprocessing optimized for ML/AI inference pipelines</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#features">Features</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/npm/v/react-native-vision-utils?style=flat-square" alt="npm version" />
  <img src="https://img.shields.io/npm/dm/react-native-vision-utils?style=flat-square" alt="downloads" />
  <img src="https://img.shields.io/github/license/user/react-native-vision-utils?style=flat-square" alt="license" />
</p>

---

Provides comprehensive tools for pixel data extraction, tensor manipulation, image augmentation, and model-specific preprocessing with native implementations in Swift (iOS) and Kotlin (Android).

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
  - [Core Functions](#-core-functions)
  - [Image Analysis](#-image-analysis)
  - [Image Augmentation](#-image-augmentation)
  - [Multi-Crop Operations](#️-multi-crop-operations)
  - [Tensor Operations](#-tensor-operations)
  - [Tensor to Image](#️-tensor-to-image)
  - [Label Database](#️-label-database)
  - [Camera Frame Utilities](#-camera-frame-utilities)
  - [Quantization](#-quantization-for-tflite-and-other-quantized-models)
  - [Bounding Box Utilities](#-bounding-box-utilities)
  - [Letterbox Utilities](#-letterbox-padding)
  - [Drawing & Visualization](#-drawing--visualization)
- [Types Reference](#-types-reference)
- [Error Handling](#-error-handling)
- [Platform Considerations](#-platform-considerations)
- [Performance Tips](#-performance-tips)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- 🚀 **High Performance**: Native implementations in Swift (iOS) and Kotlin (Android)
- 🔢 **Raw Pixel Data**: Extract pixel values as typed arrays (Float32, Int32, Uint8) ready for ML inference
- 🎨 **Multiple Color Formats**: RGB, RGBA, BGR, BGRA, Grayscale, HSV, HSL, LAB, YUV, YCbCr
- 📐 **Flexible Resizing**: Cover, contain, stretch, and letterbox strategies
- 🔢 **ML-Ready Normalization**: ImageNet, TensorFlow, custom presets
- 📊 **Multiple Data Layouts**: HWC, CHW, NHWC, NCHW (PyTorch/TensorFlow compatible)
- 📦 **Batch Processing**: Process multiple images with concurrency control
- 🖼️ **Multiple Sources**: URL, file, base64, assets, photo library
- 🤖 **Model Presets**: Pre-configured settings for YOLO, MobileNet, EfficientNet, ResNet, ViT, CLIP, SAM, DINO, DETR
- 🔄 **Image Augmentation**: Rotation, flip, brightness, contrast, saturation, blur
- 🎨 **Color Jitter**: Granular brightness/contrast/saturation/hue control with range support and seeded randomness
- ✂️ **Cutout/Random Erasing**: Mask random regions with constant/noise fill for robustness training
- 📈 **Image Analysis**: Statistics, metadata, validation, blur detection
- 🧮 **Tensor Operations**: Channel extraction, patch extraction, permutation, batch concatenation
- 🔙 **Tensor to Image**: Convert processed tensors back to images
- 🎯 **Native Quantization**: Float→Int8/Uint8/Int16 with per-tensor and per-channel support (TFLite compatible)
- 🏷️ **Label Database**: Built-in labels for COCO, ImageNet, VOC, CIFAR, Places365, ADE20K
- 📹 **Camera Frame Utils**: Direct YUV/NV12/BGRA→tensor conversion for vision-camera integration
- 📦 **Bounding Box Utilities**: Format conversion (xyxy/xywh/cxcywh), scaling, clipping, IoU, NMS
- 🖼️ **Letterbox Padding**: YOLO-style letterbox preprocessing with reverse coordinate transform
- 🎨 **Drawing/Visualization**: Draw boxes, keypoints, masks, and heatmaps for debugging
- 🎬 **Video Frame Extraction**: Extract frames from videos at timestamps, intervals, or evenly-spaced for temporal ML models
- 🔲 **Grid/Patch Extraction**: Extract image patches in grid patterns for sliding window inference
- 🎲 **Random Crop with Seed**: Reproducible random crops for data augmentation pipelines
- ✅ **Tensor Validation**: Validate tensor shapes, dtypes, and value ranges before inference
- 📦 **Batch Assembly**: Combine multiple images into NCHW/NHWC batch tensors

---

## 📦 Installation

```sh
npm install react-native-vision-utils
```

For iOS, run:

```sh
cd ios && pod install
```

---

## 🚀 Quick Start

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
| `mobilenet`       | 224×224   | cover     | ImageNet      | NHWC   |
| `mobilenet_v2`    | 224×224   | cover     | ImageNet      | NHWC   |
| `mobilenet_v3`    | 224×224   | cover     | ImageNet      | NHWC   |
| `efficientnet`    | 224×224   | cover     | ImageNet      | NHWC   |
| `resnet`          | 224×224   | cover     | ImageNet      | NCHW   |
| `resnet50`        | 224×224   | cover     | ImageNet      | NCHW   |
| `vit`             | 224×224   | cover     | ImageNet      | NCHW   |
| `clip`            | 224×224   | cover     | CLIP-specific | NCHW   |
| `sam`             | 1024×1024 | contain   | ImageNet      | NCHW   |
| `dino`            | 224×224   | cover     | ImageNet      | NCHW   |
| `detr`            | 800×800   | contain   | ImageNet      | NCHW   |

---

## 📖 API Reference

### 🔧 Core Functions

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
  colorFormat: ColorFormat;
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

---

### 🔍 Image Analysis

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

#### `detectBlur(source, options?)`

Detect if an image is blurry using Laplacian variance analysis.

```typescript
import { detectBlur } from 'react-native-vision-utils';

const result = await detectBlur(
  { type: 'file', value: '/path/to/photo.jpg' },
  {
    threshold: 100, // Higher = stricter blur detection
    downsampleSize: 500, // Downsample for faster processing
  }
);

console.log(result.isBlurry); // true if blurry
console.log(result.score); // Laplacian variance score
console.log(result.threshold); // Threshold used
console.log(result.processingTimeMs); // Processing time
```

**Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | number | 100 | Variance threshold - images with score below this are considered blurry |
| `downsampleSize` | number | 500 | Max dimension for downsampling (improves performance on large images) |

**Result:**

| Property | Type | Description |
|----------|------|-------------|
| `isBlurry` | boolean | Whether the image is considered blurry |
| `score` | number | Laplacian variance score (higher = sharper) |
| `threshold` | number | The threshold that was used |
| `processingTimeMs` | number | Processing time in milliseconds |

#### `extractVideoFrames(source, options?)`

Extract frames from video files for video analysis, action recognition, and temporal ML models.

```typescript
import { extractVideoFrames } from 'react-native-vision-utils';

// Extract 10 evenly-spaced frames
const result = await extractVideoFrames(
  { type: 'file', value: '/path/to/video.mp4' },
  {
    count: 10, // Extract 10 evenly-spaced frames
    resize: { width: 224, height: 224 },
    outputFormat: 'base64', // or 'pixelData' for ML-ready arrays
  }
);

console.log(result.frameCount); // Number of frames extracted
console.log(result.videoDuration); // Video duration in seconds
console.log(result.videoWidth); // Video width
console.log(result.videoHeight); // Video height
console.log(result.frames); // Array of extracted frames
```

**Extraction Modes:**

```typescript
// Mode 1: Specific timestamps
const result = await extractVideoFrames(source, {
  timestamps: [0.5, 1.0, 2.5, 5.0], // Extract at these exact times
});

// Mode 2: Regular intervals
const result = await extractVideoFrames(source, {
  interval: 1.0, // Extract every 1 second
  startTime: 5.0, // Start at 5 seconds
  endTime: 30.0, // End at 30 seconds
  maxFrames: 25, // Limit to 25 frames
});

// Mode 3: Count-based (evenly spaced)
const result = await extractVideoFrames(source, {
  count: 16, // Extract exactly 16 evenly-spaced frames
});
```

**ML-Ready Output:**

```typescript
// Get pixel arrays ready for inference
const result = await extractVideoFrames(
  { type: 'file', value: '/path/to/video.mp4' },
  {
    count: 16,
    resize: { width: 224, height: 224 },
    outputFormat: 'pixelData',
    colorFormat: 'rgb',
    normalization: { preset: 'imagenet' },
  }
);

// Each frame has normalized pixel data ready for models
result.frames.forEach((frame) => {
  const tensor = frame.data; // Float32 array
  // Use with your ML model
});
```

**Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timestamps` | number[] | - | Specific timestamps in seconds to extract frames |
| `interval` | number | - | Extract frames at regular intervals (seconds) |
| `count` | number | - | Number of evenly-spaced frames to extract |
| `startTime` | number | 0 | Start time in seconds (for interval/count modes) |
| `endTime` | number | duration | End time in seconds (for interval/count modes) |
| `maxFrames` | number | 100 | Maximum frames for interval mode |
| `resize` | object | - | Resize frames to { width, height } |
| `outputFormat` | string | 'base64' | 'base64' or 'pixelData' for ML arrays |
| `quality` | number | 90 | JPEG quality for base64 output (0-100) |
| `colorFormat` | string | 'rgb' | Color format for pixelData ('rgb', 'rgba', 'bgr', 'grayscale') |
| `normalization` | object | scale | Normalization preset for pixelData ('scale', 'imagenet', 'tensorflow') |

> **Note:** If no extraction mode is specified (no `timestamps`, `interval`, or `count`), a single frame at t=0 is extracted by default.

> **Platform Note:** The `asset` source type is only supported on iOS. Android supports `file` and `url` sources.

> **PixelData Note:** `colorFormat` and `normalization` are applied only when `outputFormat === 'pixelData'`.

**Result:**

| Property | Type | Description |
|----------|------|-------------|
| `frames` | ExtractedFrame[] | Array of extracted frames |
| `frameCount` | number | Number of frames extracted |
| `videoDuration` | number | Total video duration in seconds |
| `videoWidth` | number | Video width in pixels |
| `videoHeight` | number | Video height in pixels |
| `frameRate` | number | Video frame rate (fps) |
| `processingTimeMs` | number | Processing time in milliseconds |

**ExtractedFrame:**

| Property | Type | Description |
|----------|------|-------------|
| `timestamp` | number | Actual frame timestamp in seconds |
| `requestedTimestamp` | number | Originally requested timestamp |
| `width` | number | Frame width |
| `height` | number | Frame height |
| `data` | string \| number[] | Base64 string (when outputFormat='base64') or pixel array (when outputFormat='pixelData') |
| `channels` | number | Number of channels (only present when outputFormat='pixelData') |
| `error` | string | Error message if frame extraction failed (frame may still be present with partial data) |

---

### 🎨 Image Augmentation

#### `applyAugmentations(source, augmentations)`

Apply image augmentations for data augmentation pipelines.

```typescript
import { applyAugmentations } from 'react-native-vision-utils';

const augmented = await applyAugmentations(
  { type: 'file', value: '/path/to/image.jpg' },
  {
    rotation: 15, // Degrees
    horizontalFlip: true,
    verticalFlip: false,
    brightness: 1.2, // 1.0 = no change
    contrast: 1.1, // 1.0 = no change
    saturation: 0.9, // 1.0 = no change
    blur: { type: 'gaussian', radius: 2 }, // Blur options
  }
);

console.log(augmented.base64); // Base64 encoded result
console.log(augmented.width);
console.log(augmented.height);
```

#### `colorJitter(source, options)`

Apply color jitter augmentation with granular control over brightness, contrast, saturation, and hue. More flexible than `applyAugmentations` - supports ranges for each property with random sampling and optional seed for reproducibility.

```typescript
import { colorJitter } from 'react-native-vision-utils';

// Basic usage with symmetric ranges
const result = await colorJitter(
  { type: 'file', value: '/path/to/image.jpg' },
  {
    brightness: 0.2,     // Random in [-0.2, +0.2]
    contrast: 0.2,       // Random in [0.8, 1.2]
    saturation: 0.3,     // Random in [0.7, 1.3]
    hue: 0.1,            // Random in [-0.1, +0.1] (fraction of 360°)
  }
);

console.log(result.base64);              // Augmented image
console.log(result.appliedBrightness);   // Actual value applied
console.log(result.appliedContrast);
console.log(result.appliedSaturation);
console.log(result.appliedHue);

// Asymmetric ranges with seed for reproducibility
const reproducible = await colorJitter(
  { type: 'url', value: 'https://example.com/image.jpg' },
  {
    brightness: [-0.1, 0.3],  // More brightening than darkening
    contrast: [0.8, 1.5],     // Allow up to 50% more contrast
    seed: 42,                  // Same seed = same random values
  }
);
```

**Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `brightness` | number \| [min, max] | 0 | Additive brightness adjustment. Single value creates symmetric range [-v, +v] |
| `contrast` | number \| [min, max] | 1 | Contrast multiplier. Single value creates range [max(0, 1-v), 1+v] |
| `saturation` | number \| [min, max] | 1 | Saturation multiplier. Single value creates range [max(0, 1-v), 1+v] |
| `hue` | number \| [min, max] | 0 | Hue shift as fraction of color wheel (0-1 = 0-360°). Single value creates symmetric range [-v, +v] |
| `seed` | number | random | Seed for reproducible random sampling |

**Result:**

| Property | Type | Description |
|----------|------|-------------|
| `base64` | string | Augmented image as base64 PNG |
| `width` | number | Output image width |
| `height` | number | Output image height |
| `appliedBrightness` | number | Actual brightness value applied |
| `appliedContrast` | number | Actual contrast value applied |
| `appliedSaturation` | number | Actual saturation value applied |
| `appliedHue` | number | Actual hue shift value applied |
| `seed` | number | Seed used for random generation |
| `processingTimeMs` | number | Processing time in milliseconds |

#### `cutout(source, options)`

Apply cutout (random erasing) augmentation to mask random rectangular regions. Improves model robustness by forcing it to rely on diverse features.

```typescript
import { cutout } from 'react-native-vision-utils';

// Basic cutout with black fill
const result = await cutout(
  { type: 'file', value: '/path/to/image.jpg' },
  {
    numCutouts: 1,
    minSize: 0.02,      // 2% of image area min
    maxSize: 0.33,      // 33% of image area max
  }
);

console.log(result.base64);       // Augmented image
console.log(result.numCutouts);   // Number of cutouts applied
console.log(result.regions);      // Details of each cutout

// Multiple cutouts with noise fill
const noisy = await cutout(
  { type: 'url', value: 'https://example.com/image.jpg' },
  {
    numCutouts: 3,
    minSize: 0.01,
    maxSize: 0.1,
    fillMode: 'noise',
    seed: 42,  // Reproducible
  }
);

// Cutout with custom gray fill and probability
const probabilistic = await cutout(
  { type: 'base64', value: base64Data },
  {
    numCutouts: 2,
    fillMode: 'constant',
    fillValue: [128, 128, 128],  // Gray
    probability: 0.5,  // 50% chance of applying
  }
);
```

**Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numCutouts` | number | 1 | Number of cutout regions to apply |
| `minSize` | number | 0.02 | Minimum cutout size as fraction of image area (0-1) |
| `maxSize` | number | 0.33 | Maximum cutout size as fraction of image area (0-1) |
| `minAspect` | number | 0.3 | Minimum aspect ratio (width/height) |
| `maxAspect` | number | 3.3 | Maximum aspect ratio (width/height) |
| `fillMode` | string | 'constant' | Fill mode: 'constant', 'noise', or 'random' |
| `fillValue` | [R,G,B] | [0,0,0] | Fill color for 'constant' mode (0-255) |
| `probability` | number | 1.0 | Probability of applying cutout (0-1) |
| `seed` | number | random | Seed for reproducible cutouts |

**Result:**

| Property | Type | Description |
|----------|------|-------------|
| `base64` | string | Augmented image as base64 PNG |
| `width` | number | Output image width |
| `height` | number | Output image height |
| `applied` | boolean | Whether cutout was applied (based on probability) |
| `numCutouts` | number | Number of cutout regions applied |
| `regions` | array | Details of each cutout: {x, y, width, height, fill} |
| `seed` | number | Seed used for random generation |
| `processingTimeMs` | number | Processing time in milliseconds |

---

### ✂️ Multi-Crop Operations

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

#### `extractGrid(source, gridOptions, pixelOptions?)`

Extract image patches in a grid pattern. Useful for sliding window detection, image tiling, and patch-based models.

```typescript
import { extractGrid } from 'react-native-vision-utils';

// Extract 3x3 grid of patches
const result = await extractGrid(
  { type: 'file', value: '/path/to/image.jpg' },
  {
    columns: 3,      // Number of columns
    rows: 3,         // Number of rows
    overlap: 0,      // Overlap in pixels (optional)
    overlapPercent: 0.1, // Or overlap as percentage (optional)
    includePartial: false, // Include edge patches smaller than full size
  },
  {
    colorFormat: 'rgb',
    normalization: { preset: 'scale' },
  }
);

console.log(result.patchCount); // 9
console.log(result.patchWidth);  // Width of each patch
console.log(result.patchHeight); // Height of each patch
result.patches.forEach((patch) => {
  console.log(`Patch [${patch.row},${patch.column}] at (${patch.x},${patch.y}): ${patch.width}x${patch.height}`);
  console.log(patch.data); // Pixel data for this patch
});
```

**Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | number | required | Number of columns in the grid |
| `rows` | number | required | Number of rows in the grid |
| `overlap` | number | 0 | Overlap between patches in pixels |
| `overlapPercent` | number | - | Overlap as percentage (0-1), alternative to `overlap` |
| `includePartial` | boolean | false | Include edge patches that are smaller than full size |

#### `randomCrop(source, cropOptions, pixelOptions?)`

Extract random crops from an image with optional seed for reproducibility. Essential for data augmentation in training pipelines.

```typescript
import { randomCrop } from 'react-native-vision-utils';

// Extract 5 random 64x64 crops with a fixed seed
const result = await randomCrop(
  { type: 'file', value: '/path/to/image.jpg' },
  {
    width: 64,
    height: 64,
    count: 5,
    seed: 42, // For reproducibility (optional)
  },
  {
    colorFormat: 'rgb',
    normalization: { preset: 'imagenet' },
  }
);

console.log(result.cropCount); // 5
result.crops.forEach((crop, i) => {
  console.log(`Crop ${i}: position (${crop.x},${crop.y}), size ${crop.width}x${crop.height}`);
  console.log(`Seed: ${crop.seed}`); // Seed used for this specific crop
  console.log(crop.data); // Pixel data for this crop
});
```

**Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | number | required | Width of each crop |
| `height` | number | required | Height of each crop |
| `count` | number | 1 | Number of random crops to extract |
| `seed` | number | random | Seed for reproducible random positions |

#### `validateTensor(data, shape, spec?)`

Validate tensor data against a specification. Pure TypeScript function for checking tensor shapes, dtypes, and value ranges before inference.

```typescript
import { validateTensor } from 'react-native-vision-utils';

const tensorData = new Float32Array(1 * 3 * 224 * 224);
// ... fill with data

const validation = validateTensor(
  tensorData,
  [1, 3, 224, 224], // Actual shape of your data
  {
    shape: [1, 3, 224, 224], // Expected shape (-1 as wildcard)
    dtype: 'float32',
    minValue: 0,
    maxValue: 1,
  }
);

if (validation.isValid) {
  console.log('Tensor is valid for inference');
} else {
  console.log('Validation issues:', validation.issues);
}

// Also provides statistics
console.log(validation.actualMin);
console.log(validation.actualMax);
console.log(validation.actualMean);
console.log(validation.actualShape);
```

**TensorSpec Options:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | number[] | Expected shape (use -1 as wildcard for any dimension) |
| `dtype` | string | Expected dtype ('float32', 'int32', 'uint8', etc.) |
| `minValue` | number | Minimum allowed value (optional) |
| `maxValue` | number | Maximum allowed value (optional) |
| `channels` | number | Expected number of channels (optional) |

**Result:**

| Property | Type | Description |
|----------|------|-------------|
| `isValid` | boolean | Whether tensor passes all validations |
| `issues` | string[] | List of validation issues found |
| `actualShape` | number[] | Actual shape of the tensor |
| `actualMin` | number | Minimum value in tensor |
| `actualMax` | number | Maximum value in tensor |
| `actualMean` | number | Mean value of tensor |

#### `assembleBatch(pixelResults, options?)`

Assemble multiple PixelDataResults into a single batch tensor. Pure TypeScript function for preparing batched inference.

```typescript
import { getPixelData, assembleBatch, MODEL_PRESETS } from 'react-native-vision-utils';

// Process multiple images
const images = ['/path/to/1.jpg', '/path/to/2.jpg', '/path/to/3.jpg'];
const results = await Promise.all(
  images.map((path) =>
    getPixelData({
      source: { type: 'file', value: path },
      ...MODEL_PRESETS.mobilenet,
    })
  )
);

// Assemble into batch
const batch = assembleBatch(results, {
  layout: 'nchw', // or 'nhwc'
  padToSize: 4,   // Pad batch to size 4 (optional)
});

console.log(batch.shape);     // [3, 3, 224, 224] for NCHW
console.log(batch.batchSize); // 3 (or 4 if padded)
console.log(batch.data);      // Combined Float32Array
```

**Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout` | 'nchw' \| 'nhwc' | 'nchw' | Output batch layout |
| `padToSize` | number | - | Pad batch to this size with zeros (optional) |

---

### 🧮 Tensor Operations

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

---

### 🖼️ Tensor to Image

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

---

### 💾 Cache Management

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

---

### 🏷️ Label Database

Built-in label databases for common ML classification and detection models. No external files needed.

#### `getLabel(index, options)`

Get a label by its class index.

```typescript
import { getLabel } from 'react-native-vision-utils';

// Simple lookup
const label = await getLabel(0); // Returns 'person' (COCO default)

// With metadata
const labelInfo = await getLabel(0, {
  dataset: 'coco',
  includeMetadata: true,
});
console.log(labelInfo.name); // 'person'
console.log(labelInfo.displayName); // 'Person'
console.log(labelInfo.supercategory); // 'person'
```

#### `getTopLabels(scores, options)`

Get top-K labels from prediction scores (e.g., from softmax output).

```typescript
import { getTopLabels } from 'react-native-vision-utils';

// After running inference, get your output scores
const scores = modelOutput; // Array of 80 confidence scores for COCO

const topLabels = await getTopLabels(scores, {
  dataset: 'coco',
  k: 5,
  minConfidence: 0.1,
  includeMetadata: true,
});

topLabels.forEach((result) => {
  console.log(`${result.label}: ${(result.confidence * 100).toFixed(1)}%`);
});
// Output:
// person: 92.3%
// car: 45.2%
// dog: 23.1%
```

#### `getAllLabels(dataset)`

Get all labels for a dataset.

```typescript
import { getAllLabels } from 'react-native-vision-utils';

const cocoLabels = await getAllLabels('coco');
console.log(cocoLabels.length); // 80
console.log(cocoLabels[0]); // 'person'
```

#### `getDatasetInfo(dataset)`

Get information about a dataset.

```typescript
import { getDatasetInfo } from 'react-native-vision-utils';

const info = await getDatasetInfo('imagenet');
console.log(info.name); // 'imagenet'
console.log(info.numClasses); // 1000
console.log(info.description); // 'ImageNet ILSVRC 2012 classification labels'
```

#### `getAvailableDatasets()`

List all available label datasets.

```typescript
import { getAvailableDatasets } from 'react-native-vision-utils';

const datasets = await getAvailableDatasets();
// ['coco', 'coco91', 'imagenet', 'voc', 'cifar10', 'cifar100', 'places365', 'ade20k']
```

##### Available Datasets

| Dataset       | Classes | Description                          |
| ------------- | ------- | ------------------------------------ |
| `coco`        | 80      | COCO 2017 object detection labels    |
| `coco91`      | 91      | COCO original labels with background |
| `imagenet`    | 1000    | ImageNet ILSVRC 2012 classification  |
| `imagenet21k` | 21841   | ImageNet-21K full classification     |
| `voc`         | 21      | PASCAL VOC with background           |
| `cifar10`     | 10      | CIFAR-10 classification              |
| `cifar100`    | 100     | CIFAR-100 classification             |
| `places365`   | 365     | Places365 scene recognition          |
| `ade20k`      | 150     | ADE20K semantic segmentation         |

---

### 📹 Camera Frame Utilities

High-performance utilities for processing camera frames directly, optimized for integration with react-native-vision-camera.

#### `processCameraFrame(source, options)`

Convert camera frame buffer to ML-ready tensor.

```typescript
import { processCameraFrame } from 'react-native-vision-utils';

// In a vision-camera frame processor
const result = await processCameraFrame(
  {
    width: 1920,
    height: 1080,
    pixelFormat: 'yuv420', // Camera format
    bytesPerRow: 1920,
    dataBase64: frameDataBase64, // Base64-encoded frame data
    orientation: 'right', // Device orientation
    timestamp: Date.now(),
  },
  {
    outputWidth: 224, // Resize for model
    outputHeight: 224,
    normalize: true,
    outputFormat: 'rgb',
    mean: [0.485, 0.456, 0.406], // ImageNet normalization
    std: [0.229, 0.224, 0.225],
  }
);

console.log(result.tensor); // Normalized float array
console.log(result.shape); // [224, 224, 3]
console.log(result.processingTimeMs); // Processing time
```

#### `convertYUVToRGB(options)`

Direct YUV to RGB conversion for camera frames.

```typescript
import { convertYUVToRGB } from 'react-native-vision-utils';

const rgbData = await convertYUVToRGB({
  width: 640,
  height: 480,
  pixelFormat: 'yuv420',
  yPlaneBase64: yPlaneData,
  uPlaneBase64: uPlaneData,
  vPlaneBase64: vPlaneData,
  outputFormat: 'rgb', // or 'base64'
});

console.log(rgbData.data); // RGB pixel values
console.log(rgbData.channels); // 3
```

##### Supported Pixel Formats

| Format   | Description                     |
| -------- | ------------------------------- |
| `yuv420` | YUV 4:2:0 planar                |
| `yuv422` | YUV 4:2:2 planar                |
| `nv12`   | NV12 (Y plane + interleaved UV) |
| `nv21`   | NV21 (Y plane + interleaved VU) |
| `bgra`   | BGRA 8-bit                      |
| `rgba`   | RGBA 8-bit                      |
| `rgb`    | RGB 8-bit                       |

##### Frame Orientations

| Orientation | Description           |
| ----------- | --------------------- |
| `up`        | No rotation (default) |
| `down`      | 180° rotation         |
| `left`      | 90° counter-clockwise |
| `right`     | 90° clockwise         |

---

### 📊 Quantization (for TFLite and Other Quantized Models)

Native high-performance quantization for deploying to quantized ML models like TFLite int8.

#### `quantize(data, options)`

Quantize float data to int8/uint8/int16 format.

```typescript
import {
  getPixelData,
  quantize,
  MODEL_PRESETS,
} from 'react-native-vision-utils';

// Get float pixel data
const result = await getPixelData({
  source: { type: 'file', value: '/path/to/image.jpg' },
  ...MODEL_PRESETS.mobilenet,
});

// Per-tensor quantization (single scale/zeroPoint)
const quantized = await quantize(result.data, {
  mode: 'per-tensor',
  dtype: 'uint8',
  scale: 0.0078125, // From TFLite model
  zeroPoint: 128, // From TFLite model
});

console.log(quantized.data); // Uint8Array
console.log(quantized.scale); // 0.0078125
console.log(quantized.zeroPoint); // 128
console.log(quantized.dtype); // 'uint8'
console.log(quantized.mode); // 'per-tensor'
```

##### Per-Channel Quantization

For models with per-channel quantization (common in TFLite):

```typescript
// Per-channel quantization (different scale/zeroPoint per channel)
const quantized = await quantize(result.data, {
  mode: 'per-channel',
  dtype: 'int8',
  scale: [0.0123, 0.0156, 0.0189], // Per-channel scales
  zeroPoint: [0, 0, 0], // Per-channel zero points
  channels: 3,
  dataLayout: 'chw', // Specify layout for per-channel
});
```

##### Automatic Parameter Calculation

Calculate optimal quantization parameters from your data:

```typescript
import {
  calculateQuantizationParams,
  quantize,
} from 'react-native-vision-utils';

// Calculate optimal params for your data range
const params = await calculateQuantizationParams(result.data, {
  mode: 'per-tensor',
  dtype: 'uint8',
});

console.log(params.scale); // Calculated scale
console.log(params.zeroPoint); // Calculated zero point
console.log(params.min); // Data min value
console.log(params.max); // Data max value

// Use calculated params for quantization
const quantized = await quantize(result.data, {
  mode: 'per-tensor',
  dtype: 'uint8',
  scale: params.scale as number,
  zeroPoint: params.zeroPoint as number,
});
```

#### `dequantize(data, options)`

Convert quantized data back to float32.

```typescript
import { dequantize } from 'react-native-vision-utils';

// Dequantize int8 data back to float
const dequantized = await dequantize(quantized.data, {
  mode: 'per-tensor',
  dtype: 'int8',
  scale: 0.0078125,
  zeroPoint: 0,
});

console.log(dequantized.data); // Float32Array
```

##### Quantization Options

| Property     | Type                            | Required    | Description                                   |
| ------------ | ------------------------------- | ----------- | --------------------------------------------- |
| `mode`       | `'per-tensor' \| 'per-channel'` | Yes         | Quantization mode                             |
| `dtype`      | `'int8' \| 'uint8' \| 'int16'`  | Yes         | Output data type                              |
| `scale`      | `number \| number[]`            | Yes         | Scale factor(s) - array for per-channel       |
| `zeroPoint`  | `number \| number[]`            | Yes         | Zero point(s) - array for per-channel         |
| `channels`   | `number`                        | Per-channel | Number of channels (required for per-channel) |
| `dataLayout` | `'hwc' \| 'chw'`                | Per-channel | Data layout (default: 'hwc')                  |

##### Quantization Formulas

```
Quantize:   q = round(value / scale + zeroPoint)
Dequantize: value = (q - zeroPoint) * scale
```

---

### 📦 Bounding Box Utilities

High-performance utilities for working with object detection outputs.

#### `convertBoxFormat(boxes, options)`

Convert between bounding box formats (xyxy, xywh, cxcywh).

```typescript
import { convertBoxFormat } from 'react-native-vision-utils';

// Convert YOLO format (center x, center y, width, height) to corners
const result = await convertBoxFormat(
  [[320, 240, 100, 80]], // [cx, cy, w, h]
  { fromFormat: 'cxcywh', toFormat: 'xyxy' }
);
console.log(result.boxes); // [[270, 200, 370, 280]] - [x1, y1, x2, y2]
```

##### Box Formats

| Format   | Description                      | Values                  |
| -------- | -------------------------------- | ----------------------- |
| `xyxy`   | Two corners                      | [x1, y1, x2, y2]        |
| `xywh`   | Top-left corner + dimensions     | [x, y, width, height]   |
| `cxcywh` | Center + dimensions (YOLO style) | [cx, cy, width, height] |

#### `scaleBoxes(boxes, options)`

Scale bounding boxes between different image dimensions.

```typescript
import { scaleBoxes } from 'react-native-vision-utils';

// Scale boxes from 640x640 model input back to 1920x1080 original
const result = await scaleBoxes(
  [[100, 100, 200, 200]], // detections in 640x640 space
  {
    fromWidth: 640,
    fromHeight: 640,
    toWidth: 1920,
    toHeight: 1080,
    format: 'xyxy',
  }
);
// Boxes are now in original image coordinates
```

#### `clipBoxes(boxes, options)`

Clip bounding boxes to image boundaries.

```typescript
import { clipBoxes } from 'react-native-vision-utils';

const result = await clipBoxes(
  [[-10, 50, 700, 500]], // box extends beyond image
  { width: 640, height: 480, format: 'xyxy' }
);
console.log(result.boxes); // [[0, 50, 640, 480]]
```

#### `calculateIoU(box1, box2, format)`

Calculate Intersection over Union between two boxes.

```typescript
import { calculateIoU } from 'react-native-vision-utils';

const result = await calculateIoU(
  [100, 100, 200, 200],
  [150, 150, 250, 250],
  'xyxy'
);
console.log(result.iou); // ~0.142 (14% overlap)
console.log(result.intersection); // Area of overlap
console.log(result.union); // Total covered area
```

#### `nonMaxSuppression(detections, options)`

Apply Non-Maximum Suppression to filter overlapping detections.

```typescript
import { nonMaxSuppression } from 'react-native-vision-utils';

const detections = [
  { box: [100, 100, 200, 200], score: 0.9, classIndex: 0 },
  { box: [110, 110, 210, 210], score: 0.8, classIndex: 0 }, // overlaps
  { box: [300, 300, 400, 400], score: 0.7, classIndex: 1 },
];

const result = await nonMaxSuppression(detections, {
  iouThreshold: 0.5,
  scoreThreshold: 0.3,
  maxDetections: 100,
});
// Keeps first and third detection; second is suppressed due to overlap
```

---

### 📐 Letterbox Padding

YOLO-style letterbox preprocessing for preserving aspect ratio.

#### `letterbox(source, options)`

Apply letterbox padding to an image.

```typescript
import { letterbox, getPixelData } from 'react-native-vision-utils';

// Letterbox a 1920x1080 image to 640x640 for YOLO
const result = await letterbox(
  { type: 'file', value: '/path/to/image.jpg' },
  {
    targetWidth: 640,
    targetHeight: 640,
    fillColor: [114, 114, 114], // YOLO gray
  }
);

console.log(result.imageBase64); // Letterboxed image
console.log(result.letterboxInfo); // Transform info for reverse mapping
// letterboxInfo: { scale, offset, originalSize, letterboxedSize }

// Get pixel data from letterboxed image
const pixels = await getPixelData({
  source: { type: 'base64', value: result.imageBase64 },
  colorFormat: 'rgb',
  normalization: { preset: 'scale' },
  dataLayout: 'nchw',
});
```

#### `reverseLetterbox(boxes, options)`

Transform detection boxes back to original image coordinates.

```typescript
import { letterbox, reverseLetterbox } from 'react-native-vision-utils';

// First letterbox the image
const lb = await letterbox(source, { targetWidth: 640, targetHeight: 640 });

// Run detection on letterboxed image...
const detections = [
  { box: [100, 150, 200, 250], score: 0.9 }, // in 640x640 space
];

// Reverse transform to original coordinates
const result = await reverseLetterbox(
  detections.map((d) => d.box),
  {
    scale: lb.letterboxInfo.scale,
    offset: lb.letterboxInfo.offset,
    originalSize: lb.letterboxInfo.originalSize,
    format: 'xyxy',
  }
);
// Boxes are now in original 1920x1080 coordinates
```

---

### 🎨 Drawing & Visualization

Utilities for visualizing detection and segmentation results.

#### `drawBoxes(source, boxes, options)`

Draw bounding boxes with labels on an image.

```typescript
import { drawBoxes } from 'react-native-vision-utils';

const result = await drawBoxes(
  { type: 'base64', value: imageBase64 },
  [
    { box: [100, 100, 200, 200], label: 'person', score: 0.95, classIndex: 0 },
    { box: [300, 150, 400, 350], label: 'dog', score: 0.87, classIndex: 16 },
    { box: [500, 200, 600, 300], color: [255, 0, 0] }, // custom red color
  ],
  {
    lineWidth: 3,
    fontSize: 14,
    drawLabels: true,
    labelBackgroundAlpha: 0.7,
  }
);

// Display result.imageBase64
console.log(result.width, result.height);
console.log(result.processingTimeMs);
```

#### `drawKeypoints(source, keypoints, options)`

Draw pose keypoints with skeleton connections.

```typescript
import { drawKeypoints } from 'react-native-vision-utils';

// COCO pose keypoints
const keypoints = [
  { x: 320, y: 100, confidence: 0.9 }, // nose
  { x: 310, y: 90, confidence: 0.85 }, // left_eye
  { x: 330, y: 90, confidence: 0.87 }, // right_eye
  // ... 17 keypoints total
];

const result = await drawKeypoints(
  { type: 'base64', value: imageBase64 },
  keypoints,
  {
    pointRadius: 5,
    minConfidence: 0.5,
    lineWidth: 2,
    skeleton: [
      { from: 0, to: 1 }, // nose to left_eye
      { from: 0, to: 2 }, // nose to right_eye
      { from: 5, to: 6 }, // left_shoulder to right_shoulder
      // ... more connections
    ],
  }
);
```

#### `overlayMask(source, mask, options)`

Overlay a segmentation mask on an image.

```typescript
import { overlayMask } from 'react-native-vision-utils';

// Segmentation output (class indices for each pixel)
const segmentationMask = new Array(160 * 160).fill(0); // 160x160 mask
segmentationMask.fill(1, 0, 5000); // Some pixels class 1
segmentationMask.fill(15, 5000, 10000); // Some pixels class 15

const result = await overlayMask(
  { type: 'base64', value: imageBase64 },
  segmentationMask,
  {
    maskWidth: 160,
    maskHeight: 160,
    alpha: 0.5,
    isClassMask: true, // Each value is a class index
  }
);
```

#### `overlayHeatmap(source, heatmap, options)`

Overlay an attention/saliency heatmap on an image.

```typescript
import { overlayHeatmap } from 'react-native-vision-utils';

// Attention weights from a ViT model (14x14 patches)
const attentionWeights = new Array(14 * 14).fill(0).map(() => Math.random());

const result = await overlayHeatmap(
  { type: 'base64', value: imageBase64 },
  attentionWeights,
  {
    heatmapWidth: 14,
    heatmapHeight: 14,
    alpha: 0.6,
    colorScheme: 'jet', // 'jet', 'hot', or 'viridis'
  }
);
```

##### Heatmap Color Schemes

| Scheme    | Description                                  |
| --------- | -------------------------------------------- |
| `jet`     | Blue → Cyan → Green → Yellow → Red (default) |
| `hot`     | Black → Red → Yellow → White                 |
| `viridis` | Purple → Blue → Green → Yellow               |

---

## 📝 Type Reference

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
  | 'contain' // Fit within target, padColor padding
  | 'stretch' // Stretch to fill (may distort)
  | 'letterbox'; // Fit within target, letterboxColor padding (YOLO-style)

interface ResizeOptions {
  width: number;
  height: number;
  strategy?: ResizeStrategy;
  padColor?: [number, number, number, number]; // RGBA for 'contain' (default: black)
  letterboxColor?: [number, number, number]; // RGB for 'letterbox' (default: [114, 114, 114])
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

---

## ⚠️ Error Handling

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

| Code | Description |
|:-----|:------------|
| `INVALID_SOURCE` | Invalid image source |
| `LOAD_FAILED` | Failed to load image |
| `INVALID_ROI` | Invalid region of interest |
| `PROCESSING_FAILED` | Processing error |
| `INVALID_OPTIONS` | Invalid options provided |
| `INVALID_CHANNEL` | Invalid channel index |
| `INVALID_PATCH` | Invalid patch dimensions |
| `DIMENSION_MISMATCH` | Tensor dimension mismatch |
| `EMPTY_BATCH` | Empty batch provided |
| `UNKNOWN` | Unknown error |

---

## 🔄 Platform Considerations

> ⚠️ **Cross-Platform Variance**

When processing identical images on iOS and Android, you may observe slight differences in pixel values (typically 5-15%). This is **expected behavior** due to:

| Source | Description |
|:-------|:------------|
| 🖼️ **Image Decoders** | iOS uses ImageIO, Android uses Skia - JPEG/PNG decoding differs slightly |
| 📐 **Resize Algorithms** | Bilinear/bicubic interpolation implementations vary between platforms |
| 🎨 **Color Space Handling** | sRGB gamma curves may be applied differently |
| 🔢 **Floating-Point Precision** | Minor rounding differences in native math operations |

**Impact on ML Models:**
- ✅ Relative comparisons work identically across platforms
- ✅ Classification/detection results are consistent  
- ✅ Threshold-based logic (`value > 0`, `value < 1`) works the same
- ⚠️ Exact numerical equality between platforms is not guaranteed

**Best Practices:**
```typescript
// ✅ Good - threshold-based comparison
const isValid = result.actualMin >= 0 && result.actualMax <= 1;

// ✅ Good - relative comparison
const isBlurry = blurScore < 100;

// ⚠️ Avoid - exact value comparison across platforms
const isExact = result.actualMin === 0.0431; // May differ on iOS
```

This variance is standard across cross-platform ML libraries (TensorFlow Lite, ONNX Runtime, PyTorch Mobile).

---

## ⚡ Performance Tips

> 💡 **Pro Tips for optimal performance**

| Tip | Description |
|:----|:------------|
| 🎯 **Resize Strategies** | Use `letterbox` for YOLO, `cover` for classification models |
| 📦 **Batch Processing** | Use `batchGetPixelData` with appropriate concurrency for multiple images |
| 💾 **Cache Management** | Call `clearCache()` when memory is constrained |
| ⚙️ **Model Presets** | Pre-configured settings are optimized for each model |
| 🔄 **Data Layout** | Choose the right `dataLayout` upfront to avoid unnecessary conversions |

---

## 🤝 Contributing

We welcome contributions! Please see our guides:

| Resource | Description |
|:---------|:------------|
| 📋 [Development workflow](CONTRIBUTING.md#development-workflow) | How to set up your dev environment |
| 🔀 [Sending a pull request](CONTRIBUTING.md#sending-a-pull-request) | Guidelines for submitting PRs |
| 📜 [Code of conduct](CODE_OF_CONDUCT.md) | Community guidelines |

---

## 📄 License

MIT

---

<p align="center">
  Made with ❤️ using <a href="https://github.com/callstack/react-native-builder-bob">create-react-native-library</a>
</p>

<p align="center">
  <a href="#react-native-vision-utils">⬆️ Back to Top</a>
</p>
