import {
  mockQuantize,
  mockDequantize,
  mockCalculateQuantizationParams,
} from './jest.setup';
import { quantize, dequantize, calculateQuantizationParams } from '../index';
import type {
  QuantizeResult,
  DequantizeResult,
  QuantizationParams,
} from '../types';

describe('Quantization Functions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('quantize', () => {
    describe('per-tensor quantization', () => {
      it('should quantize float data to uint8 with per-tensor mode', async () => {
        const mockResult: QuantizeResult = {
          data: new Uint8Array([0, 64, 128, 192, 255]),
          scale: 0.00392157,
          zeroPoint: 0,
          dtype: 'uint8',
          mode: 'per-tensor',
          processingTimeMs: 0.5,
        };
        mockQuantize.mockResolvedValue(mockResult);

        const inputData = [0.0, 0.25, 0.5, 0.75, 1.0];
        const result = await quantize(inputData, {
          mode: 'per-tensor',
          dtype: 'uint8',
          scale: 0.00392157,
          zeroPoint: 0,
        });

        expect(mockQuantize).toHaveBeenCalledTimes(1);
        expect(result.dtype).toBe('uint8');
        expect(result.mode).toBe('per-tensor');
        expect(result.scale).toBe(0.00392157);
        expect(result.zeroPoint).toBe(0);
      });

      it('should quantize float data to int8 with per-tensor mode', async () => {
        const mockResult: QuantizeResult = {
          data: new Int8Array([-128, -64, 0, 64, 127]),
          scale: 0.0078125,
          zeroPoint: 0,
          dtype: 'int8',
          mode: 'per-tensor',
          processingTimeMs: 0.5,
        };
        mockQuantize.mockResolvedValue(mockResult);

        const inputData = [-1.0, -0.5, 0.0, 0.5, 1.0];
        const result = await quantize(inputData, {
          mode: 'per-tensor',
          dtype: 'int8',
          scale: 0.0078125,
          zeroPoint: 0,
        });

        expect(result.dtype).toBe('int8');
        expect(result.mode).toBe('per-tensor');
      });

      it('should quantize float data to int16 with per-tensor mode', async () => {
        const mockResult: QuantizeResult = {
          data: new Int16Array([0, 8192, 16384, 24576, 32767]),
          scale: 0.0000305,
          zeroPoint: 0,
          dtype: 'int16',
          mode: 'per-tensor',
          processingTimeMs: 0.5,
        };
        mockQuantize.mockResolvedValue(mockResult);

        const inputData = [0.0, 0.25, 0.5, 0.75, 1.0];
        const result = await quantize(inputData, {
          mode: 'per-tensor',
          dtype: 'int16',
          scale: 0.0000305,
          zeroPoint: 0,
        });

        expect(result.dtype).toBe('int16');
      });

      it('should handle zero point offset for uint8', async () => {
        const mockResult: QuantizeResult = {
          data: new Uint8Array([128, 192, 255, 64, 0]),
          scale: 0.00392157,
          zeroPoint: 128,
          dtype: 'uint8',
          mode: 'per-tensor',
          processingTimeMs: 0.5,
        };
        mockQuantize.mockResolvedValue(mockResult);

        const inputData = [0.0, 0.25, 0.5, -0.25, -0.5];
        const result = await quantize(inputData, {
          mode: 'per-tensor',
          dtype: 'uint8',
          scale: 0.00392157,
          zeroPoint: 128,
        });

        expect(result.zeroPoint).toBe(128);
      });
    });

    describe('per-channel quantization', () => {
      it('should quantize with per-channel scales and zero points', async () => {
        const mockResult: QuantizeResult = {
          data: new Int8Array(new Array(224 * 224 * 3).fill(0)),
          scale: [0.0123, 0.0156, 0.0189],
          zeroPoint: [0, 0, 0],
          dtype: 'int8',
          mode: 'per-channel',
          processingTimeMs: 0.5,
        };
        mockQuantize.mockResolvedValue(mockResult);

        const inputData = new Array(224 * 224 * 3).fill(0.5);
        const result = await quantize(inputData, {
          mode: 'per-channel',
          dtype: 'int8',
          scale: [0.0123, 0.0156, 0.0189],
          zeroPoint: [0, 0, 0],
          channels: 3,
          dataLayout: 'chw',
        });

        expect(result.mode).toBe('per-channel');
        expect(Array.isArray(result.scale)).toBe(true);
        expect((result.scale as number[]).length).toBe(3);
      });

      it('should handle HWC data layout for per-channel', async () => {
        const mockResult: QuantizeResult = {
          data: new Uint8Array(new Array(224 * 224 * 3).fill(128)),
          scale: [0.01, 0.02, 0.03],
          zeroPoint: [128, 128, 128],
          dtype: 'uint8',
          mode: 'per-channel',
          processingTimeMs: 0.5,
        };
        mockQuantize.mockResolvedValue(mockResult);

        const inputData = new Array(224 * 224 * 3).fill(0.0);
        await quantize(inputData, {
          mode: 'per-channel',
          dtype: 'uint8',
          scale: [0.01, 0.02, 0.03],
          zeroPoint: [128, 128, 128],
          channels: 3,
          dataLayout: 'hwc',
        });

        expect(mockQuantize).toHaveBeenCalledWith(
          inputData,
          expect.objectContaining({
            dataLayout: 'hwc',
          })
        );
      });
    });

    describe('input validation', () => {
      it('should require scale and zeroPoint options', async () => {
        await expect(
          quantize([0.5], {
            mode: 'per-tensor',
            dtype: 'uint8',
          } as any)
        ).rejects.toThrow('scale and zeroPoint are required');
      });

      it('should accept Float32Array input', async () => {
        const mockResult: QuantizeResult = {
          data: new Uint8Array([128]),
          scale: 0.01,
          zeroPoint: 0,
          dtype: 'uint8',
          mode: 'per-tensor',
          processingTimeMs: 0.5,
        };
        mockQuantize.mockResolvedValue(mockResult);

        const inputData = new Float32Array([0.5]);
        const result = await quantize(Array.from(inputData), {
          mode: 'per-tensor',
          dtype: 'uint8',
          scale: 0.01,
          zeroPoint: 0,
        });

        expect(result).toBeDefined();
      });
    });

    describe('edge cases', () => {
      it('should handle very small scale values', async () => {
        const mockResult: QuantizeResult = {
          data: new Int16Array([32767]),
          scale: 0.00001,
          zeroPoint: 0,
          dtype: 'int16',
          mode: 'per-tensor',
          processingTimeMs: 0.5,
        };
        mockQuantize.mockResolvedValue(mockResult);

        const result = await quantize([0.32767], {
          mode: 'per-tensor',
          dtype: 'int16',
          scale: 0.00001,
          zeroPoint: 0,
        });

        expect(result.scale).toBe(0.00001);
      });

      it('should handle negative zero points for int8', async () => {
        const mockResult: QuantizeResult = {
          data: new Int8Array([0]),
          scale: 0.01,
          zeroPoint: -10,
          dtype: 'int8',
          mode: 'per-tensor',
          processingTimeMs: 0.5,
        };
        mockQuantize.mockResolvedValue(mockResult);

        const result = await quantize([0.1], {
          mode: 'per-tensor',
          dtype: 'int8',
          scale: 0.01,
          zeroPoint: -10,
        });

        expect(result.zeroPoint).toBe(-10);
      });
    });
  });

  describe('dequantize', () => {
    describe('per-tensor dequantization', () => {
      it('should dequantize uint8 data to float32', async () => {
        const mockResult: DequantizeResult = {
          data: new Float32Array([0.0, 0.25, 0.5, 0.75, 1.0]),
          processingTimeMs: 0.5,
        };
        mockDequantize.mockResolvedValue(mockResult);

        const inputData = [0, 64, 128, 192, 255];
        const result = await dequantize(inputData, {
          mode: 'per-tensor',
          dtype: 'uint8',
          scale: 0.00392157,
          zeroPoint: 0,
        });

        expect(mockDequantize).toHaveBeenCalledTimes(1);
        expect(result.data).toBeInstanceOf(Float32Array);
      });

      it('should dequantize int8 data to float32', async () => {
        const mockResult: DequantizeResult = {
          data: new Float32Array([-1.0, -0.5, 0.0, 0.5, 1.0]),
          processingTimeMs: 0.5,
        };
        mockDequantize.mockResolvedValue(mockResult);

        const inputData = [-128, -64, 0, 64, 127];
        const result = await dequantize(inputData, {
          mode: 'per-tensor',
          dtype: 'int8',
          scale: 0.0078125,
          zeroPoint: 0,
        });

        expect(result.data).toBeInstanceOf(Float32Array);
      });

      it('should handle zero point offset during dequantization', async () => {
        const mockResult: DequantizeResult = {
          data: new Float32Array([0.0, 0.25, 0.5]),
          processingTimeMs: 0.5,
        };
        mockDequantize.mockResolvedValue(mockResult);

        const inputData = [128, 192, 255];
        await dequantize(inputData, {
          mode: 'per-tensor',
          dtype: 'uint8',
          scale: 0.00392157,
          zeroPoint: 128,
        });

        expect(mockDequantize).toHaveBeenCalledWith(
          inputData,
          expect.objectContaining({
            zeroPoint: 128,
          })
        );
      });
    });

    describe('per-channel dequantization', () => {
      it('should dequantize with per-channel scales', async () => {
        const mockResult: DequantizeResult = {
          data: new Float32Array(new Array(224 * 224 * 3).fill(0.5)),
          processingTimeMs: 0.5,
        };
        mockDequantize.mockResolvedValue(mockResult);

        const inputData = new Array(224 * 224 * 3).fill(64);
        const result = await dequantize(inputData, {
          mode: 'per-channel',
          dtype: 'int8',
          scale: [0.0123, 0.0156, 0.0189],
          zeroPoint: [0, 0, 0],
          channels: 3,
          dataLayout: 'chw',
        });

        expect(result.data).toBeInstanceOf(Float32Array);
      });
    });

    describe('input validation', () => {
      it('should require scale and zeroPoint options', async () => {
        await expect(
          dequantize([128], {
            mode: 'per-tensor',
            dtype: 'uint8',
          } as any)
        ).rejects.toThrow('scale and zeroPoint are required');
      });
    });
  });

  describe('calculateQuantizationParams', () => {
    describe('per-tensor parameter calculation', () => {
      it('should calculate params for uint8 with 0-1 range data', async () => {
        const mockResult: QuantizationParams = {
          scale: 0.00392157,
          zeroPoint: 0,
          min: 0.0,
          max: 1.0,
        };
        mockCalculateQuantizationParams.mockResolvedValue(mockResult);

        const inputData = [0.0, 0.25, 0.5, 0.75, 1.0];
        const result = await calculateQuantizationParams(inputData, {
          mode: 'per-tensor',
          dtype: 'uint8',
        });

        expect(mockCalculateQuantizationParams).toHaveBeenCalledTimes(1);
        expect(result.scale).toBe(0.00392157);
        expect(result.zeroPoint).toBe(0);
        expect(result.min).toBe(0.0);
        expect(result.max).toBe(1.0);
      });

      it('should calculate params for int8 with -1 to 1 range data', async () => {
        const mockResult: QuantizationParams = {
          scale: 0.0078125,
          zeroPoint: 0,
          min: -1.0,
          max: 1.0,
        };
        mockCalculateQuantizationParams.mockResolvedValue(mockResult);

        const inputData = [-1.0, -0.5, 0.0, 0.5, 1.0];
        const result = await calculateQuantizationParams(inputData, {
          mode: 'per-tensor',
          dtype: 'int8',
        });

        expect(result.min).toBe(-1.0);
        expect(result.max).toBe(1.0);
      });

      it('should calculate params for int16', async () => {
        const mockResult: QuantizationParams = {
          scale: 0.0000305,
          zeroPoint: 0,
          min: 0.0,
          max: 1.0,
        };
        mockCalculateQuantizationParams.mockResolvedValue(mockResult);

        const inputData = [0.0, 0.5, 1.0];
        const result = await calculateQuantizationParams(inputData, {
          mode: 'per-tensor',
          dtype: 'int16',
        });

        expect(result.scale).toBeLessThan(0.001);
      });
    });

    describe('per-channel parameter calculation', () => {
      it('should calculate per-channel params for CHW layout', async () => {
        const mockResult: QuantizationParams = {
          scale: [0.0123, 0.0156, 0.0189],
          zeroPoint: [0, 0, 0],
          min: [-2.5, -2.3, -2.1],
          max: [2.5, 2.3, 2.1],
        };
        mockCalculateQuantizationParams.mockResolvedValue(mockResult);

        const inputData = new Array(224 * 224 * 3).fill(0.0);
        const result = await calculateQuantizationParams(inputData, {
          mode: 'per-channel',
          dtype: 'int8',
          channels: 3,
          dataLayout: 'chw',
        });

        expect(Array.isArray(result.scale)).toBe(true);
        expect((result.scale as number[]).length).toBe(3);
        expect(Array.isArray(result.zeroPoint)).toBe(true);
        expect((result.zeroPoint as number[]).length).toBe(3);
      });

      it('should calculate per-channel params for HWC layout', async () => {
        const mockResult: QuantizationParams = {
          scale: [0.01, 0.02, 0.03],
          zeroPoint: [128, 128, 128],
          min: [0.0, 0.0, 0.0],
          max: [1.0, 1.0, 1.0],
        };
        mockCalculateQuantizationParams.mockResolvedValue(mockResult);

        const inputData = new Array(224 * 224 * 3).fill(0.5);
        await calculateQuantizationParams(inputData, {
          mode: 'per-channel',
          dtype: 'uint8',
          channels: 3,
          dataLayout: 'hwc',
        });

        expect(mockCalculateQuantizationParams).toHaveBeenCalledWith(
          inputData,
          expect.objectContaining({
            dataLayout: 'hwc',
            channels: 3,
          })
        );
      });
    });

    describe('default options', () => {
      it('should use default options when not specified', async () => {
        const mockResult: QuantizationParams = {
          scale: 0.00392157,
          zeroPoint: 0,
          min: 0.0,
          max: 1.0,
        };
        mockCalculateQuantizationParams.mockResolvedValue(mockResult);

        const inputData = [0.0, 0.5, 1.0];
        await calculateQuantizationParams(inputData);

        expect(mockCalculateQuantizationParams).toHaveBeenCalledWith(
          inputData,
          expect.objectContaining({
            mode: 'per-tensor',
            dtype: 'int8',
          })
        );
      });
    });

    describe('input validation', () => {
      it('should call native module with data array', async () => {
        const mockResult: QuantizationParams = {
          scale: 0.01,
          zeroPoint: 0,
          min: 0.0,
          max: 1.0,
        };
        mockCalculateQuantizationParams.mockResolvedValue(mockResult);

        const inputData = [0.1, 0.5, 0.9];
        await calculateQuantizationParams(inputData);

        expect(mockCalculateQuantizationParams).toHaveBeenCalledWith(
          inputData,
          expect.any(Object)
        );
      });
    });
  });

  describe('quantization roundtrip', () => {
    it('should maintain data integrity through quantize-dequantize cycle', async () => {
      const quantizeResult: QuantizeResult = {
        data: new Uint8Array([0, 64, 128, 192, 255]),
        scale: 0.00392157,
        zeroPoint: 0,
        dtype: 'uint8',
        mode: 'per-tensor',
        processingTimeMs: 0.5,
      };
      mockQuantize.mockResolvedValue(quantizeResult);

      const dequantizeResult: DequantizeResult = {
        data: new Float32Array([0.0, 0.251, 0.502, 0.753, 1.0]),
        processingTimeMs: 0.5,
      };
      mockDequantize.mockResolvedValue(dequantizeResult);

      const originalData = [0.0, 0.25, 0.5, 0.75, 1.0];

      const quantized = await quantize(originalData, {
        mode: 'per-tensor',
        dtype: 'uint8',
        scale: 0.00392157,
        zeroPoint: 0,
      });

      const dequantized = await dequantize(Array.from(quantized.data), {
        mode: 'per-tensor',
        dtype: 'uint8',
        scale: 0.00392157,
        zeroPoint: 0,
      });

      expect(mockQuantize).toHaveBeenCalled();
      expect(mockDequantize).toHaveBeenCalled();
      expect(dequantized.data.length).toBe(originalData.length);
    });
  });

  describe('TFLite compatibility', () => {
    it('should support typical TFLite uint8 quantization params', async () => {
      const mockResult: QuantizeResult = {
        data: new Uint8Array([128, 148, 168, 188, 208]),
        scale: 0.0078125,
        zeroPoint: 128,
        dtype: 'uint8',
        mode: 'per-tensor',
        processingTimeMs: 0.5,
      };
      mockQuantize.mockResolvedValue(mockResult);

      const result = await quantize([0.0, 0.156, 0.312, 0.468, 0.625], {
        mode: 'per-tensor',
        dtype: 'uint8',
        scale: 0.0078125,
        zeroPoint: 128,
      });

      expect(result.scale).toBe(0.0078125);
      expect(result.zeroPoint).toBe(128);
    });

    it('should support typical TFLite int8 quantization params', async () => {
      const mockResult: QuantizeResult = {
        data: new Int8Array([0, 20, 40, 60, 80]),
        scale: 0.0078125,
        zeroPoint: 0,
        dtype: 'int8',
        mode: 'per-tensor',
        processingTimeMs: 0.5,
      };
      mockQuantize.mockResolvedValue(mockResult);

      const result = await quantize([0.0, 0.156, 0.312, 0.468, 0.625], {
        mode: 'per-tensor',
        dtype: 'int8',
        scale: 0.0078125,
        zeroPoint: 0,
      });

      expect(result.zeroPoint).toBe(0);
    });
  });
});
