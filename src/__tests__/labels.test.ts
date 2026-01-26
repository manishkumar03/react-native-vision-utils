/**
 * Label Database Tests
 */

import type { LabelDataset } from '../types';
import {
  mockGetLabel,
  mockGetTopLabels,
  mockGetAllLabels,
  mockGetDatasetInfo,
  mockGetAvailableDatasets,
} from './testSetup';
import * as VisionUtils from '../index';

describe('Label Database', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getLabel', () => {
    it('should get label by index with default dataset (coco)', async () => {
      mockGetLabel.mockResolvedValue('person');

      const result = await VisionUtils.getLabel(0);

      expect(mockGetLabel).toHaveBeenCalledWith(0, 'coco', false);
      expect(result).toBe('person');
    });

    it('should get label with metadata', async () => {
      mockGetLabel.mockResolvedValue({
        index: 0,
        name: 'person',
        displayName: 'Person',
        supercategory: 'person',
      });

      const result = await VisionUtils.getLabel(0, 'coco', true);

      expect(mockGetLabel).toHaveBeenCalledWith(0, 'coco', true);
      expect(result).toEqual({
        index: 0,
        name: 'person',
        displayName: 'Person',
        supercategory: 'person',
      });
    });

    it('should get label from different datasets', async () => {
      const datasets: LabelDataset[] = [
        'coco',
        'coco91',
        'imagenet',
        'voc',
        'cifar10',
        'cifar100',
        'places365',
        'ade20k',
      ];

      for (const dataset of datasets) {
        mockGetLabel.mockResolvedValue('label');
        await VisionUtils.getLabel(0, dataset);

        expect(mockGetLabel).toHaveBeenLastCalledWith(0, dataset, false);
      }
    });

    it('should reject for invalid index', async () => {
      mockGetLabel.mockRejectedValue(
        new Error('Index 100 out of range for dataset coco with 80 classes')
      );

      await expect(VisionUtils.getLabel(100)).rejects.toThrow('out of range');
    });

    it('should reject for negative index', async () => {
      await expect(VisionUtils.getLabel(-1)).rejects.toThrow();
    });
  });

  describe('getTopLabels', () => {
    it('should get top 5 labels by default', async () => {
      const mockScores = Array(80).fill(0.01);
      mockScores[0] = 0.9; // person
      mockScores[15] = 0.8; // cat

      mockGetTopLabels.mockResolvedValue([
        { index: 0, label: 'person', confidence: 0.9 },
        { index: 15, label: 'cat', confidence: 0.8 },
      ]);

      const result = await VisionUtils.getTopLabels(mockScores);

      expect(mockGetTopLabels).toHaveBeenCalledWith(mockScores, {
        dataset: undefined,
        k: 5,
        minConfidence: 0,
        includeMetadata: false,
      });
      expect(result).toHaveLength(2);
      expect(result[0]?.label).toBe('person');
    });

    it('should filter by minimum confidence', async () => {
      const mockScores = Array(80).fill(0.01);
      mockScores[0] = 0.9;
      mockScores[15] = 0.3; // Below threshold

      mockGetTopLabels.mockResolvedValue([
        { index: 0, label: 'person', confidence: 0.9 },
      ]);

      const result = await VisionUtils.getTopLabels(mockScores, {
        minConfidence: 0.5,
      });

      expect(mockGetTopLabels).toHaveBeenCalledWith(mockScores, {
        dataset: undefined,
        k: 5,
        minConfidence: 0.5,
        includeMetadata: false,
      });
      expect(result).toHaveLength(1);
    });

    it('should return custom k results', async () => {
      const mockScores = Array(80).fill(0.01);
      mockScores[0] = 0.9;
      mockScores[1] = 0.8;
      mockScores[2] = 0.7;

      mockGetTopLabels.mockResolvedValue([
        { index: 0, label: 'person', confidence: 0.9 },
        { index: 1, label: 'bicycle', confidence: 0.8 },
        { index: 2, label: 'car', confidence: 0.7 },
      ]);

      const result = await VisionUtils.getTopLabels(mockScores, { k: 3 });

      expect(result).toHaveLength(3);
    });

    it('should include metadata when requested', async () => {
      const mockScores = Array(80).fill(0.01);
      mockScores[0] = 0.9;

      mockGetTopLabels.mockResolvedValue([
        {
          index: 0,
          label: 'person',
          confidence: 0.9,
          supercategory: 'person',
        },
      ]);

      const result = await VisionUtils.getTopLabels(mockScores, {
        includeMetadata: true,
      });

      expect(result[0]).toHaveProperty('supercategory');
    });

    it('should reject for empty scores array', async () => {
      await expect(VisionUtils.getTopLabels([])).rejects.toThrow();
    });
  });

  describe('getAllLabels', () => {
    it('should return all labels for coco', async () => {
      mockGetAllLabels.mockResolvedValue([
        'person',
        'bicycle',
        'car',
        // ... 80 labels
      ]);

      const result = await VisionUtils.getAllLabels('coco');

      expect(mockGetAllLabels).toHaveBeenCalledWith('coco');
      expect(Array.isArray(result)).toBe(true);
    });

    it('should reject for invalid dataset', async () => {
      mockGetAllLabels.mockRejectedValue(new Error('Unknown dataset: invalid'));

      await expect(
        VisionUtils.getAllLabels('invalid' as LabelDataset)
      ).rejects.toThrow('Unknown dataset');
    });
  });

  describe('getDatasetInfo', () => {
    it('should return dataset information', async () => {
      mockGetDatasetInfo.mockResolvedValue({
        name: 'coco',
        numClasses: 80,
        description: 'COCO 2017 object detection labels (80 classes)',
        isAvailable: true,
      });

      const result = await VisionUtils.getDatasetInfo('coco');

      expect(result.name).toBe('coco');
      expect(result.numClasses).toBe(80);
      expect(result.isAvailable).toBe(true);
    });

    it('should return correct class counts for each dataset', async () => {
      const expectedCounts: Record<LabelDataset, number> = {
        coco: 80,
        coco91: 91,
        imagenet: 1000,
        imagenet21k: 21841,
        voc: 21,
        cifar10: 10,
        cifar100: 100,
        places365: 365,
        ade20k: 150,
      };

      for (const [dataset, count] of Object.entries(expectedCounts)) {
        mockGetDatasetInfo.mockResolvedValue({
          name: dataset,
          numClasses: count,
          description: `${dataset} labels`,
          isAvailable: true,
        });

        const result = await VisionUtils.getDatasetInfo(
          dataset as LabelDataset
        );
        expect(result.numClasses).toBe(count);
      }
    });
  });

  describe('getAvailableDatasets', () => {
    it('should return list of available datasets', async () => {
      mockGetAvailableDatasets.mockResolvedValue([
        'coco',
        'coco91',
        'imagenet',
        'voc',
        'cifar10',
        'cifar100',
        'places365',
        'ade20k',
      ]);

      const result = await VisionUtils.getAvailableDatasets();

      expect(result).toContain('coco');
      expect(result).toContain('imagenet');
      expect(result.length).toBeGreaterThan(0);
    });
  });
});

describe('Label Database - Edge Cases', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should handle boundary index (first label)', async () => {
    mockGetLabel.mockResolvedValue('person');
    const result = await VisionUtils.getLabel(0);
    expect(result).toBe('person');
  });

  it('should handle boundary index (last label for coco)', async () => {
    mockGetLabel.mockResolvedValue('toothbrush');
    const result = await VisionUtils.getLabel(79); // Last COCO class
    expect(result).toBe('toothbrush');
  });

  it('should handle empty scores array gracefully', async () => {
    await expect(VisionUtils.getTopLabels([])).rejects.toThrow();
  });

  it('should handle all zero scores', async () => {
    mockGetTopLabels.mockResolvedValue([
      { index: 0, label: 'person', confidence: 0 },
      { index: 1, label: 'bicycle', confidence: 0 },
    ]);

    const result = await VisionUtils.getTopLabels(Array(80).fill(0));
    expect(result).toBeDefined();
  });

  it('should handle scores with NaN/Infinity', async () => {
    const scores = Array(80).fill(0.1);
    scores[0] = NaN;
    scores[1] = Infinity;

    mockGetTopLabels.mockResolvedValue([]);

    // Should not crash
    await VisionUtils.getTopLabels(scores);
  });
});
