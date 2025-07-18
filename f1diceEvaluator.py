# Evaluador personalizado para segmentación por instancias (tipo COCO)
# Cálculo de DICE y F1-score a partir de máscaras de predicciones y anotaciones

import numpy as np
import pycocotools.mask as mask_util
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from collections import defaultdict, OrderedDict
import logging

setup_logger()

class InstanceSegEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)
        self.logger = logging.getLogger(__name__)
        self.reset()

    def reset(self):
        self.dice_scores = []
        self.f1_scores = []

    def _decode_masks(self, instances):
        if len(instances) == 0:
            return np.zeros((0, 1, 1), dtype=bool)
        masks = instances.pred_masks.cpu().numpy()
        return masks.astype(bool)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            height, width = input["height"], input["width"]

            gt_masks_encoded = [ann["segmentation"] for ann in input["annotations"] if ann.get("iscrowd", 0) == 0]
            gt_masks_decoded = mask_util.decode(gt_masks_encoded)
            if gt_masks_decoded.ndim == 2:
                gt_masks_decoded = gt_masks_decoded[:, :, None]  # 1 instancia
            gt_masks = gt_masks_decoded.astype(bool).transpose(2, 0, 1)

            pred_masks = output["instances"].pred_masks.cpu().numpy().astype(bool) if len(output["instances"]) > 0 else np.zeros((0, height, width), dtype=bool)

            dice, f1 = self._calculate_metrics(gt_masks, pred_masks)
            self.dice_scores.append(dice)
            self.f1_scores.append(f1)

    def _calculate_metrics(self, gt_masks, pred_masks):
        if len(gt_masks) == 0 and len(pred_masks) == 0:
            return 1.0, 1.0  # Perfect match (both empty)
        if len(gt_masks) == 0 or len(pred_masks) == 0:
            return 0.0, 0.0

        # Union todas las máscaras
        gt_union = np.any(gt_masks, axis=0)
        pred_union = np.any(pred_masks, axis=0)

        intersection = np.logical_and(gt_union, pred_union).sum()
        gt_area = gt_union.sum()
        pred_area = pred_union.sum()

        dice = 2 * intersection / (gt_area + pred_area + 1e-6)
        precision = intersection / (pred_area + 1e-6)
        recall = intersection / (gt_area + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return dice, f1

    def evaluate(self):
        if len(self.dice_scores) == 0:
            return {}
        mean_dice = 100 * np.mean(self.dice_scores)
        mean_f1 = 100 * np.mean(self.f1_scores)
        results = OrderedDict({
            "instance_seg": {
                "mean_dice": mean_dice,
                "mean_f1": mean_f1,
            }
        })
        self.logger.info(results)
        return results
