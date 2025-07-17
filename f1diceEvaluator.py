# evaluators.py
import numpy as np
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import BitMasks, Instances
from detectron2.data.detection_utils import annotations_to_instances
from detectron2.utils.comm import all_gather, is_main_process
from scipy.ndimage import label

class SegEvaluator(DatasetEvaluator):
    def __init__(self, iou_thresh=0.5):
        self.iou_thresh = iou_thresh
        self._reset()

    def _reset(self):
        self.inter = 0
        self.union = 0
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def reset(self):
        self._reset()

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            h, w = inp["height"], inp["width"]

            # --- Reconstruir GT desde anotaciones (formato COCO) ---
            gt_instances = annotations_to_instances(inp["annotations"], image_shape=(h, w))

            if hasattr(gt_instances, "gt_masks") and len(gt_instances) > 0:
                gt_masks = gt_instances.gt_masks.tensor.cpu().numpy().astype(bool)
            else:
                gt_masks = np.zeros((0, h, w), dtype=bool)

            # --- Obtener máscaras predichas ---
            if "instances" in out and len(out["instances"]) > 0:
                pred_masks = out["instances"].pred_masks.cpu().numpy().astype(bool)
            else:
                pred_masks = np.zeros((0, h, w), dtype=bool)

            # ---------- Dice (volumétrico por imagen) ------------
            gt_union = gt_masks.any(axis=0)
            pred_union = pred_masks.any(axis=0)
            inter = np.logical_and(gt_union, pred_union).sum()
            union = gt_union.sum() + pred_union.sum()

            self.inter += 2 * inter
            self.union += union

            # ---------- Lesion-wise F1 ----------------------------
            gt_cc, n_gt = label(gt_union)
            pred_cc, n_pred = label(pred_union)

            matched_gt = set()
            matched_pred = set()

            for i in range(1, n_pred + 1):
                pmask = pred_cc == i
                for j in range(1, n_gt + 1):
                    gmask = gt_cc == j
                    iou = np.logical_and(pmask, gmask).sum() / np.logical_or(pmask, gmask).sum()
                    if iou >= self.iou_thresh:
                        matched_pred.add(i)
                        matched_gt.add(j)

            self.TP += len(matched_pred)
            self.FP += n_pred - len(matched_pred)
            self.FN += n_gt - len(matched_gt)

    def evaluate(self):
        dice = self.inter / self.union if self.union > 0 else 0.0
        prec = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0.0
        rec  = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "dice": dice,
            "lesion_F1": f1,
            "lesion_precision": prec,
            "lesion_recall": rec
        }
