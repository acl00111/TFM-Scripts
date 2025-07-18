# InstanceSegEvaluator compatible with COCO-style instance segmentation datasets
import logging
import os
import numpy as np
from collections import OrderedDict, defaultdict
import torch
import itertools
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from datetime import datetime

class InstanceSegEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed=True, log_dir="./logs"):
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")

        # Logger único por dataset
        self._logger = logging.getLogger(f"InstanceSegEvaluator_{dataset_name}")
        self._logger.setLevel(logging.DEBUG)

        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"instance_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_filename for h in self._logger.handlers):
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(file_handler)

        meta = MetadataCatalog.get(dataset_name)
        self._class_names = meta.thing_classes

        if not hasattr(meta, "thing_dataset_id_to_contiguous_id"):
            meta.thing_dataset_id_to_contiguous_id = {i: i for i in range(len(meta.thing_classes))}
        self._contiguous_id_to_dataset_id = {
            v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()
        }

        self._json_file = meta.json_file
        self._coco_gt = COCO(self._json_file)

        self._predictions = defaultdict(list)

    def reset(self):
        self._predictions = defaultdict(list)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            height = input["height"]
            width = input["width"]
            pred_instances = output["instances"].to(self._cpu_device)

            for i in range(len(pred_instances)):
                pred = pred_instances[i]
                if not pred.has("pred_masks"):
                    continue
                mask = pred.pred_masks.numpy()
                category_id = int(pred.pred_classes)
                score = float(pred.scores)

                self._logger.debug(f"Processed prediction for image {image_id}, category {category_id}, score {score}, mask shape {mask.shape}")

                self._predictions[(image_id, category_id)].append({
                    "mask": mask,
                    "score": score,
                    "height": height,
                    "width": width
                })

    def evaluate(self):
        if self._distributed:
            synchronize()
            pred_list = all_gather(self._predictions)
            self._predictions = defaultdict(list)
            for pred in pred_list:
                for key, val in pred.items():
                    self._predictions[key].extend(val)
            if not is_main_process():
                return

        if len(self._predictions) == 0:
            self._logger.warning("No predictions to evaluate.")
            return {}

        per_class_scores = defaultdict(list)

        for (image_id, contiguous_category), preds in self._predictions.items():
            category = self._contiguous_id_to_dataset_id.get(contiguous_category, contiguous_category)
            ann_ids = self._coco_gt.getAnnIds(imgIds=image_id, catIds=[category], iscrowd=None)
            self._logger.debug(f"Evaluating image {image_id}, category {category}, found ann_ids: {ann_ids}")
            anns = self._coco_gt.loadAnns(ann_ids)

            if len(anns) == 0:
                gt_mask = np.zeros((preds[0]["height"], preds[0]["width"]), dtype=np.bool_)
            else:
                gt_mask = np.zeros((preds[0]["height"], preds[0]["width"]), dtype=np.bool_)
                for ann in anns:
                    rle = self._coco_gt.annToRLE(ann)
                    m = mask_util.decode(rle).astype(np.bool_)
                    gt_mask = np.logical_or(gt_mask, m)

            # Combina todas las máscaras predichas para este (image_id, category)
            pred_mask = np.zeros_like(gt_mask, dtype=np.bool_)
            for pred in preds:
                pred_mask = np.logical_or(pred_mask, pred["mask"][0])

            if gt_mask.sum() == 0 and pred_mask.sum() == 0:
                dice, f1 = 1.0, 1.0
            else:
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                gt_sum = gt_mask.sum()
                pred_sum = pred_mask.sum()

                dice = (2 * intersection) / (gt_sum + pred_sum + 1e-6)
                precision = intersection / (pred_sum + 1e-6)
                recall = intersection / (gt_sum + 1e-6)
                f1 = (2 * precision * recall) / (precision + recall + 1e-6)

            self._logger.debug(f"Image {image_id}, Dice={dice:.4f}, F1={f1:.4f}, intersection={intersection}, gt_sum={gt_sum}, pred_sum={pred_sum}")
            per_class_scores[contiguous_category].append((dice, f1))

        results = OrderedDict()
        for cat_id, scores in per_class_scores.items():
            if len(scores) == 0:
                continue
            mean_dice = np.mean([s[0] for s in scores])
            mean_f1 = np.mean([s[1] for s in scores])
            cat_name = self._class_names[cat_id] if cat_id < len(self._class_names) else str(cat_id)
            results[f"Dice-{cat_name}"] = 100 * mean_dice
            results[f"F1-{cat_name}"] = 100 * mean_f1

        dice_vals = [v / 100 for k, v in results.items() if k.startswith("Dice-")]
        f1_vals = [v / 100 for k, v in results.items() if k.startswith("F1-")]

        results["mean_dice"] = 100 * np.mean(dice_vals)
        results["mean_f1"] = 100 * np.mean(f1_vals)

        self._logger.info("Final evaluation results:")
        for k, v in results.items():
            self._logger.info(f"{k}: {v:.2f}")

        return OrderedDict({"instance_seg": results})
