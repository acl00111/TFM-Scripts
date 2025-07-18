# InstanceSegEvaluator compatible with COCO-style instance segmentation datasets
import logging
import os
import numpy as np
from collections import OrderedDict
import torch
import itertools
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import pycocotools.mask as mask_util
from pycocotools.coco import COCO

class InstanceSegEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed=True):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")

        meta = MetadataCatalog.get(dataset_name)
        self._class_names = meta.thing_classes
        self._num_classes = len(self._class_names)
        self._json_file = meta.json_file
        self._coco_gt = COCO(self._json_file)

        self._predictions = []

    def reset(self):
        self._predictions = []

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

                self._predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "mask": mask,
                    "score": score,
                    "height": height,
                    "width": width
                })

    def evaluate(self):
        if self._distributed:
            synchronize()
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

        if len(self._predictions) == 0:
            self._logger.warning("No predictions to evaluate.")
            return {}

        # Compute Dice and F1
        per_image_scores = {}
        for pred in self._predictions:
            mask = pred["mask"].astype(np.bool_)
            category = pred["category_id"]
            image_id = pred["image_id"]

            # Load ground truth masks for the image and category
            ann_ids = self._coco_gt.getAnnIds(imgIds=image_id, catIds=[category], iscrowd=None)
            anns = self._coco_gt.loadAnns(ann_ids)

            gt_mask = np.zeros((pred["height"], pred["width"]), dtype=np.bool_)
            for ann in anns:
                rle = self._coco_gt.annToRLE(ann)
                m = mask_util.decode(rle).astype(np.bool_)
                gt_mask = np.logical_or(gt_mask, m)

            intersection = np.logical_and(mask, gt_mask).sum()
            union = np.logical_or(mask, gt_mask).sum()
            gt_sum = gt_mask.sum()
            pred_sum = mask.sum()

            dice = (2 * intersection) / (gt_sum + pred_sum + 1e-6)
            precision = intersection / (pred_sum + 1e-6)
            recall = intersection / (gt_sum + 1e-6)
            f1 = (2 * precision * recall) / (precision + recall + 1e-6)

            per_image_scores.setdefault(category, []).append((dice, f1))

        results = OrderedDict()
        for cat_id, scores in per_image_scores.items():
            if len(scores) == 0:
                continue
            mean_dice = np.mean([s[0] for s in scores])
            mean_f1 = np.mean([s[1] for s in scores])
            cat_name = self._class_names[cat_id] if cat_id < len(self._class_names) else str(cat_id)
            results[f"Dice-{cat_name}"] = 100 * mean_dice
            results[f"F1-{cat_name}"] = 100 * mean_f1

        results["mean_dice"] = 100 * np.mean([v for k, v in results.items() if k.startswith("Dice-")])
        results["mean_f1"] = 100 * np.mean([v for k, v in results.items() if k.startswith("F1-")])

        self._logger.info(results)
        return OrderedDict({"instance_seg": results})
