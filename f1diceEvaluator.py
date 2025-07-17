# Modified SemSegEvaluator with Dice and F1 metrics
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator

try:
    import cv2  # noqa
    _CV2_IMPORTED = True
except ImportError:
    _CV2_IMPORTED = False

def load_image_into_numpy_array(filename: str, dtype: Optional[Union[np.dtype, str]] = None) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.asarray(Image.open(f), dtype=dtype)
    return array

class SemSegEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed=True, output_dir=None,
                 *, sem_seg_loading_fn=load_image_into_numpy_array,
                 num_classes=None, ignore_label=None):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            x["file_name"]: x["sem_seg_file_name"]
            for x in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        self._compute_boundary_iou = True
        if not _CV2_IMPORTED or self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros_like(self._conf_matrix)
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)

            gt[gt == self._ignore_label] = self._num_classes
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))
                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return
            self._conf_matrix = sum(conf_matrix_list)
            self._b_conf_matrix = sum(b_conf_matrix_list)

        acc = np.full(self._num_classes, np.nan)
        iou = np.full(self._num_classes, np.nan)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)

        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.nanmean(acc)
        miou = np.nanmean(iou)
        fiou = np.nansum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        # Dice and F1
        dice = np.full(self._num_classes, np.nan)
        f1 = np.full(self._num_classes, np.nan)
        precision = tp / (tp + (pos_pred - tp) + 1e-6)
        recall = tp / (tp + (pos_gt - tp) + 1e-6)
        dice = 2 * tp / (2 * tp + (pos_pred - tp) + (pos_gt - tp) + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        mean_dice = np.nanmean(dice)
        mean_f1 = np.nanmean(f1)

        res = OrderedDict({
            "mIoU": 100 * miou,
            "fwIoU": 100 * fiou,
            "mACC": 100 * macc,
            "pACC": 100 * pacc,
            "mean_dice": 100 * mean_dice,
            "mean_f1": 100 * mean_f1
        })

        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            res[f"ACC-{name}"] = 100 * acc[i]
            res[f"Dice-{name}"] = 100 * dice[i]
            res[f"F1-{name}"] = 100 * f1[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

        self._logger.info(res)
        return OrderedDict({"sem_seg": res})

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append({"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle})
        return json_list

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)
        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask = cv2.erode(padded_mask, kernel, iterations=dilation)[1:-1, 1:-1]
        return mask - eroded_mask
