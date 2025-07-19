import os
import numpy as np
import cv2
from collections import OrderedDict
from sklearn.metrics import f1_score, precision_score, recall_score
from glob import glob

def dice_coefficient(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return 2. * intersection / (y_true.sum() + y_pred.sum() + 1e-6)

def evaluate_masksDiceF1(gt_dir, pred_dir):
    gt_files = sorted(glob(os.path.join(gt_dir, "*.png")))
    pred_files = sorted(glob(os.path.join(pred_dir, "*.png")))

    assert len(gt_files) == len(pred_files), "Mismatch in number of masks"

    dice_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for gt_path, pred_path in zip(gt_files, pred_files):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        gt_bin = (gt > 0).astype(np.uint8).flatten()
        pred_bin = (pred > 0).astype(np.uint8).flatten()

        dice = dice_coefficient(gt_bin, pred_bin)
        f1 = f1_score(gt_bin, pred_bin)
        precision = precision_score(gt_bin, pred_bin)
        recall = recall_score(gt_bin, pred_bin)

        print(f"{os.path.basename(gt_path)} — Dice: {dice:.4f}, F1: {f1:.4f}")

        dice_scores.append(dice)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    results = OrderedDict({
        "external_seg_dice": 100 * np.mean(dice_scores),
        "external_seg_f1": 100 * np.mean(f1_scores),
        "external_seg_precision": 100 * np.mean(precision_scores),
        "external_seg_recall": 100 * np.mean(recall_scores),
    })

    return results

# USO:
# evaluate_folder("path/to/gt", "path/to/pred")
