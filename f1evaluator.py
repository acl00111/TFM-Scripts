import os
import numpy as np
import cv2
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

def evaluate_masksDiceF1(gt_masks, pred_masks):
    assert len(gt_masks) == len(pred_masks), f"Las listas de máscaras de ground truth {len(gt_masks)} y predicciones {len(pred_masks)} deben tener la misma longitud."

    dice_scores = []
    f1_scores = []
    precisions = []
    recalls = []

    skipped = 0

    for gt_bin, pred_bin in zip(gt_masks, pred_masks):
        gt_flat = gt_bin.flatten()
        pred_flat = pred_bin.flatten()

        # Si no hay nada que detectar y no se ha predicho nada → ignorar
        if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
            skipped += 1
            continue

        # Calcular métricas solo cuando hay algo que evaluar
        intersection = np.logical_and(gt_flat, pred_flat).sum()
        total = gt_flat.sum() + pred_flat.sum()

        dice = (2 * intersection) / (total + 1e-6)
        dice_scores.append(dice)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = f1_score(gt_flat, pred_flat, zero_division=0)
            prec = precision_score(gt_flat, pred_flat, zero_division=0)
            rec = recall_score(gt_flat, pred_flat, zero_division=0)

        f1_scores.append(f1)
        precisions.append(prec)
        recalls.append(rec)

    if len(dice_scores) == 0:
        # Evitar división por cero si todas fueron ignoradas
        return OrderedDict({
            "seg_dice": 0.0,
            "seg_f1": 0.0,
            "seg_precision": 0.0,
            "seg_recall": 0.0,
            "skipped_images": skipped,
        })

    return OrderedDict({
        "seg_dice": 100 * np.mean(dice_scores),
        "seg_f1": 100 * np.mean(f1_scores),
        "seg_precision": 100 * np.mean(precisions),
        "seg_recall": 100 * np.mean(recalls),
        "skipped_images": skipped,
    })


def evaluate_binary_masks(pred_dir, gt_dir, threshold=127):
    dice_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    pred_files = sorted(os.listdir(pred_dir))
    for file_name in pred_files:
        pred_path = os.path.join(pred_dir, file_name)
        mask_filename = os.path.basename("_".join(file_name.split("_")[:3]) + "_MASK_" + "_".join(file_name.split("_")[4:]))
        gt_path = os.path.join(gt_dir, mask_filename)

        if not os.path.exists(gt_path):
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            continue

        # Asegurarse de que ambas máscaras tengan el mismo tamaño
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Binarizar
        pred_bin = (pred > threshold).astype(np.uint8).flatten()
        gt_bin = (gt > threshold).astype(np.uint8).flatten()

        if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
            continue

        # Métricas
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total = gt_bin.sum() + pred_bin.sum()
            intersection = np.sum(pred_bin * gt_bin)
            dice = (2. * intersection) / (total + 1e-6)
            f1 = f1_score(gt_bin, pred_bin, zero_division=0)
            precision = precision_score(gt_bin, pred_bin, zero_division=0)
            recall = recall_score(gt_bin, pred_bin, zero_division=0)

        dice_scores.append(dice)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Resultados promedio en formato compatible
    results = OrderedDict({
        "seg_dice": 100 * np.mean(dice_scores),
        "seg_f1": 100 * np.mean(f1_scores),
        "seg_precision": 100 * np.mean(precision_scores),
        "seg_recall": 100 * np.mean(recall_scores),
    })

    return results