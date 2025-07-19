from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import warnings
from collections import OrderedDict

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
