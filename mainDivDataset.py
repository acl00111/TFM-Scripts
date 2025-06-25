from divisionDataset import dividir_y_mover_val_simple, k_fold_split_images_masks
from processMasks import batch_masks_to_single_json
import os


def main():
    train_images_dir = "data/train/images"
    train_masks_dir = "data/train/masks"
    val_images_dir = "data/val/images"
    val_masks_dir = "data/val/masks"

    # Dividir en entrenamiento y validación
    dividir_y_mover_val_simple(
        train_images_dir, train_masks_dir,
        val_images_dir, val_masks_dir,
        seed=42, ratio_train=0.8
    )

    # Realizar k-fold si es necesario
    k = 5
    k_fold_split_images_masks(
        train_images_dir, train_masks_dir,
        k=k, seed=42
    )

    # Por cada carpeta entrenamiento/validación, procesar máscaras a JSON
    batch_masks_to_single_json(train_masks_dir, "data/train/annotations.json")

if __name__ == "__main__":
    main()


