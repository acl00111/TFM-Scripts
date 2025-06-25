import random
import shutil
from pathlib import Path
from math import ceil
from tqdm import tqdm
from math import floor

def dividir_y_mover_val_simple(
    train_images_dir, train_masks_dir,
    val_images_dir, val_masks_dir,
    seed=42, ratio_train=0.8
):
    train_images_dir = Path(train_images_dir)
    train_masks_dir = Path(train_masks_dir)
    val_images_dir = Path(val_images_dir)
    val_masks_dir = Path(val_masks_dir)

    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_masks_dir.mkdir(parents=True, exist_ok=True)

    # Obtener todos los IDs de escaneos únicos, por ejemplo: P12_T2
    all_scan_ids = set(f.name.split('_slice')[0] for f in train_images_dir.glob("*.png"))
    all_scan_ids = sorted(all_scan_ids)

    total = len(all_scan_ids)
    num_train = floor(total * ratio_train)

    random.seed(seed)
    random.shuffle(all_scan_ids)

    train_scans = all_scan_ids[:num_train]
    val_scans = all_scan_ids[num_train:]

    print(f"Total escaneos únicos: {total}")
    print(f"Escaneos de entrenamiento: {len(train_scans)}")
    print(f"Escaneos de validación: {len(val_scans)}")

    # Mover imágenes y máscaras para validación
    total_imgs, total_masks = 0, 0

    for scan_id in val_scans:
        imgs = list(train_images_dir.glob(f"{scan_id}_slice_*.png"))
        masks = list(train_masks_dir.glob(f"{scan_id}_slice_*.png"))

        for img in imgs:
            shutil.move(str(img), val_images_dir / img.name)
        for mask in masks:
            shutil.move(str(mask), val_masks_dir / mask.name)

        total_imgs += len(imgs)
        total_masks += len(masks)
        print(f"{scan_id}: {len(imgs)} imágenes, {len(masks)} máscaras movidas.")

    print(f"Total imágenes movidas: {total_imgs}")
    print(f"Total máscaras movidas: {total_masks}")

    return train_scans, val_scans


def k_fold_split_images_masks(
    images_dir, masks_dir, output_dir,
    k=5, seed=42, move_files=False
):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extraer escaneos únicos: P1_T1, P12_T2, etc.
    all_scan_ids = sorted(set(f.name.split('_slice')[0] for f in images_dir.glob("*.png")))
    random.seed(seed)
    random.shuffle(all_scan_ids)

    # Dividir escaneos en K partes
    fold_size = ceil(len(all_scan_ids) / k)
    folds = [all_scan_ids[i * fold_size: (i + 1) * fold_size] for i in range(k)]

    print(f"Total escaneos únicos: {len(all_scan_ids)}")
    print(f"Dividido en {k} folds, de tamaño aproximado {fold_size} escaneos por fold.")

    # Elegir método de transferencia
    file_op = shutil.move if move_files else shutil.copy

    for fold_idx in range(k):
        val_scans = set(folds[fold_idx])
        train_scans = set(all_scan_ids) - val_scans

        print(f"Carpeta {fold_idx + 1}")
        print(f"Escaneos de validación ({len(val_scans)}): {sorted(val_scans)}")

        fold_path = output_dir / f"{fold_idx+1}CrossVal"
        train_img_dir = fold_path / "trainImages"
        val_img_dir = fold_path / "valImages"
        train_mask_dir = fold_path / "trainMasks"
        val_mask_dir = fold_path / "valMasks"

        # Crear carpetas
        for d in [train_img_dir, val_img_dir, train_mask_dir, val_mask_dir]:
            d.mkdir(parents=True, exist_ok=True)

        def transfer_files(scan_ids, src_dir, dst_dir, desc=""):
            matched_files = []
            for scan_id in scan_ids:
                matched_files.extend(src_dir.glob(f"{scan_id}_slice_*.png"))
            for f in tqdm(matched_files, desc=desc, leave=False):
                file_op(f, dst_dir / f.name)
            return len(matched_files)

        train_imgs = transfer_files(train_scans, images_dir, train_img_dir, desc="Copiando imágenes de entrenamiento")
        train_masks = transfer_files(train_scans, masks_dir, train_mask_dir, desc="Copiando máscaras de entrenamiento")
        val_imgs = transfer_files(val_scans, images_dir, val_img_dir, desc="Copiando imágenes de validación")
        val_masks = transfer_files(val_scans, masks_dir, val_mask_dir, desc="Copiando máscaras de validación")

        print(f"Train: {len(train_scans)} escaneos -> {train_imgs} imágenes, {train_masks} máscaras")
        print(f"Val: {len(val_scans)} escaneos -> {val_imgs} imágenes, {val_masks} máscaras")

    print("K-Fold división completada.")