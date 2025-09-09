import cv2
import os
import random
import numpy as np
from detectron2.utils.visualizer import Visualizer, ColorMode
from matplotlib import pyplot as plt
from tqdm import tqdm
from detectron2.structures import Instances

def inference(predictor, val_dataset_dicts, val_metadata, val_mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for d in val_dataset_dicts:  # prueba con una imagen
        file_name = d["file_name"]
        im = cv2.imread(file_name)

        print("Imagen de validación:", file_name)

        # Inferencia con el modelo
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=val_metadata,
                    scale=2,
                    instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_img = out.get_image()[:, :, ::-1]

        # Cargar máscara real
        image_filename = os.path.basename("_".join(file_name.split("_")[:3]) + "_MASK_" + "_".join(file_name.split("_")[4:]))
        mask_path = os.path.join(val_mask_dir, image_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"No se encontró la máscara para {image_filename}")
            continue

        # Convertir la imagen a blanco y negro (escala de grises)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray_3ch = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        # Crear superposición en rojo (color de lesión) sobre la imagen en escala de grises
        mask_colored = np.zeros_like(im_gray_3ch)
        mask_colored[:, :, 2] = 255  # rojo puro

        alpha = 0.4
        overlay = np.where(mask[..., None] == 255,
                        (alpha * mask_colored + (1 - alpha) * im_gray_3ch).astype(np.uint8),
                        im_gray_3ch)

        # Mostrar lado a lado: predicción y GT
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        axs[0].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Predicción del modelo")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Imagen + Máscara real")
        axs[1].axis("off")

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"comparison_{os.path.basename(file_name)}.png")
        plt.savefig(output_path)
        print(f"Imagen guardada en: {output_path}")


def save_predicted_masks(predictor, val_dataset_dicts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for d in tqdm(val_dataset_dicts, desc="Generando máscaras predichas"):
        file_name = d["file_name"]
        image_id = os.path.splitext(os.path.basename(file_name))[0]

        im = cv2.imread(file_name)
        outputs = predictor(im)
        instances: Instances = outputs["instances"].to("cpu")

        # Crear máscara combinada binaria
        combined_mask = np.zeros(im.shape[:2], dtype=np.uint8)
        if instances.has("pred_masks"):
            for mask in instances.pred_masks:
                combined_mask = np.logical_or(combined_mask, mask.numpy())

        # Guardar máscara binaria en escala de grises (0 o 255)
        combined_mask = (combined_mask.astype(np.uint8)) * 255
        out_path = os.path.join(output_dir, f"{image_id}.png")
        cv2.imwrite(out_path, combined_mask)
