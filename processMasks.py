import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import uuid

def generate_annotation(mask, image_id, annotation_id, category_id=1):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotations = []

    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue  # Ignora contornos pequeños
        segmentation = contour.flatten().tolist()
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [segmentation],
            "bbox": [x, y, w, h],
            "area": area,
            "iscrowd": 0
        }
        annotations.append(annotation)
        annotation_id += 1

    return annotations, annotation_id

def batch_masks_to_single_json(mask_dir, output_json_path):
    mask_dir = Path(mask_dir)
    mask_files = sorted(mask_dir.glob("*.png"))

    coco_output = {
        "info": {
        "description": "Dataset for Detectron2 training",
        "url": "http://example.com",
        "version": "1.0",
        "year": 2025,
        "contributor": "Alba Cano Lara",
        "date_created": "2025-07-14"
    },
    "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "MS lesion"}
        ]
    }

    image_id = 1
    annotation_id = 1

    for mask_path in tqdm(mask_files, desc="Procesando máscaras"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape

        # Agregar entrada de imagen
        coco_output["images"].append({
            "id": image_id,
            "file_name": mask_path.name,
            "width": width,
            "height": height
        })

        # Generar anotaciones para esta máscara
        anns, annotation_id = generate_annotation(mask, image_id, annotation_id)
        coco_output["annotations"].extend(anns)

        image_id += 1

    # Guardar JSON
    with open(output_json_path, 'w') as f:
        json.dump(coco_output, f, indent=2)

    print(f"Anotaciones guardadas en: {output_json_path}")