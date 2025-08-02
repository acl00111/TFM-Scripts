import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
import yaml
import itertools
import pandas as pd
import pathlib

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from matplotlib import pyplot as plt
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator
from f1evaluator import evaluate_binary_masks
from datetime import datetime

from inference import inference, save_predicted_masks  # Importamos la función de inferencia desde el archivo inference.py


# Guardamos la configuración en un archivo .yaml
def save_config(cfg, path_dir):
    print("Guardando la configuración del modelo en un archivo YAML...")
    config_yaml_path = f"{path_dir}/config.yaml"
    with open(config_yaml_path, 'w') as file:
        yaml.dump(cfg, file)


def run_training_pipeline(config_dict):
    """
    Ejecuta el entrenamiento y evaluación para una configuración y fold dado.
    `config_dict` debe incluir: modalidad, modelo, flip, batch_size, base_lr, maxiter_steps, etc.
    """
    # Construir nombre único
    base_path_dir = '/mnt/Data1/MSLesSeg-Dataset'
    path_dir_model = "/home/albacano/TFM-Scripts/Detectron2_models"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{config_dict['modelo']}_{config_dict['modalidad']}_{config_dict['maxiter']}_{config_dict['flip']}_{config_dict['batch_size']}_{timestamp}"

    path_dir_train = base_path_dir + f"/outputDivided_train{config_dict['modalidad']}"
    path_dir_val = base_path_dir + f"/outputDivided_val{config_dict['modalidad']}"

    output_dir = os.path.join(path_dir_model, name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup dataset
    train_dataset_name = f"train_{name}"
    val_dataset_name = f"val_{name}"
    train_json = f"{path_dir_train}/annotations.json"
    val_json = f"{path_dir_val}/annotations.json"
    train_img_dir = f"{path_dir_train}"
    val_img_dir = f"{path_dir_val}"

    register_coco_instances(train_dataset_name, {}, train_json, train_img_dir)
    register_coco_instances(val_dataset_name, {}, val_json, val_img_dir)
    train_metadata = MetadataCatalog.get(train_dataset_name)
    train_dataset_dicts = DatasetCatalog.get(train_dataset_name)
    val_metadata = MetadataCatalog.get(val_dataset_name)
    val_dataset_dicts = DatasetCatalog.get(val_dataset_name)

    # Config
    cfg = get_cfg()
    cfg.OUTPUT_DIR = output_dir
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{config_dict['modelo']}"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{config_dict['modelo']}")
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.BASE_LR = config_dict['base_lr']
    cfg.SOLVER.MAX_ITER = config_dict['maxiter']
    cfg.SOLVER.IMS_PER_BATCH = config_dict['batch_size']
    cfg.SOLVER.STEPS = config_dict['steps']
    cfg.INPUT.MIN_SIZE_TRAIN = 364 # Redimensionamiento de las imágenes de entrenamiento
    cfg.INPUT.MAX_SIZE_TRAIN = 364    
    cfg.INPUT.MIN_SIZE_TEST  = 364
    cfg.INPUT.MAX_SIZE_TEST  = 364
    cfg.SOLVER.GAMMA = config_dict['gamma']
    cfg.SOLVER.WEIGHT_DECAY = config_dict['weight_decay']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice" 
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT.RANDOM_FLIP = config_dict['flip']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    # Entrenamiento
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    save_config(cfg, output_dir)  # Guardamos la configuración del modelo en un archivo YAML

    # Inferencia y visualización de resultados
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    # Evaluación
    inference(predictor, val_dataset_dicts, val_metadata, f"{base_path_dir}/output_maskDivided_val{config_dict['modalidad']}", f"{output_dir}/output_images")
    val_loader = build_detection_test_loader(cfg, val_dataset_name)
    coco_eval = COCOEvaluator(val_dataset_name, output_dir=os.path.join(output_dir, "eval"))
    results = inference_on_dataset(predictor.model, val_loader, coco_eval)

    # Máscaras y métricas externas
    save_predicted_masks(predictor, DatasetCatalog.get(val_dataset_name), os.path.join(output_dir, "predicted_masks"))
    f1_metrics = evaluate_binary_masks(
        f"{base_path_dir}/output_maskDivided_val{config_dict['modalidad']}", 
        f"{output_dir}/predicted_masks"
    )
    results.update(f1_metrics)

    # Guardar resultados
    results["configuracion"] = name
    df = pd.json_normalize(results, sep='_')
    results_path = pathlib.Path(f"{path_dir_model}/resultados_finalesBig.csv")
    df.to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
    
    torch.cuda.empty_cache()  # Limpiar caché de CUDA

    return results["seg_f1"]
