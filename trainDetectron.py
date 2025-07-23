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

from inference import inference, save_predicted_masks  # Importamos la función de inferencia desde el archivo inference.py





def run_training_pipeline(config_dict):
    """
    Ejecuta el entrenamiento y evaluación para una configuración y fold dado.
    `config_dict` debe incluir: modalidad, modelo, flip, batch_size, base_lr, maxiter_steps, etc.
    """
    # Construir nombre único
    base_path_dir = '/mnt/Data1/MSLesSeg-Dataset'
    path_dir_model = "/home/albacano/TFM-Scripts/Detectron2_models"
    name = f"{config_dict['modelo']}_{config_dict['modalidad']}_{config_dict['base_lr']}_{config_dict['maxiter_steps']}"

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

    # Config
    cfg = get_cfg()
    cfg.OUTPUT_DIR = output_dir
    cfg.merge_from_file(model_zoo.get_config_file(config_dict['modelo']))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.SOLVER.BASE_LR = config_dict['base_lr']
    cfg.SOLVER.MAX_ITER = config_dict['maxiter_steps']
    cfg.SOLVER.IMS_PER_BATCH = config_dict['batch_size']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.INPUT.RANDOM_FLIP = config_dict['flip']
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_dict['modelo'])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    # Entrenamiento
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluación
    predictor = DefaultPredictor(cfg)
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
    results_path = pathlib.Path(f"{path_dir_model}/resultados_finales.csv")
    df.to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
