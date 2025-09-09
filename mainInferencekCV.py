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

from inference import inference, save_predicted_masks

def inference_kcv(config_dict):
    base_path_dir = '/mnt/Data1/MSLesSeg-Dataset/5kfold_FLAIR' + f"/{config_dict['modalidad']}"
    path_dir_model = "/home/albacano/TFM-Scripts/Detectron2_models"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{config_dict['modelo']}_{config_dict['modalidad']}_{config_dict['maxiter']}_{config_dict['flip']}_{config_dict['batch_size']}_{config_dict['base_lr']}_{config_dict['weight_decay']}_{config_dict['roi_batch_size_per_image']}_{config_dict['roi_positive_fraction']}_{config_dict['rpn_fg_iou_thresh']}_{config_dict['rpn_bg_iou_thresh']}_{config_dict['lr_scheduler']}"

    path_dir_train = base_path_dir + f"/trainImages"
    path_dir_val = base_path_dir + f"/valImages"

    output_dir = os.path.join(path_dir_model, name)
    os.makedirs(output_dir, exist_ok=True)

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
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config_dict['roi_batch_size_per_image']
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = config_dict['roi_positive_fraction']
    cfg.MODEL.RPN.FG_IOU_THRESH = config_dict['rpn_fg_iou_thresh']
    cfg.MODEL.RPN.BG_IOU_THRESH = config_dict['rpn_bg_iou_thresh']
    cfg.SOLVER.LR_SCHEDULER_NAME = config_dict['lr_scheduler']

    # Inferencia y visualización de resultados
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    # Evaluación
    inference(predictor, val_dataset_dicts, val_metadata, f"{base_path_dir}/valMasks", f"{output_dir}/output_kcvimages")

def main():
    config_dict50 = {
        'modalidad': 'FLAIR',
        'modelo': 'mask_rcnn_R_50_FPN_3x.yaml',
        'flip': 'vertical',
        'batch_size': 4,
        'gamma': 0.05,
        'base_lr': 0.001,
        'weight_decay': 0.0001,
        'maxiter': 5000,
        'steps': [2500],
        'roi_batch_size_per_image': 256,
        'roi_positive_fraction': 0.7,
        'rpn_fg_iou_thresh': 0.5,
        'rpn_bg_iou_thresh': 0.5,
        'lr_scheduler': 'WarmupMultiStepLR'
    } 

    config_dict101 = {
        'modalidad': 'FLAIR',
        'modelo': 'mask_rcnn_R_101_FPN_3x.yaml',
        'flip': 'none',
        'batch_size': 4,
        'gamma': 0.05,
        'base_lr': 0.00095935,
        'weight_decay': 0.00010362,
        'maxiter': 5000,
        'steps': [2500],
        'roi_batch_size_per_image': 256,
        'roi_positive_fraction': 0.5,
        'rpn_fg_iou_thresh': 0.7,
        'rpn_bg_iou_thresh': 0.5,
        'lr_scheduler': 'WarmupMultiStepLR'
    } 

    config_dictX101 = {
        'modalidad': 'FLAIR',
        'modelo': 'mask_rcnn_X_101_32x8d_FPN_3x.yaml',
        'flip': 'none',
        'batch_size': 1,
        'gamma': 0.05,
        'base_lr': 0.001,
        'weight_decay': 0.0001,
        'maxiter': 5000,
        'steps': [2500],
        'roi_batch_size_per_image': 256,
        'roi_positive_fraction': 0.5,
        'rpn_fg_iou_thresh': 0.5,
        'rpn_bg_iou_thresh': 0.5,
        'lr_scheduler': 'WarmupMultiStepLR'
    }

    configs = [config_dictX101]  # Aquí puedes añadir más configuraciones si lo deseas
    base_dir_kcv = '/mnt/Data1/MSLesSeg-Dataset/5kfold_FLAIR/'
    modalidad =['2CrossVal'] 

    for conf in configs:
        for fold in modalidad:
            print(f"Procesando fold: {fold}")
            conf['modalidad'] = fold
            print(f"Configuración actualizada con fold: {base_dir_kcv + conf['modalidad']}")
            print(f"Entrenando con configuración: {conf}")
            # Ejecutar el pipeline de entrenamiento
            inference_kcv(conf)


if __name__ == "__main__":
    main()