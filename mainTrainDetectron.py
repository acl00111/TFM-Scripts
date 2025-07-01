import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
import yaml

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

from inference import inference  # Importamos la función de inferencia desde el archivo inference.py

base_path_dir = '/mnt/Data1/MSLesSeg-Dataset'
path_dir_model = "/home/albacano/TFM-Scripts/Detectron2_models"
path_dir_train = base_path_dir + "/outputDivided_trainFLAIR"
path_dir_val = base_path_dir + "/outputDivided_valFLAIR"

def setup_datasets(nameTrainDataset, nameValDataset):
    # Registra el dataset de entrenamiento
    print("Registrando los datasets de entrenamiento y validación...")
    if not os.path.exists(f"{path_dir_train}/annotations.json"):
        raise FileNotFoundError(f"El archivo de anotaciones de entrenamiento no se encuentra en {path_dir_train}/annotations.json")
    if not os.path.exists(f"{path_dir_val}/annotations.json"):
        raise FileNotFoundError(f"El archivo de anotaciones de validación no se encuentra en {path_dir_val}/annotations.json")
    register_coco_instances(nameTrainDataset, {}, f"{path_dir_train}/annotations.json", f"{path_dir_train}")
    # Registra el dataset de validación
    register_coco_instances(nameValDataset, {}, f"{path_dir_val}/annotations.json", f"{path_dir_val}")
    train_metadata = MetadataCatalog.get(nameTrainDataset)
    train_dataset_dicts = DatasetCatalog.get(nameTrainDataset)
    val_metadata = MetadataCatalog.get(nameValDataset)
    val_dataset_dicts = DatasetCatalog.get(nameValDataset)
    return train_dataset_dicts, val_dataset_dicts, train_metadata, val_metadata

def buildConfig():
    print("Configurando el modelo Detectron2 para entrenamiento...")
    cfg = get_cfg()
    cfg.OUTPUT_DIR = f"{path_dir_model}/2000epochsFLAIR"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Descarga del modelo troncal Mask R-CNN con ResNet50 y FPN
    cfg.SOLVER.IMS_PER_BATCH = 2  # El batch es pequeño dada la dimensión de nuestro conjunto de datos
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 2000    # 6000 iteraciones han sido las óptimas para el proyecto
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Tenemos tan solo la clase Lesion, no se tiene en cuenta el fondo de la imagen
    
    return cfg


# Guardamos la configuración en un archivo .yaml
def save_config(cfg, path_dir):
    print("Guardando la configuración del modelo en un archivo YAML...")
    config_yaml_path = f"{path_dir}/2000epochsFLAIR/config.yaml"
    with open(config_yaml_path, 'w') as file:
        yaml.dump(cfg, file)


def main():
    setup_logger()
    if not torch.cuda.is_available():
        print("CUDA (GPU) no está disponible. Por favor, verifica tu instalación.")
    else:
        print(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        cfg = buildConfig()  # Se construye la configuración del modelo
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg) # Se crea un objeto Trainer con la configuración
        trainer.resume_or_load(resume=False) # cambiar a True para utilizar el último checkpoint y reanudar el entrenamiento
        train_dataset_dicts, val_dataset_dicts, train_metadata, val_metadata = setup_datasets("my_dataset_train", "my_dataset_val")  # Se configuran los datasets de entrenamiento y validación
        trainer.train()  # Se inicia el entrenamiento del modelo


        # Inferencia y visualización de resultados
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # ponemos un umbral para el test, puede modificarse
        predictor = DefaultPredictor(cfg)

        inference(predictor, val_dataset_dicts, val_metadata, f"{base_path_dir}/output_maskDivided_valFLAIR")

        # evaluamos las métricas del modelo con COCOEvaluator
        evaluator = COCOEvaluator("my_dataset_val", output_dir=f"{path_dir_model}/evaluacion/2000epochs")
        val_loader = build_detection_test_loader(cfg, "my_dataset_val")
        print(inference_on_dataset(predictor.model, val_loader, evaluator))

        

if __name__ == "__main__":
    main()