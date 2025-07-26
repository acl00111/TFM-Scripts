
from trainDetectron import run_training_pipeline

def main():
    # Definir la configuración del modelo  mask_rcnn_R_50_C4_3x.yaml_RGB_0.00025_2000_vertical_2_[1000]
    config_dict = {
        'modalidad': 'RGB',
        'modelo': 'mask_rcnn_R_50_C4_3x.yaml',
        'flip': 'vertical',
        'batch_size': 2,
        'gamma': 0.05,
        'base_lr': 0.001,
        'weight_decay': 0.0001,
        'maxiter': 2000,
        'steps': [1000]
    } 

    # Ejecutar el pipeline de entrenamiento
    run_training_pipeline(config_dict)

if __name__ == "__main__":
    main()