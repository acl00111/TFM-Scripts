
from trainDetectron import run_training_pipeline
from preProcessFunctions import *
from divisionDataset import dividir_y_mover_val_simple
from processMasks import batch_masks_to_single_json
import os

def main():
    # Definir la configuración del modelo  mask_rcnn_R_50_C4_3x.yaml_RGB_0.00025_2000_vertical_2_[1000]
    
    base_path_dir = 'D:/Uni/5oMISIA/TFM/MSLesSeg-Dataset'
    input_path_train = base_path_dir + '/train'
    output_dir_trainMasks = base_path_dir + '/output_trainMasks'
    output_dir_trainFLAIR = base_path_dir + '/output_trainFLAIR'

    #process_training_set(input_path_train, output_dir_trainFLAIR, scan_type='*FLAIR.nii.gz', rgb=True)
    #process_training_set(input_path_train, output_dir_trainMasks, scan_type='*MASK.nii.gz', rgb=False)

    augmentedFLAIR = base_path_dir + '/augmentedFLAIR'
    augmentedMask = base_path_dir + '/augmentedMask'

    dataAugmentationRotation(output_dir_trainFLAIR, augmentedFLAIR)
    dataAugmentationRotation(output_dir_trainMasks, augmentedMask)

    output_div_trainFLAIR = base_path_dir + '/outputDivided_trainFLAIR-AUG'
    output_div_masktrainFLAIR = base_path_dir + '/output_maskDivided_trainFLAIR-AUG'
    output_div_valFLAIR = base_path_dir + '/outputDivided_valFLAIR-AUG'
    output_div_maskvalFLAIR = base_path_dir + '/output_maskDivided_valFLAIR-AUG'

    print("Dividiendo el conjunto FLAIR")
    dividir_y_mover_val_simple(augmentedFLAIR, augmentedMask,
        output_div_trainFLAIR, output_div_masktrainFLAIR,
        output_div_valFLAIR, output_div_maskvalFLAIR,
        seed=42, ratio_train=0.8
    )

    output_mask_directories = [
        output_div_masktrainFLAIR,
        output_div_maskvalFLAIR
    ]

    all_directories = [
        output_div_trainFLAIR,
        output_div_valFLAIR
    ]

    for dir_mask, dir_train in zip(output_mask_directories, all_directories):
        if not os.path.exists(dir_mask):
            print(f"Directorio {dir_mask} no existe, omitiendo.")
            continue
        json_file_path = os.path.join(dir_train, "annotations.json")
        print(f"Procesando máscaras en {dir_mask} con imágenes en {dir_train} a JSON...")
        batch_masks_to_single_json(dir_mask, json_file_path)

        
    config_dict = {
        'modalidad': 'FLAIR-AUG',
        'modelo': 'mask_rcnn_R_101_FPN_3x.yaml',
        'flip': 'vertical',
        'batch_size': 4,
        'gamma': 0.05,
        'base_lr': 0.001,
        'weight_decay': 0.0001,
        'maxiter': 5000,
        'steps': [2500]
    } 

    # Ejecutar el pipeline de entrenamiento
    run_training_pipeline(config_dict)


if __name__ == "__main__":
    main()