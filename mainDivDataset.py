from divisionDataset import dividir_y_mover_val_simple, k_fold_split_images_masks
from processMasks import batch_masks_to_single_json
import os


def main():
    base_path_dir = '/mnt/Data1/MSLesSeg-Dataset'

    input_dir_trainMasks = base_path_dir + '/output_trainMasks'
    input_dir_trainFLAIR = base_path_dir + '/output_trainFLAIR'
    output_dir_trainFLAIR = base_path_dir + '/outputDivided_trainFLAIR'
    output_dir_masktrainFLAIR = base_path_dir + '/output_maskDivided_trainFLAIR'
    output_dir_valFLAIR = base_path_dir + '/outputDivided_valFLAIR'
    output_dir_maskvalFLAIR = base_path_dir + '/output_maskDivided_valFLAIR'


    input_dir_trainT1 = base_path_dir + '/output_trainT1'
    output_dir_trainT1 = base_path_dir + '/outputDivided_trainT1'
    output_dir_masktrainT1 = base_path_dir + '/output_maskDivided_trainT1'
    output_dir_valT1 = base_path_dir + '/outputDivided_valT1'
    output_dir_maskvalT1 = base_path_dir + '/output_maskDivided_valT1'

    input_dir_trainT2 = base_path_dir + '/output_trainT2'
    output_dir_trainT2 = base_path_dir + '/outputDivided_trainT2'
    output_dir_masktrainT2 = base_path_dir + '/output_maskDivided_trainT2'
    output_dir_valT2 = base_path_dir + '/outputDivided_valT2'
    output_dir_maskvalT2 = base_path_dir + '/output_maskDivided_valT2'

    input_dir_trainRGB = base_path_dir + '/output_trainRGB'
    output_dir_trainRGB = base_path_dir + '/outputDivided_trainRGB'
    output_dir_masktrainRGB = base_path_dir + '/output_maskDivided_trainRGB'
    output_dir_valRGB = base_path_dir + '/outputDivided_valRGB'
    output_dir_maskvalRGB = base_path_dir + '/output_maskDivided_valRGB'


    # Dividir en entrenamiento y validación
    print("Dividiendo el conjunto FLAIR")
    dividir_y_mover_val_simple(input_dir_trainFLAIR, input_dir_trainMasks,
        output_dir_trainFLAIR, output_dir_masktrainFLAIR,
        output_dir_valFLAIR, output_dir_maskvalFLAIR,
        seed=42, ratio_train=0.8
    )

    print("Dividiendo el conjunto T1")
    dividir_y_mover_val_simple(input_dir_trainT1, input_dir_trainMasks,
        output_dir_trainT1, output_dir_masktrainT1,
        output_dir_valT1, output_dir_maskvalT1,
        seed=42, ratio_train=0.8
    )

    print("Dividiendo el conjunto T2")
    dividir_y_mover_val_simple(input_dir_trainT2, input_dir_trainMasks,
        output_dir_trainT2, output_dir_masktrainT2,
        output_dir_valT2, output_dir_maskvalT2,
        seed=42, ratio_train=0.8
    )

    print("Dividiendo el conjunto RGB")
    dividir_y_mover_val_simple(input_dir_trainRGB, input_dir_trainMasks,
        output_dir_trainRGB, output_dir_masktrainRGB,
        output_dir_valRGB, output_dir_maskvalRGB,
        seed=42, ratio_train=0.8
    )

    

    output_mask_directories = [
        output_dir_masktrainFLAIR,
        output_dir_maskvalFLAIR,
        output_dir_masktrainT1,
        output_dir_maskvalT1,
        output_dir_masktrainT2,
        output_dir_maskvalT2,
        output_dir_masktrainRGB,
        output_dir_maskvalRGB
    ]

    all_directories = [
        output_dir_trainFLAIR,
        output_dir_valFLAIR,
        output_dir_trainT1,
        output_dir_valT1,
        output_dir_trainT2,
        output_dir_valT2,
        output_dir_trainRGB,
        output_dir_valRGB
    ]

    # Por cada carpeta entrenamiento/validación, procesar máscaras a JSON
    for dir_mask, dir_train in zip(output_mask_directories, all_directories):
        if not os.path.exists(dir_mask):
            print(f"Directorio {dir_mask} no existe, omitiendo.")
            continue
        json_file_path = os.path.join(dir_train, "annotations.json")
        print(f"Procesando máscaras en {dir_mask} con imágenes en {dir_train} a JSON...")
        batch_masks_to_single_json(dir_mask, json_file_path)

    
    k = 5
    print(f"Dividiendo el conjunto RGB en {k}-folds")
    k_fold_split_images_masks(
        output_dir_trainRGB, output_dir_masktrainRGB, output_dir=base_path_dir + '/5kfold_RGB',
        k=k, seed=42
    )

if __name__ == "__main__":
    main()


