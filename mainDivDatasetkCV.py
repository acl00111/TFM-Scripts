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

    input_dir_trainRGB = base_path_dir + '/output_trainRGB'
    output_dir_trainRGB = base_path_dir + '/outputDivided_trainRGB'
    output_dir_masktrainRGB = base_path_dir + '/output_maskDivided_trainRGB'
    output_dir_valRGB = base_path_dir + '/outputDivided_valRGB'
    output_dir_maskvalRGB = base_path_dir + '/output_maskDivided_valRGB'


    # Dividir en entrenamiento y validación
    print("Dividiendo el conjunto FLAIR")
   # dividir_y_mover_val_simple(input_dir_trainFLAIR, input_dir_trainMasks,
  #      output_dir_trainFLAIR, output_dir_masktrainFLAIR,
  #      output_dir_valFLAIR, output_dir_maskvalFLAIR,
 #       seed=42, ratio_train=0.8
  #  )


    print("Dividiendo el conjunto RGB")
   # dividir_y_mover_val_simple(input_dir_trainRGB, input_dir_trainMasks,
  #      output_dir_trainRGB, output_dir_masktrainRGB,
  #      output_dir_valRGB, output_dir_maskvalRGB,
  #      seed=42, ratio_train=0.8
 #  )

    k = 5
    print(f"Dividiendo el conjunto FLAIR en {k}-folds")
    k_fold_split_images_masks(
        output_dir_trainFLAIR, output_dir_masktrainFLAIR, output_dir=base_path_dir + '/5kfold_FLAIR',
        k=k, seed=42
    )

    output_mask_directories = [
        base_path_dir + '/5kfold_FLAIR/1CrossVal/trainMasks',
        base_path_dir + '/5kfold_FLAIR/1CrossVal/valMasks', 
        base_path_dir + '/5kfold_FLAIR/2CrossVal/trainMasks',
        base_path_dir + '/5kfold_FLAIR/2CrossVal/valMasks',
        base_path_dir + '/5kfold_FLAIR/3CrossVal/trainMasks',
        base_path_dir + '/5kfold_FLAIR/3CrossVal/valMasks',
        base_path_dir + '/5kfold_FLAIR/4CrossVal/trainMasks',
        base_path_dir + '/5kfold_FLAIR/4CrossVal/valMasks',
        base_path_dir + '/5kfold_FLAIR/5CrossVal/trainMasks',
        base_path_dir + '/5kfold_FLAIR/5CrossVal/valMasks',
        base_path_dir + '/5kfold_RGB/1CrossVal/trainMasks',
        base_path_dir + '/5kfold_RGB/1CrossVal/valMasks', 
        base_path_dir + '/5kfold_RGB/2CrossVal/trainMasks',
        base_path_dir + '/5kfold_RGB/2CrossVal/valMasks',
        base_path_dir + '/5kfold_RGB/3CrossVal/trainMasks',
        base_path_dir + '/5kfold_RGB/3CrossVal/valMasks',
        base_path_dir + '/5kfold_RGB/4CrossVal/trainMasks',
        base_path_dir + '/5kfold_RGB/4CrossVal/valMasks',
        base_path_dir + '/5kfold_RGB/5CrossVal/trainMasks',
        base_path_dir + '/5kfold_RGB/5CrossVal/valMasks'
    ]

    all_directories = [
        base_path_dir + '/5kfold_FLAIR/1CrossVal/trainImages',
        base_path_dir + '/5kfold_FLAIR/1CrossVal/valImages', 
        base_path_dir + '/5kfold_FLAIR/2CrossVal/trainImages',
        base_path_dir + '/5kfold_FLAIR/2CrossVal/valImages',
        base_path_dir + '/5kfold_FLAIR/3CrossVal/trainImages',
        base_path_dir + '/5kfold_FLAIR/3CrossVal/valImages',
        base_path_dir + '/5kfold_FLAIR/4CrossVal/trainImages',
        base_path_dir + '/5kfold_FLAIR/4CrossVal/valImages',
        base_path_dir + '/5kfold_FLAIR/5CrossVal/trainImages',
        base_path_dir + '/5kfold_FLAIR/5CrossVal/valImages',
        base_path_dir + '/5kfold_RGB/1CrossVal/trainImages',
        base_path_dir + '/5kfold_RGB/1CrossVal/valImages', 
        base_path_dir + '/5kfold_RGB/2CrossVal/trainImages',
        base_path_dir + '/5kfold_RGB/2CrossVal/valImages',
        base_path_dir + '/5kfold_RGB/3CrossVal/trainImages',
        base_path_dir + '/5kfold_RGB/3CrossVal/valImages',
        base_path_dir + '/5kfold_RGB/4CrossVal/trainImages',
        base_path_dir + '/5kfold_RGB/4CrossVal/valImages',
        base_path_dir + '/5kfold_RGB/5CrossVal/trainImages',
        base_path_dir + '/5kfold_RGB/5CrossVal/valImages'
    ]

    # Por cada carpeta entrenamiento/validación, procesar máscaras a JSON
    for dir_mask, dir_train in zip(output_mask_directories, all_directories):
        if not os.path.exists(dir_mask):
            print(f"Directorio {dir_mask} no existe, omitiendo.")
            continue
        json_file_path = os.path.join(dir_train, "annotations.json")
        print(f"Procesando máscaras en {dir_mask} con imágenes en {dir_train} a JSON...")
        batch_masks_to_single_json(dir_mask, json_file_path)

    
    

if __name__ == "__main__":
    main()


