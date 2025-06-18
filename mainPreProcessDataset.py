from preProcessFunctions import extract_zip, process_training_set, process_dataset_new
import torch

def main():
    base_path_dir = '/mnt/Data1/MSLesSeg-Dataset'
    input_root = '/mnt/Data1/MSLesSeg-Dataset/'
    output_root = '/mnt/Data1/MSLesSeg-DatasetProcesado'
    
    # Extraer archivos zip si es necesario
    zip_path = '/mnt/Data1/MSLesSeg-Dataset.zip'
   # print(f"Extrayendo {zip_path} a {input_root}...")
   # extract_zip(zip_path, input_root)
    
    input_path_train = base_path_dir + '/train'
    output_dir_trainMasks = base_path_dir + '/output_trainMasks'
    output_dir_trainFLAIR = base_path_dir + '/output_trainFLAIR'
    output_dir_trainT1 = base_path_dir + '/output_trainT1'
    output_dir_trainT2 = base_path_dir + '/output_trainT2'

    input_path_test = base_path_dir + '/test'
    output_dir_testFLAIR = base_path_dir + '/output_testFLAIR'
    output_dir_testT1 = base_path_dir + '/output_testT1'
    output_dir_testT2 = base_path_dir + '/output_testT2'

    output_dir_trainRGB = base_path_dir + '/output_trainRGB'
    output_dir_testRGB = base_path_dir + '/output_testRGB'

    print(f"Procesando el conjunto de entrenamiento FLAIR en {input_path_train} y guardando en {output_dir_trainFLAIR}...")
   # process_training_set(input_path_train, output_dir_trainFLAIR, scan_type='*FLAIR.nii.gz', rgb=True)

    print(f"Procesando el conjunto de entrenamiento T1 en {input_path_train} y guardando en {output_dir_trainT1}...")
   # process_training_set(input_path_train, output_dir_trainT1, scan_type='*T1.nii.gz', rgb=True)

    print(f"Procesando el conjunto de entrenamiento T2 en {input_path_train} y guardando en {output_dir_trainT2}...")
   # process_training_set(input_path_train, output_dir_trainT2, scan_type='*T2.nii.gz', rgb=True)

    print(f"Procesando el conjunto de test FLAIR en {input_path_test} y guardando en {output_dir_testFLAIR}...")
   # process_training_set(input_path_test, output_dir_testFLAIR, scan_type='*FLAIR.nii.gz', rgb=True)

    print(f"Procesando el conjunto de test T1 en {input_path_test} y guardando en {output_dir_testT1}...")
   # process_training_set(input_path_test, output_dir_testT1, scan_type='*T1.nii.gz', rgb=True)

    print(f"Procesando el conjunto de test T2 en {input_path_test} y guardando en {output_dir_testT2}...")
   # process_training_set(input_path_test, output_dir_testT2, scan_type='*T2.nii.gz', rgb=True)


    # Procesar el conjunto de entrenamiento
    print(f"Procesando el conjunto de Máscaras entrenamiento en {input_path_train} y guardando en {output_dir_trainMasks}...")
   # process_training_set(input_path_train, output_dir_trainMasks, scan_type='*MASK.nii.gz', rgb=False)

   # Procesar el conjunto de entrenamiento RGB (FLAIR, T1, T2)
    print(f"Procesando el conjunto de entrenamiento FLAIR, T1, T2 en {input_path_train} y guardando en {output_dir_trainRGB}...")
    process_dataset_new(output_dir_trainFLAIR, output_dir_trainT1, output_dir_trainT2, output_dir_trainRGB)
    
    # Procesar el conjunto de test RGB (FLAIR, T1, T2)
    print(f"Procesando el conjunto de test FLAIR, T1, T2 en {input_path_test} y guardando en {output_dir_testRGB}...")
    process_dataset_new(output_dir_testFLAIR, output_dir_testT1, output_dir_testT2, output_dir_testRGB)

if __name__ == "__main__":
    main()  