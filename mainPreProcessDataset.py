from preProcessFunctions import extract_zip, process_training_set

def main():
    input_root = 'd:/Uni/5oMISIA/TFM/Scripts/'
    output_root = 'd:/Uni/5oMISIA/TFM/Scripts/MSLesSeg-DatasetProcesado'
    
    # Extraer archivos zip si es necesario
    zip_path = 'MSLesSeg-Dataset.zip'
    print(f"Extrayendo {zip_path} a {input_root}...")
    #extract_zip(zip_path, input_root)
    
    input_path_train = 'd:/Uni/5oMISIA/TFM/Scripts/MSLesSeg-Dataset/train'
    output_dir_train = 'd:/Uni/5oMISIA/TFM/Scripts/MSLesSeg-Dataset/output_trainMasks'
    output_dir_trainFLAIR = 'd:/Uni/5oMISIA/TFM/Scripts/MSLesSeg-Dataset/output_trainFLAIR'
    input_path_test = 'd:/Uni/5oMISIA/TFM/Scripts/MSLesSeg-Dataset/test'
    output_dir_test = 'd:/Uni/5oMISIA/TFM/Scripts/MSLesSeg-Dataset/output_testMasks'
    output_dir_testFLAIR = 'd:/Uni/5oMISIA/TFM/Scripts/MSLesSeg-Dataset/output_testFLAIR'

    print(f"Procesando el conjunto de Máscaras de entrenamiento en {input_path_train} y guardando en {output_dir_trainFLAIR}...")
    process_training_set(input_path_train, output_dir_trainFLAIR, scan_type='*FLAIR.nii.gz', rgb=True)
    print(f"Procesando el conjunto de Máscaras de test en {input_path_test} y guardando en {output_dir_test}...")
    process_training_set(input_path_test, output_dir_testFLAIR, scan_type='*FLAIR.nii.gz', rgb=True)

    # Procesar el conjunto de entrenamiento
    #print(f"Procesando el conjunto de entrenamiento en {input_root} y guardando en {output_root}...")
    #process_training_set(input_root, output_root, rgb=False)

if __name__ == "__main__":
    main()  