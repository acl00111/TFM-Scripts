import torch
from torch.utils.data import Subset
import numpy as np
import os
from zipfile import ZipFile
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from zipfile import ZipFile
import nibabel as nib
import numpy as np
from pathlib import Path
from skimage import io
from tqdm import tqdm
import cv2

def extract_zip(zip_path, extract_to):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def to_uint8(data):
    data -= data.min()
    ptp = np.ptp(data)
    if ptp == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    data /= ptp
    data *= 255
    return data.astype(np.uint8)

def nii_to_png(input_path, output_dir, rgb=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = nib.load(input_path).get_fdata()
    num_slices, width, height = data.shape
    nombre_archivo = input_path.stem
    nombre_base = nombre_archivo.replace('.nii', '')
   # print(nombre_base)
   # nombre_base = nombre_archivo.replace('_FLAIR.nii', '')
   # nombre_base = nombre_archivo.replace('_T1.nii', '')
   # nombre_base = nombre_archivo.replace('_T2.nii', '')
    #print(nombre_base)
    for slice in range(num_slices):
        slice_data = data[slice, ...]
        slice_data = to_uint8(slice_data)
        if rgb:
            slice_data = np.stack(3 * [slice_data], axis=2)
        output_path = output_dir / f'{nombre_base}_slice_{slice}.png'
        #print(output_path)
        io.imsave(output_path, slice_data)

def process_training_set(input_root, output_root, scan_type='*FLAIR.nii.gz', rgb=False):
    input_root = Path(input_root)
    output_root = Path(output_root)

   # nii_files = list(input_root.rglob('*MASK.nii.gz'))
    nii_files = list(input_root.rglob(scan_type))

    print(f"Se encontraron {len(nii_files)} archivos .nii.gz en {input_root}")

    for nii_file in tqdm(nii_files, desc='Procesando archivos'):
        # Ruta relativa desde input_root
        relative_path = nii_file.relative_to(input_root)
        # Sacar nombre de archivo sin extensión .nii.gz
        nii_name = nii_file.stem
        # Crear carpeta de salida para este archivo
        #output_dir = output_root / relative_path.parent / nii_name
        #output_dir.mkdir(parents=True, exist_ok=True)
        # Procesar el archivo
        nii_to_png(nii_file, output_root, rgb=rgb)

def combine_modalities(flair_dir, t1_dir, t2_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    flair_slices = sorted(flair_dir.glob("*.png"))
    for flair_path in flair_slices:
        fname = flair_path.name
        t1_path = t1_dir / fname
        t2_path = t2_dir / fname

        if not (t1_path.exists() and t2_path.exists()):
            print(f"Slice {fname} faltante en alguna modalidad, se omite.")
            continue

        flair = cv2.imread(str(flair_path), cv2.IMREAD_GRAYSCALE)
        t1 = cv2.imread(str(t1_path), cv2.IMREAD_GRAYSCALE)
        t2 = cv2.imread(str(t2_path), cv2.IMREAD_GRAYSCALE)

        if flair is None or t1 is None or t2 is None:
            print(f"Error al cargar slice {fname}")
            continue

        rgb = cv2.merge([flair, t1, t2])
        cv2.imwrite(str(output_dir / fname), rgb)

def process_dataset_new(root_input_flair, root_input_t1, root_input_t2, root_output):

    root_input_flair = Path(root_input_flair)
    root_input_t1 = Path(root_input_t1)
    root_input_t2 = Path(root_input_t2)
    root_output = Path(root_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    flair_slices = sorted(root_input_flair.glob("*.png"))
    t1_slices = sorted(root_input_t1.glob("*.png"))
    t2_slices = sorted(root_input_t2.glob("*.png"))

    print(f"FLAIR slices: {len(flair_slices)}, T1 slices: {len(t1_slices)}, T2 slices: {len(t2_slices)}")

    for flair_path in tqdm(flair_slices, desc="Procesando slices"):
        fname = flair_path.name
        t1_path = root_input_t1 / fname.replace("_FLAIR_", "_T1_")
        t2_path = root_input_t2 / fname.replace("_FLAIR_", "_T2_")

        if not (t1_path.exists() and t2_path.exists()):
            print(f"Slice {fname} faltante en alguna modalidad, se omite.")
            continue

        flair = cv2.imread(str(flair_path), cv2.IMREAD_GRAYSCALE)
        t1 = cv2.imread(str(t1_path), cv2.IMREAD_GRAYSCALE)
        t2 = cv2.imread(str(t2_path), cv2.IMREAD_GRAYSCALE)

        if flair is None or t1 is None or t2 is None:
            print(f"Error al cargar slice {fname}")
            continue

        rgb = cv2.merge([flair, t1, t2])
        output_path = root_output / fname.replace("_FLAIR_", "_RGB_")
        cv2.imwrite(str(output_path), rgb)