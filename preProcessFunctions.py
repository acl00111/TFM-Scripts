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

def nii_to_jpgs(input_path, output_dir, rgb=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = nib.load(input_path).get_fdata()
    num_slices, width, height = data.shape
    nombre_archivo = input_path.stem
    nombre_base = nombre_archivo.replace('_MASK.nii', '')
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

    nii_files = list(input_root.rglob('*MASK.nii.gz'))
    #nii_files = list(input_root.rglob(scan_type))

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
        nii_to_jpgs(nii_file, output_root, rgb=rgb)