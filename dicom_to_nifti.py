import os
import numpy as np
import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse

def convert_dcm_to_nifti(dicom_files, output_folder, output_filename):
    try:
        if not dicom_files:
            print(f"No DICOM files found. Skipping...")
            return
        
        slices = sorted([pydicom.dcmread(dcm) for dcm in dicom_files], key=lambda x: float(x.ImagePositionPatient[2]))
        image_3d = np.stack([apply_voi_lut(s.pixel_array, s) for s in slices], axis=-1)

        nifti_img = nib.Nifti1Image(image_3d, np.eye(4))
        output_path = os.path.join(output_folder, f"{output_filename}.nii.gz")
        nib.save(nifti_img, output_path)
        print(f"Saved NIfTI file: {output_path}")
    except Exception as e:
        print(f"Failed to process files {dicom_files}: {e}")

def find_dicom_files(input_dir):
    dicom_file_groups = []
    
    for root, dirs, files in os.walk(input_dir):
        dcm_files = [os.path.join(root, f) for f in files if f.endswith('.dcm')]
        if dcm_files:
            dicom_file_groups.append((dcm_files, root))
    
    return dicom_file_groups

def process_patient_folders(input_dir, output_dir, num_cores):
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    dicom_file_groups = find_dicom_files(input_dir)

    tasks = [
        (dcm_files, os.path.join(output_dir, os.path.relpath(root, input_dir)), os.path.basename(root))
        for dcm_files, root in dicom_file_groups
    ]

    print(f"Using {num_cores} CPU cores for processing")

    with Pool(num_cores) as pool:
        list(tqdm(pool.imap(lambda args: convert_dcm_to_nifti(*args), tasks), total=len(tasks)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM to NIfTI using multiprocessing.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing DICOM files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory where NIfTI files will be saved.')
    parser.add_argument('--num_cores', type=int, default=cpu_count(), help='Number of CPU cores to use (default: all available cores).')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
    else:
        process_patient_folders(args.input_dir, args.output_dir, args.num_cores)
