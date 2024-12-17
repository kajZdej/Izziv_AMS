import os
import nibabel as nib
import numpy as np
import pickle

def convert_nii_to_pkl(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.nii.gz'):
            # Load the .nii.gz file
            nii_path = os.path.join(input_dir, file_name)
            nii_img = nib.load(nii_path)
            img_data = nii_img.get_fdata()

            # Create a corresponding .pkl file
            pkl_path = os.path.join(output_dir, file_name.replace('.nii.gz', '.pkl'))
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(img_data, pkl_file)

            print(f'Converted {file_name} to {pkl_path}')

# Define input and output directories
input_dir = '/media/FastDataMama/anton/Release_06_12_23/masksTr'
output_dir = '/media/FastDataMama/anton/Release_06_12_23/atlas_pkl'

# Run the conversion
convert_nii_to_pkl(input_dir, output_dir)
