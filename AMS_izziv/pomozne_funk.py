import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import pickle

def downsample_nii(input_folder, output_folder, factor):
    """
    Downsamples 3D NIfTI (.nii.gz) images by a given factor.

    Parameters:
        input_folder (str): Path to the folder containing input .nii.gz files.
        output_folder (str): Path to save the downsampled .nii.gz files.
        factor (float): The downsampling factor (e.g., 2 means reduce each dimension by half).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each .nii.gz file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.nii.gz'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            try:
                # Load the NIfTI file
                nii = nib.load(input_file_path)
                data = nii.get_fdata()
                affine = nii.affine

                # Calculate the zoom factors for each dimension
                zoom_factors = [1 / factor] * data.ndim

                # Downsample the image
                downsampled_data = zoom(data, zoom_factors, order=3)  # Use cubic interpolation
                
                # Save the downsampled image
                downsampled_nii = nib.Nifti1Image(downsampled_data, affine)
                nib.save(downsampled_nii, output_file_path)
                
                print(f"Processed {file_name}: Downsampled file saved to {output_file_path}")

            except Exception as e:
                print(f"An error occurred while processing {file_name}: {e}")

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
                pickle.dump(tuple(img_data), pkl_file)

            print(f'Converted {file_name} to {pkl_path}')

# Example usage
input_folder = '/media/FastDataMama/anton/Release_06_12_23/imagesTr/'  # Folder containing .nii.gz files
output_folder = '/media/FastDataMama/anton/Release_06_12_23/nii_decim_train/'  # Folder to save downsampled .nii.gz files
downsampling_factor = 2  # Reduce each dimension by half
pkl_folder = '/media/FastDataMama/anton/Release_06_12_23/pkl_imgs/'


data1 = nib.load('/media/FastDataMama/anton/Release_06_12_23/imagesTr/ThoraxCBCT_0003_0002.nii.gz')

data2 = '/media/FastDataMama/anton/Release_06_12_23/atlas_pkl/ThoraxCBCT_0003_0002.pkl'
with open(data2, 'rb') as f:
    data2 = pickle.load(f)
data3 = '/media/FastDataMama/anton/pkl_testdata/Test/subject_3.pkl'
with open(data3, 'rb') as f:
    data3 = pickle.load(f)
data3 = np.array(data3)

data4 = '/media/FastDataMama/anton/OASIS_L2R_2021_task03/All/p_0015.pkl'
with open(data4, 'rb') as f:
    data34= pickle.load(f)
data4 = np.array(data3)

print(type(data1))
print(type(data2))
print(type(data3))
print(type(data4))

print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)

print(data1.header)
print('----------------------')
print(data2[0:10,:,:])
print('----------------------')
print(data3[:,0:10,:,:])

#downsample_nii(input_folder, output_folder, downsampling_factor)
#convert_nii_to_pkl(output_folder, pkl_folder)
