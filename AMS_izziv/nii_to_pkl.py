import os
import nibabel as nib
import numpy as np
import pickle
import matplotlib.pyplot as plt

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


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))

    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.savefig('output_image2.png')
    #plt.show()
    
# Load the .nii.gz file
file_path = '/media/FastDataMama/anton/Release_06_12_23/imagesTr/ThoraxCBCT_0000_0000.nii.gz'
img = nib.load(file_path)
data = img.get_fdata()
print(type(data))

# Select slices to display
slice_0 = data[:, :, data.shape[2] // 2]
slice_1 = data[:, data.shape[1] // 2, :]
slice_2 = data[data.shape[0] // 2, :, :]

# Display the slices
#show_slices([slice_0, slice_1, slice_2])



def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
    plt.savefig('output_image5.png')

# Load the .pkl file
file_path = '/media/FastDataMama/anton/OASIS_L2R_2021_task03/All/p_0001.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)


# Check the contents of the .pkl file
data = tuple(data)
print(type(data))
if isinstance(data, tuple):
    data = np.array(data)  # Adjust this based on the actual structure of your .pkl file

# Ensure data is a numpy array
if not isinstance(data, np.ndarray):
    raise ValueError("The loaded data is not a numpy array")

# Select slices to display
slice_0 = data[0,:, :, data.shape[2] // 2]
slice_1 = data[0,:, data.shape[1] // 2, :]
slice_2 = data[0,data.shape[0] // 2, :, :]

# Display the slices
#show_slices([slice_0, slice_1, slice_2])

import nibabel as nib

# Load the .nii.gz file
nii_img = nib.load('/media/FastDataMama/anton/Release_06_12_23/imagesTr/ThoraxCBCT_0000_0002.nii.gz')
img_data = nii_img.get_fdata()

# Print the structure of the data
print(f'Data type: {type(img_data)}, shape: {img_data.shape}')
