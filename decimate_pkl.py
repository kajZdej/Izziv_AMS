import os
import pickle

def decimate_pkl_folder(input_folder, output_folder, factor):
    """
    Decimates all .pkl files in a folder by taking every nth item.

    Parameters:
        input_folder (str): Path to the folder containing input .pkl files.
        output_folder (str): Path to save the decimated .pkl files.
        factor (int): The decimation factor (e.g., 2 means every 2nd item).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each .pkl file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pkl'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            
            try:
                # Load the data from the .pkl file
                with open(input_file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Check if the data is iterable
                if not isinstance(data, (list, tuple)):
                    raise ValueError(f"The data in {file_name} must be a list or tuple to decimate.")
                
                # Decimate the data
                decimated_data = data[::factor]
                
                # Save the decimated data back to a new pickle file
                with open(output_file_path, 'wb') as f:
                    pickle.dump(decimated_data, f)
                
                print(f"Processed {file_name}: Decimated data saved to {output_file_path}")
            
            except Exception as e:
                print(f"An error occurred while processing {file_name}: {e}")

# Example usage
input_folder = '/media/FastDataMama/anton/Release_06_12_23/pkl_imgs/'  # Folder containing .pkl files
output_folder = '/media/FastDataMama/anton/Release_06_12_23/pkl_decim_train'  # Folder to save decimated .pkl files
decimation_factor = 2  # Keep every 2nd item

decimate_pkl_folder(input_folder, output_folder, decimation_factor)
