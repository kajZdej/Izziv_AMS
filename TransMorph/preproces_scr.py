import glob
from data import datasets, trans
from torchvision import transforms

# Define paths to your raw data
train_dir = 'path/to/train/data'
val_dir = 'path/to/val/data'

# Define transformations
train_composed = transforms.Compose([
    trans.RandomFlip(),
    trans.RandomRotation(),
    trans.Seg_norm(),
    trans.NumpyType((np.float32, np.int16)),
])

val_composed = transforms.Compose([
    trans.Seg_norm(),
    trans.NumpyType((np.float32, np.int16)),
])

# Create datasets
train_set = datasets.JHUBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
val_set = datasets.JHUBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)

