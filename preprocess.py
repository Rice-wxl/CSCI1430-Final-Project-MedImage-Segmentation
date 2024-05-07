from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def list_nii_files(directory):
    """ Returns a list of paths to NIfTI files in the specified directory. """
    return [os.path.join(directory, f) for f in os.listdir(directory)]

class NiiDataset(Dataset):
    def __init__(self, directory, transforms=None):
        self.data_paths = list_nii_files(directory)
        self.transforms = transforms
        self.cache = {}

    def __len__(self):
        return len(self.data_paths) * 155

    def __getitem__(self, idx):
        folder_idx = idx // 155
        slice_idx = idx % 155
        folder_path = self.data_paths[folder_idx]
        
        if folder_path not in self.cache:
            self.cache[folder_path] = self.load_folder_data(folder_path)
        
        data, seg = self.cache[folder_path][slice_idx]

        if self.transforms:
            data = self.transforms(data)
            seg = self.transforms(seg)

        return torch.as_tensor(data, dtype=torch.float32), torch.as_tensor(seg, dtype=torch.float32)

    def load_folder_data(self, folder_path):
        nii_files = list_nii_files(folder_path)
        img_data = []
        seg_data = None
        
        for file_name in nii_files:
            nii = nib.load(file_name)
            nii_data = nii.get_fdata().transpose((2, 0, 1))  # (155, 240, 240)

            if 'seg' in file_name:
                seg_data = nii_data
            else:
                img_data.append(nii_data)

        combined_array = np.stack(img_data, axis=-1)  # (155, 240, 240, 4)
        return [(combined_array[i], seg_data[i]) for i in range(155)]

        

HGG_directory = 'MICCAI_BraTS_2019_Data_Training/HGG'
HGG_dataset = NiiDataset(HGG_directory)
HGG_dataloader =  DataLoader(HGG_dataset, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(HGG_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Feature batch shape: {train_labels.size()}")
#mask_slice = seg[:, :, slice_index]
image_slice = train_features[0,:, :, 0]
plt.imshow(image_slice, cmap='gray')
plt.show()
image_slice = train_features[0,:, :, 1]
plt.imshow(image_slice, cmap='gray')
plt.show()
image_slice = train_features[0,:, :, 2]
plt.imshow(image_slice, cmap='gray')
plt.show()
image_slice = train_features[0,:, :, 3]
plt.imshow(image_slice, cmap='gray')
plt.show()
