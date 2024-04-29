import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
# Load the NIfTI file
flair = nib.load('MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_flair.nii')
flair_data = flair.get_fdata()
print(flair_data.shape)
seg = nib.load('MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_seg.nii')
seg_data = seg.get_fdata()
print(seg_data.shape)
t1 = nib.load('MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_t1.nii')
t1_data = t1.get_fdata()
print(t1_data.shape)
t2 = nib.load('MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_t2.nii')
t2_data = t2.get_fdata()
print(t2_data.shape)
t1ce = nib.load('MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_t1ce.nii')
t1ce_data = t1ce.get_fdata()
print(t1ce_data.shape)
slice_index = 77
#mask_slice = seg[:, :, slice_index]
image_slice = flair_data[:, :, slice_index]
plt.imshow(image_slice, cmap='gray')
plt.show()
image_slice = t1_data[:, :, slice_index]
plt.imshow(image_slice, cmap='gray')
plt.show()
image_slice = t2_data[:, :, slice_index]
plt.imshow(image_slice, cmap='gray')
plt.show()
image_slice = t1ce_data[:, :, slice_index]
plt.imshow(image_slice, cmap='gray')
plt.show()
image_slice = seg_data[:, :, slice_index]
plt.imshow(image_slice, cmap='gray')
plt.show()

#get all nii files from folder
def list_nii_files(directory):
    """ Returns a list of paths to NIfTI files in the specified directory. """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii')]

res = list_nii_files("MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1")
print(res[0])
flair = nib.load(res[0])
flair_data = flair.get_fdata()
flair_data = np.transpose(flair_data,(2,0,1))
t1_data = np.transpose(t1_data,(2,0,1))
t1ce_data = np.transpose(t1ce_data,(2,0,1))
t2_data = np.transpose(t2_data,(2,0,1))
print(flair_data.shape)
img_data = [flair_data,t1_data,t1ce_data,t2_data]
combined_array = np.stack(img_data, axis=-1)
print(combined_array.shape)