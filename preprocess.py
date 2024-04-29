from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torch
import os
import numpy as np

def list_nii_files(directory):
    """ Returns a list of paths to NIfTI files in the specified directory. """
    return [os.path.join(directory, f) for f in os.listdir(directory)]

class NiiDataset(Dataset):
    def __init__(self, directory,transforms=None):
        '''
        directory: should be MICCAI_BraTS_2019_Data_Training/HGG or LGG
        '''
        self.data_paths = list_nii_files(directory) #data_paths should be a list of things like MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1 
        self.transforms = transforms
        #self.seg_masks, self.data = self.getAllItem() #get all data
        self.current_folder = None
        self.seg_data = None
        self.combined_array = None

    def __len__(self):
        return len(self.data_paths)*155
    
    def __getitem__(self, idx):
        '''
        Get one 2d item from the dataset
        Return: data          #ndarray (1,240,240)
                seg_mask      #ndarray (1,240,240,4)
        '''
        # seg_mask = self.seg_masks[idx]
        # data = self.data[idx]
        index = idx//155 #get the index for the folder so that idx can be seen as index for each image
        folder_path = self.data_paths[index] #get the specific folder path
        if folder_path != self.current_folder:
            #if it is a new folder
            nii_files = list_nii_files(folder_path) #get the list of nii files
            img_data = []
            for i in range(len(nii_files)):
                file_name = nii_files[i]
                nii = nib.load(file_name)
                nii_data = nii.get_fdata() #should have shape (240, 240, 155)
                nii_data = nii_data.transpose((2,0,1)) #should have shape (155, 240, 240)
                #get the segmentation mask out
                if "seg" in file_name:
                    self.seg_data = nii_data
                else:
                    #append all other file into the same list
                    img_data.append(nii_data)
            self.combined_array = np.stack(img_data, axis=-1)#stack all img_data together should have shape (155,240,240,4)
        res_data = self.combined_array[idx%155]
        res_seg = self.seg_data[idx%155]
        # if self.transforms:
        #     #augmentation
        #     seg_mask = self.transform(seg_mask)
        #     data = self.transform(data)
        return res_data, res_seg

        # def getAllItem(self):
        #     '''
        #     Get all item from the dataset at once
        #     Return: all_seg_masks #ndarray (nx155,240,240)
        #             all_data      #ndarray (nx155,240,240,4)
        #     '''
        #     all_seg_masks = [] #the list contains all segmentation masks(n,155,240,240)
        #     all_data = [] #the list contains all data(n,155,240,240,4)
        #     for idx in range(len(self.data_paths)):
        #         #for all data folder in the data path
        #         folder_path = self.data_paths[idx] #get the specific folder path
        #         nii_files = list_nii_files(folder_path) #get the list of nii files
        #         seg_data = None
        #         img_data = []
        #         for i in range(len(nii_files)):
        #             file_name = nii_files[i]
        #             nii = nib.load(file_name)
        #             nii_data = nii.get_fdata() #should have shape (240, 240, 155)
        #             nii_data = nii_data.transpose((2,0,1)) #should have shape (155, 240, 240)
        #             #get the segmentation mask out
        #             if "seg" in file_name:
        #                 seg_data = nii_data
        #             else:
        #                 #append all other file into the same list
        #                 img_data.append(nii_data)
        #         combined_array = np.stack(img_data, axis=-1)#stack all img_data together should have shape (155,240,240,4)
        #         all_seg_masks.append(seg_data)
        #         all_data.append(combined_array)
        #     all_seg_masks = np.concatenate(all_seg_masks, axis=0)#(nx155,240,240)
        #     all_data = np.concatenate(all_data, axis=0)#(nx155,240,240,4)
            
        #    return all_seg_masks, all_data

        

HGG_directory = 'MICCAI_BraTS_2019_Data_Training/HGG'
HGG_dataset = NiiDataset(HGG_directory)
HGG_dataloader =  DataLoader(HGG_dataset, batch_size=64, shuffle=True)
train_features, train_labels = next(iter(HGG_dataloader))
print(f"Feature batch shape: {train_features.size()}")