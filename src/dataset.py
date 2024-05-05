import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import glob
import os
import numpy as np
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, datapath, augmentation=None, predict=False, mean=0, std=1, train=True):
        self.datapath = datapath
        self.augmentation = augmentation
        self.predict = predict
        self.train = train

        self.mean = mean
        self.std = std
        
        # Get the list of image files for prediction
        
        if predict:
            image_files = sorted(glob.glob(f'{datapath}/*.jpg'))
            image_seg_files = sorted(glob.glob(f'{datapath}/*_seg.jpg'))
            
            self.dataset_filename = []
            for img_file in image_files:
                if img_file not in image_seg_files:
                    base_name = img_file[:-4]  # Get the base name of the image file
                    seg_file = f'{base_name}_seg.png' # Construct the name of the segmentation file
                
                    if not os.path.exists(seg_file) :  # Check if the segmentation file exists
                        self.dataset_filename.append(base_name)  # Append the base name of the image file to the dataset
                
    def __len__(self):
        if self.train:
            files = glob.glob(os.path.join(self.datapath + "/*[0-9].jpg"))
        elif self.predict:
            files = self.dataset_filename
        else:
            files = glob.glob(os.path.join(self.datapath + "/*[0-9].jpg"))
        return len(files)
    
    def __getitem__(self, idx):
        if not self.predict:
            if self.train:
                img = Image.open(self.datapath + '/img_' + str(idx) + '.jpg')
                mask = Image.open(self.datapath + '/img_' + str(idx)+ '_seg.jpg')
            else:
                img = Image.open(self.datapath + '/img_' + str(idx) + '.jpg')
                mask = Image.open(self.datapath + '/img_' + str(idx) + '_seg.jpg') 

            if self.augmentation:
                img, mask = self.augmentation[0](img, mask)
                img = self.augmentation[1](img)
                
            
            img = T.ToTensor()(img)
            
            mask = self.onehot(mask)
            mask = torch.Tensor(mask)
            
            return img, mask
        
        else:
            filename = self.dataset_filename[idx]
            img = Image.open(filename + '.jpg')
            img = T.ToTensor()(img)
            
            filename = filename.split('/')[-1]
            return img, filename
    
    def onehot(self, img):
        img = np.array(img)[:,:,1]
        binary_mask = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.uint8)
        binary_mask[:, :, 0] = (img < 122) # Background channel
        binary_mask[:, :, 1] = (img[:,:] > 122)
        return binary_mask.transpose(2,0,1)