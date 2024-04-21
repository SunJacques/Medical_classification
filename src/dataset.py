import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import glob
import os
import cv2
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, datapath, augmentation=None, preprocessing=None, mean=0, std=1, train=True):
        self.datapath = datapath
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.train = train

        self.mean = mean
        self.std = std
    
    def __len__(self):
        if self.train:
            files = glob.glob(os.path.join(self.datapath + "/*[0-9].jpg"))
        else:
            files = glob.glob(os.path.join(self.datapath + "/*[0-9].jpg"))
        return len(files)
    
    def __getitem__(self, idx):
        if self.train:
            img = cv2.imread(self.datapath + '/img' + str(idx) + '.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.datapath + '/img' + str(idx)+ '_seg.jpg', cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(self.datapath + '/img' + str(idx) + '.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.datapath + '/img' + str(idx) + '_seg.jpg', cv2.IMREAD_GRAYSCALE) 
            
        img = T.ToTensor()(img)
        mask = self.onehot(mask)
        mask = T.ToTensor()(mask)
        
        if self.augmentation:
            img, mask = self.augmentation(img, mask)
            img = img
            mask = mask

        if self.preprocessing:
            img,mask = self.preprocessing(img, mask)
        
        
        return img, mask
    
    def onehot(self, img):
        binary_mask = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.uint8)
        binary_mask[:, :, 0] = (img == 0).astype(np.uint8)  # Background channel
        binary_mask[:, :, 1] = (img == 255).astype(np.uint8) 
        return binary_mask