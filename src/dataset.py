import torch
from torch.utils.data import Dataset
import cv2
import glob
import os
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
            files = self.train_dataset
        else:
            files = self.test_dataset
        return len(files)
    
    def __getitem__(self, idx):
        if self.train:
            img = cv2.imread(self.datapath + 'Train_seg/img' + str(idx) + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.datapath + 'Train_seg/img' + str(idx)+ '_seg.png')
        else:
            img = cv2.imread(self.datapath + 'Test_seg/img' + str(idx) + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.datapath + 'Test_seg/img' + str(idx) + '_seg.png')
        
        mask = self.onehot(torch.as_tensor(np.array(mask), dtype=torch.int64))
        mask = np.transpose(mask, (1, 2, 0))

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']
        
        return img, mask
    
    def onehot(self, img):
        oh = np.zeros((2, img.shape[0], img.shape[1]))
        for i in range(2):
            oh[i, :,:] = (img[:,:, 0] == i)
        return oh