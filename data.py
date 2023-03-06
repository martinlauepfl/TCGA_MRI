import os
import math
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp

from train import batchsize

kaggle_3m='./kaggle_3m/'
dirs=glob.glob(kaggle_3m+'*')

data_img=[]
data_label=[]
for subdir in dirs:
    dirname=subdir.split('\\')[-1]
    for filename in os.listdir(subdir):
        img_path=subdir+'/'+filename
        if 'mask' in img_path:
            data_label.append(img_path)
        else:
            data_img.append(img_path)

data_imgx=[]
for i in range(len(data_label)):
    img_mask=data_label[i]
    img=img_mask[:-9]+'.tif'
    data_imgx.append(img)

data_newimg=[]
data_newlabel=[]
for i in data_label:
    value=np.max(cv2.imread(i))
    try:
        if value>0:
            data_newlabel.append(i)
            i_img=i[:-9]+'.tif'
            data_newimg.append(i_img)
    except:
        pass

train_transformer=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])
test_transformer=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


class BrainMRIdataset(Dataset):
    def __init__(self, img, mask, transformer):
        self.img = img
        self.mask = mask
        self.transformer = transformer

    def __getitem__(self, index):
        img = self.img[index]
        mask = self.mask[index]

        img_open = Image.open(img)
        img_tensor = self.transformer(img_open)

        mask_open = Image.open(mask)
        mask_tensor = self.transformer(mask_open)

        mask_tensor = torch.squeeze(mask_tensor).type(torch.long)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img)

s=1000
train_img=data_newimg[:s]
train_label=data_newlabel[:s]
test_img=data_newimg[s:]
test_label=data_newlabel[s:]

train_data=BrainMRIdataset(train_img,train_label,train_transformer)
test_data=BrainMRIdataset(test_img,test_label,test_transformer)

dl_train=DataLoader(train_data,batch_size=batchsize,shuffle=True)
dl_test=DataLoader(test_data,batch_size=batchsize,shuffle=True)



