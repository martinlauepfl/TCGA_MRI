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

kaggle_3m='./kaggle_3m/'
dirs=glob.glob(kaggle_3m+'*')
# dirs

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

im=data_img[1]
im=Image.open(im)
