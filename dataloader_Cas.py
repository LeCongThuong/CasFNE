import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.nn import Parameter
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import os

import pandas as pd
import numpy as np
import copy, random

IMAGE_SIZE = 256
#-------------------------- start photoface dataset ---------------------
def getPhotoDB(csvPath=None, IMAGE_SIZE=256):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    test_dataset = PhotoDBPretrain(face, transform)
    return test_dataset

class PhotoDBPretrain(Dataset):
    def __init__(self, face, root_dir, transform=None):

        self.face = face
        self.transform = transform
        self.root_dir = root_dir
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self._tensor = transforms.ToTensor()

    def __getitem__(self, index):
        face_path = self.face[index]
        face_path = os.path.join(self.root_dir, face_path)
        faceimg = Image.open(face_path)
        faceimg_ten = self._tensor(faceimg)
        faceimg_L = faceimg.convert('L')

        face_L = self._tensor(faceimg_L)

        gt_path = os.path.join(Path(face_path).parent, '/sn.npy')
        gt = self.DataTrans(np.load(gt_path))
        return faceimg_ten, face_L, gt, face_path

    def __len__(self):
        return self.dataset_len