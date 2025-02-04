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

#-------------------------- start photoface dataset ---------------------
def getPhotoDB(csvPath=None, root_dir=None, image_size=256):

    df = pd.read_csv(csvPath, header=None)
    face = df[0].tolist()

    dataset_size = len(df)

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    test_dataset = PhotoDBPretrain(face, root_dir, image_size=image_size, transform=transform)
    return test_dataset

class PhotoDBPretrain(Dataset):
    def __init__(self, face, root_dir, image_size=256, transform=None):

        self.face = face
        self.transform = transform
        self.root_dir = root_dir
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self._tensor = transforms.ToTensor()

    def __getitem__(self, index):
        face_name = self.face[index]
        face_path = os.path.join(self.root_dir, str(face_name))
        faceimg = Image.open(face_path)
        faceimg_ten = self._tensor(faceimg)
        faceimg_L = faceimg.convert('L')

        face_L = self._tensor(faceimg_L)
        return faceimg_ten, face_L, str(face_name)

    def __len__(self):
        return self.dataset_len