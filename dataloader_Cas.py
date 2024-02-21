import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import pandas as pd
from torch.nn import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from imgaug import augmenters as iaa
import pandas as pd
from torch.utils.data import Dataset, random_split
import numpy as np
import copy, random

IMAGE_SIZE = 256
#-------------------------- start photoface dataset ---------------------
def getPhotoDB_PreTrain_23(csvPath=None, IMAGE_SIZE=256, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    full_dataset = PhotoDB_PreTrain_23(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB_PreTrain_23(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self._tensor = transforms.ToTensor()

    def __getitem__(self, index):
        face_path = self.face[index]
        faceimg = Image.open(face_path)
        faceimg_ten = self._tensor(faceimg)
        faceimg_L = faceimg.convert('L')

        face_L = self._tensor(faceimg_L)

        tpimg = face_path.split('/')[-1]
        gt_path = face_path[:-len(tpimg)] +  '/sn_c.png'
        gPN = self.DataTrans(Image.open(gt_path))
        gtN = 2 * (gPN - 0.5)

        maskph = face_path[:-len(tpimg)] +  '/mask_c.png'

        mask =  self.DataTrans(Image.open(maskph))
        return faceimg_ten, face_L, gtN, mask, face_path

    def __len__(self):
        return self.dataset_len
#-------------------------- end hotoface dataset ---------------------



def get_300W_23(csvPath=None, validation_split=0):
    df = pd.read_csv(csvPath)
    face = list(df['face'])
    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    full_dataset = Dataset_300W_23(face, transform)
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class Dataset_300W_23(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self._tensor = transforms.ToTensor()

    def __getitem__(self, index):
        face_path = self.face[index]
        faceimg = Image.open(face_path)

        face_t =  self._tensor(faceimg)


        faceimg_L = faceimg.convert('L')

        face_L = self._tensor(faceimg_L)
        return face_t, face_L, face_path

    def __len__(self):
        return self.dataset_len
