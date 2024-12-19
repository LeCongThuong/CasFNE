import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import numpy as np
from net.model import *
from torchvision import transforms
from dataloader_Cas import  getPhotoDB, PhotoDBPretrain
import argparse
from pathlib import Path

from tqdm import tqdm
from torch.nn import functional as F
from torchvision.utils import save_image



NWORKERS = 4

parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--exp_name', type=str, default='TransMEF_experiments', help='Name of the experiment')
parser.add_argument('--root_data_dir', type=str, default=1, help='Data directory')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--model_path', type=str, default='model', help='model folder path')
parser.add_argument('--ssl_transformations', type=bool, default=True, help='use ssl_transformations or not')
parser.add_argument('--miniset', type=bool, default=False, help='to choose a mini dataset')
parser.add_argument('--minirate', type=float, default=0.2, help='to detemine the size of a mini dataset')
parser.add_argument('--seed', type=int, default=3, help='random seed (default: 1)')
parser.add_argument('--epoch', type=int, default=200, help='training epoch')
parser.add_argument('--batch_size', type=int, default=8, help='batchsize')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--model_type', type=str, default='Train', choices=['Train','Test'])

parser.add_argument('--gpuID', type=int, default='0', help='comma delimited of gpu ids to use. Use "-1" for cpu usage')

parser.add_argument('--lamda_dnorm', type=float, default=0.005, help='weight of the SSIM loss')
parser.add_argument('--lamda_grad', type=float, default=0.1, help='weight of the SSIM loss')
parser.add_argument('--lamda_mse', type=float, default=10, help='weight of the mse loss')
parser.add_argument('--lamda_vggC', type=float, default=0.05, help='weight of the SSIM loss')
parser.add_argument('--lamda_vggF', type=float, default=0.05, help='weight of the SSIM loss')

parser.add_argument('--lamda_ssim', type=float, default=0.1, help='weight of the SSIM loss')
parser.add_argument('--lamda_tv', type=float, default=100, help='weight of the tv loss')


parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--summary_name', type=str, default='Norm',
                    help='Name of the tensorboard summmary')

args = parser.parse_args()

np.random.seed(args.seed)  # to get the same images and leave it fixed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# ==================
pathd = '/home/xteam/PaperCode/MM23/csv/23Phdb_test_pre.csv'
testDataset = getPhotoDB(csvPath=pathd, IMAGE_SIZE=256, validation_split=0)
test_dl  = DataLoader(testDataset, batch_size=args.batch_size, shuffle=True, num_workers=8)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EDmodel = CasFNE_5N(featdim=32).to(device)
D_model_path = args.model_path
load_Mod = torch.load(D_model_path)
EDmodel.load_state_dict(load_Mod['model'])



# save images
print('============ test Begins ===============')
testImgsPath = os.path.join(args.out_path, '/PhotofaceDatabase/')
Path(testImgsPath).mkdir(parents=True, exist_ok=True) 

def N_SFS2CM(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 1, :, :]
    tt[:, 1, :, :] = - normal[:, 0, :, :]
    tt[:, 2, :, :] = normal[:, 2, :, :]
    return tt

def get_normal_255(normal):
    normal = (normal + 1) / 2
    normal = normal * 255
    return normal

EDmodel.eval()
with torch.no_grad():
    for index, image in enumerate(test_dl):
        img_orig = image[0].to(device)
        img_orig_L = image[1].to(device)
        face_path = image[2]
        b, c, w, h = img_orig.shape

        predNorm = EDmodel(img_orig)
        img_Ce_Norm_C = F.normalize(predNorm[-1])

        for ii in range(b):
            tpName = face_path[ii].split('/')[-1]   # face_path[ii][-9:]
            save_image(img_orig[ii], os.path.join(testImgsPath, tpName.replace('.png', '_input.png')), nrow=1, normalize=True)
            save_image(get_normal_255(N_SFS2CM(img_Ce_Norm_C))[ii], os.path.join(testImgsPath, tpName.replace('.png', '_norm1.png')), nrow=1, normalize=True)
        print(index, len(test_dl))

