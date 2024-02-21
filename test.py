import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import numpy as np
from net.model import *
from torchvision import transforms
from dataloader_Cas import  getPhotoDB_PreTrain_23, get_300W_23
\import argparse
import log
import copy

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from utils import mkdir, cal_normal_acc, get_normal_255, get_Normal_Std_MeanMask
from torchvision.utils import save_image



NWORKERS = 4

parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--exp_name', type=str, default='TransMEF_experiments', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--root', type=str, default='./coco', help='data path')
parser.add_argument('--save_path', type=str, default='/home/xteam/PaperCode/MM23/CasNet/result/E5_3N_0718', help='model and pics save path')
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
# Read Data
# pathd = '/home/xteam/PaperCode/MM23/csv/p2_300w_train.csv'
pathd = '/home/xteam/PaperCode/MM23/csv/FFHQ_69000.csv'
# pathd = '/home/xteam/PaperCode/MM_IJCV/csv_data/CelebAtest_crop256.csv'
train_dataset_CA, _ = get_300W_23(csvPath=pathd, validation_split=0)
celebA_dl  = DataLoader(train_dataset_CA, batch_size=args.batch_size-1, shuffle=False, num_workers=8)

pathd = '/home/xteam/PaperCode/MM23/csv/23Phdb_train_pre.csv'
train_dataset, val_dataset = getPhotoDB_PreTrain_23(csvPath=pathd, IMAGE_SIZE=256, validation_split=0)
train_dl  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

pathd = '/home/xteam/PaperCode/MM23/csv/23Phdb_test_pre.csv'
val_dataset, _ = getPhotoDB_PreTrain_23(csvPath=pathd, IMAGE_SIZE=256, validation_split=0)
valtrain_dl  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)


device = "cuda:1"
from aug_arc.NormalEncoder_L4 import *
EDmodel = CascadedNetResNetUNV1FNorm_5N(featdim=32).to(device)
D_model_path = '/home/xteam/PaperCode/MM23/CasNet/epoch_500.pth'
load_Mod = torch.load(D_model_path)
EDmodel.load_state_dict(load_Mod['model'])



# save images
print('============ test Begins ===============')
save_path = D_model_path[: -18]
testImgsPath = save_path + '/celebatest/'
mkdir(testImgsPath) 

def N_SFS2CM(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 1, :, :]
    tt[:, 1, :, :] = - normal[:, 0, :, :]
    tt[:, 2, :, :] = normal[:, 2, :, :]
    return tt

EDmodel.eval()
with torch.no_grad():
    for index, image in enumerate(celebA_dl):
        img_orig = image[0].to(device)
        img_orig_L = image[1].to(device)
        face_path = image[2]
        b, c, w, h = img_orig.shape

        predNorm = EDmodel(img_orig)
        img_Ce_Norm_C = F.normalize(predNorm[-1])

        for ii in range(b):
            tpName = face_path[ii].split('/')[-1]   # face_path[ii][-9:]
            svgname = testImgsPath #+ tpName.split('.')[0] + '/' 
            mkdir(svgname)
            save_image(img_orig[ii], svgname + tpName.replace('.png', '_input.png'), nrow=1, normalize=True)
            save_image(get_normal_255(N_SFS2CM(img_Ce_Norm_C))[ii], svgname + tpName.replace('.png', '_norm1.png'), nrow=1, normalize=True)
        print(index, len(celebA_dl))

