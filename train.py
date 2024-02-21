from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import numpy as np
from aug_arc.Discriminators import Discriminator
from aug_arc.CascadedNetResNetUNV1FNorm import *

from torchvision import transforms
from dataloader_TransMEF import  getPhotoDB_PreTrain_23, get_300W_23
from ssim import SSIM, TV_Loss
import time
import argparse
import log
import copy
from PIL import Image
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from utils import mkdir, cal_normal_acc, get_normal_255, get_Normal_Std_MeanMask
from torchvision.utils import save_image
from renderNormal import *

NWORKERS = 4

parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--exp_name', type=str, default='R+D0.0001+TV0.001', help='Name of the experiment')
parser.add_argument('--save_path', type=str, default='/home/xteam/PaperCode/MM23/CasNet/result/TIP/', help='model and pics save path')

parser.add_argument('--lamda_dnorm', type=float, default=0.0001, help='weight of the SSIM loss')
parser.add_argument('--lamda_mse', type=float, default=1, help='weight of the mse loss')
parser.add_argument('--lamda_tv', type=float, default=0.001, help='weight of the tv loss')

parser.add_argument('--ssl_transformations', type=bool, default=True, help='use ssl_transformations or not')
parser.add_argument('--miniset', type=bool, default=False, help='to choose a mini dataset')
parser.add_argument('--minirate', type=float, default=0.2, help='to detemine the size of a mini dataset')
parser.add_argument('--seed', type=int, default=3, help='random seed (default: 1)')
parser.add_argument('--epoch', type=int, default=2000, help='training epoch')
parser.add_argument('--batch_size', type=int, default=8, help='batchsize')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--model_type', type=str, default='Train', choices=['Train','Test'])

parser.add_argument('--gpuID', type=int, default='1', help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--summary_name', type=str, default='Norm',
                    help='Name of the tensorboard summmary')

args = parser.parse_args()

savepath_TIP = args.save_path + args.exp_name
logpath_TIP = savepath_TIP + '/log/'
mkdir(logpath_TIP) 
ckptPath = savepath_TIP + '/ckpt/'
mkdir(ckptPath) 
writer = SummaryWriter(logpath_TIP)
with open(logpath_TIP + '/weights.txt', 'a+') as f:
    details = 'lamda_dnorm: ' + str(args.lamda_dnorm) + '\t' + 'lamda_mse: ' + str(args.lamda_mse) + '\t' + 'lamda_tv: ' + str(args.lamda_tv) +  '\n'
    f.write(details)
# ==================
# init
# ==================
np.random.seed(args.seed)  # to get the same images and leave it fixed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = "cuda:" + str(args.gpuID)
# device = "cpu"


# ==================
# Read Data
pathd = '/home/xteam/PaperCode/MM23/csv/p2_300w_train.csv'
train_dataset_CA, _ = get_300W_23(csvPath=pathd, validation_split=0)
celebA_dl  = DataLoader(train_dataset_CA, batch_size=args.batch_size, shuffle=True, num_workers=8)


pathd = '/home/xteam/PaperCode/MM23/csv/23Phdb_train_pre.csv'
train_dataset, val_dataset = getPhotoDB_PreTrain_23(csvPath=pathd, IMAGE_SIZE=256, validation_split=0)
train_dl  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

pathd = '/home/xteam/PaperCode/MM23/csv/23Phdb_test_pre.csv'
val_dataset, _ = getPhotoDB_PreTrain_23(csvPath=pathd, IMAGE_SIZE=256, validation_split=0)
valtrain_dl  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

start = time.time()

# ==================
# Init Model
# ==================
flag, epoch = 0, 0

EDmodel = CascadedNetResNetUNV1FNorm_5N(featdim=32).to(device)
D_model_path = '/home/xteam/PaperCode/MM23/CasNet/result/TIP/R+D0.0001+TV0.001_/ckpt/epoch_460.pth'
load_Mod = torch.load(D_model_path)
if D_model_path=='':
    epoch = 0
else:
    epoch = load_Mod['epoch']
EDmodel.load_state_dict(load_Mod['model'])

optimizer = optim.SGD(EDmodel.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" else optim.Adam(EDmodel.parameters(), lr=args.lr, weight_decay=args.wd)
# scheduler_N = CosineAnnealingLR(optimizer, args.epoch)


# ==================
# Model Training
# ==================
trainImgsPath = savepath_TIP + '/trainImgs/'
mkdir(trainImgsPath) 
testImgsPath = savepath_TIP + '/test/'
mkdir(testImgsPath) 
print('============ Training Begins ===============')
EDmodel.train()

criterion_GAN = nn.BCEWithLogitsLoss().to(device)#
D_Norm = Discriminator(3).to(device)
D_Norm.load_state_dict(load_Mod['Dmodel'])
optimizer_D_Norm = torch.optim.Adam(D_Norm.parameters(), lr=0.0001, weight_decay=0.0005)
# scheduler_D = CosineAnnealingLR(optimizer_D_Norm, args.epoch)


def total_variation_WM(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

def N_SFS2CM(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 1, :, :]
    tt[:, 1, :, :] = - normal[:, 0, :, :]
    tt[:, 2, :, :] = normal[:, 2, :, :]
    return tt
CosLoss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

for epoch_ing in tqdm(range(epoch, args.epoch + 1)):

    for index, image in enumerate(train_dl):
        img_orig = image[0].to(device)
        img_orig_L = image[1].to(device)

        b, c, w, h = img_orig_L.shape
        img_gtN = F.normalize(N_SFS2CM(image[2]).to(device))
        optimizer.zero_grad()

        preNorm = EDmodel(img_orig)
        pNorm_1, pNorm_2, pNorm_3, pNorm_4, pNorm_5 = F.normalize(preNorm[0]), F.normalize(preNorm[1]), F.normalize(preNorm[2]), F.normalize(preNorm[3]), F.normalize(preNorm[4])

        if index % 200 == 0:
            sampleImgs = torch.cat([img_orig, get_normal_255((img_gtN)), get_normal_255((pNorm_1)), get_normal_255((pNorm_2)), get_normal_255((pNorm_3)), get_normal_255((pNorm_4))], 0)
            save_image(sampleImgs, trainImgsPath + '/train_%d'%(1) + '_.png', nrow=b, normalize=True)
    
        # mse_loss_C = F.smooth_l1_loss(img_gtN, C_Norm)
        # mse_loss_F = F.smooth_l1_loss(img_gtN, F_Norm)
        # mse_loss_F2 = F.smooth_l1_loss(img_gtN, F_Norm2)
        recLoss_1 = 1 - CosLoss(img_gtN, pNorm_1).mean()
        recLoss_2 = 1 - CosLoss(img_gtN, pNorm_2).mean()
        recLoss_3 = 1 - CosLoss(img_gtN, pNorm_3).mean()
        recLoss_4 = 1 - CosLoss(img_gtN, pNorm_4).mean()
        recLoss_5 = 1 - CosLoss(img_gtN, pNorm_5).mean()
        # (1 - CosLoss(face_gtN, preNorm).mean())
        To_RecLoss = args.lamda_mse * (recLoss_1 + recLoss_2 + recLoss_3 + recLoss_4 + recLoss_5)

        D_real = D_Norm(img_gtN)
        D_fake_1 = D_Norm(pNorm_1.detach())
        D_fake_2 = D_Norm(pNorm_2.detach())
        D_fake_3 = D_Norm(pNorm_3.detach())
        D_fake_4 = D_Norm(pNorm_4.detach())
        D_fake_5 = D_Norm(pNorm_5.detach())

        valid =  torch.ones_like(D_real, requires_grad=False).to(device)
        fake =  torch.zeros_like(D_fake_1, requires_grad=False).to(device)

        optimizer_D_Norm.zero_grad()
        loss_real = criterion_GAN(D_real, valid)
        loss_fake_1 = criterion_GAN(D_fake_1, fake)
        loss_fake_2 = criterion_GAN(D_fake_2, fake)
        loss_fake_3 = criterion_GAN(D_fake_3, fake)
        loss_fake_4 = criterion_GAN(D_fake_4, fake)
        loss_fake_5 = criterion_GAN(D_fake_5, fake)

        D_loss = loss_real + (loss_fake_1 + loss_fake_2 + loss_fake_3 + loss_fake_4 + loss_fake_5)/5.0
        D_loss.backward()
        optimizer_D_Norm.step()

        To_DLoss = args.lamda_dnorm * (criterion_GAN(D_Norm(pNorm_1), valid) + criterion_GAN(D_Norm(pNorm_2), valid) + criterion_GAN(D_Norm(pNorm_3), valid) + criterion_GAN(D_Norm(pNorm_4), valid) + + criterion_GAN(D_Norm(pNorm_5), valid))

        To_TVLoss = args.lamda_tv * (total_variation_WM(pNorm_1/2+0.5) + total_variation_WM(pNorm_2/2+0.5) + total_variation_WM(pNorm_3/2+0.5) + total_variation_WM(pNorm_4/2+0.5) + total_variation_WM(pNorm_5/2+0.5))/(256*256)
        TLoss =  To_RecLoss + To_DLoss + To_TVLoss #+ gd_loss_F ##  #+  

        TLoss.backward()
        optimizer.step()

        print('%s E/L/i/:[%d-%d-%d/%d]-----TLOSS:%.4f--MSE:%.4f---dnorm:%.4f---tvLoss:%.4f' % (args.exp_name, epoch_ing, args.epoch, index, len(train_dl), TLoss, To_RecLoss, To_DLoss, To_TVLoss))
        flag += 1
        
        writer.add_scalar('/Train/TLoss', TLoss, flag)
        writer.add_scalar('/Train/To_RecLoss', To_RecLoss, flag)
        writer.add_scalar('/Train/Dnorm', To_DLoss, flag)
        writer.add_scalar('/Train/TV', To_TVLoss, flag)

        # ==================
        # Model Saving
        # ==================
        # save model every epoch
        state = {
            'epoch': epoch_ing,
            'model': EDmodel.state_dict(),
            'Dmodel': D_Norm.state_dict(),
        }
        torch.save(state, ckptPath+ '/epoch_' + str(epoch_ing) + '.pth')
        end = time.time()

