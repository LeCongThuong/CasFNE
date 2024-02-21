import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import torchvision.models as models


class CasFNE_Based(nn.Module):
    def __init__(self, featdim=128):
        super(CasFNE_Based, self).__init__()
        # Res feat  
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
               
        self.deconv4 = nn.ConvTranspose2d(512, featdim, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(featdim)
        self.deconv5 = nn.ConvTranspose2d(featdim + 256, featdim, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(featdim)
        self.deconv6 = nn.ConvTranspose2d(featdim +  128, featdim, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(featdim)
        self.deconv7 = nn.ConvTranspose2d(featdim +  64, featdim, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(featdim)
        self.deconv8 = nn.ConvTranspose2d(featdim +  64, featdim, 4, 2, 1)
        self.deconv8_bn = nn.BatchNorm2d(featdim)
        self.conhead = nn.ConvTranspose2d(featdim, 3, 3, 1, 1)

    # forward method
    def forward(self, ipnut):
        layer0 = self.layer0(ipnut)  #64*128*128
        layer1 = self.layer1(layer0) #64*64*64
        layer2 = self.layer2(layer1) #128*32*32
        layer3 = self.layer3(layer2) #256*16*16
        layer4 = self.layer4(layer3) #512*8*8

        d4 = self.deconv4_bn(self.deconv4(layer4))    # 8,  512  --> 512
        d4 = torch.cat([d4, layer3], 1)               
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))    # 16, 512 +  256 + dimwm*16    ---> 512 
        d5 = torch.cat([d5, layer2], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))    # 32, 512 +  128 + dimwm*8   ---> 512 
        d6 = torch.cat([d6, layer1], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))    # 64, 512 +  64    ---> 512 
        d7 = torch.cat([d7, layer0], 1)
        d8 = self.deconv8_bn(self.deconv8(F.relu(d7)))    # 64, 512 +  64    ---> 512 
        d9 = self.conhead(d8)   
        o = F.normalize(torch.tanh(d9))
        return o, (layer0, layer1, layer2, layer3, layer4)
            
class CasNormalEncoder(nn.Module):
    def __init__(self, featdim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, featdim, 4, 2, 1)
        self.conv2 = nn.Conv2d(featdim, featdim * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(featdim * 2)
        self.conv3 = nn.Conv2d(featdim * 2, featdim * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(featdim * 4)
        self.conv4 = nn.Conv2d(featdim * 4, featdim * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(featdim * 8)
        self.conv5 = nn.Conv2d(featdim * 8, featdim * 8, 4, 2, 1)
        # self.conv5_bn = nn.BatchNorm2d(d * 8)

    def forward(self, input):
        e1 = self.conv1(input)#128
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))#64
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))#32
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))#16
        e5 = self.conv5(F.leaky_relu(e4, 0.2))#8
        return e5

class CasNormalDecoder(nn.Module):
    def __init__(self, featdim=64):
        super().__init__()
        self.N2deconv4 = nn.ConvTranspose2d(512 + featdim * 8, featdim * 8, 4, 2, 1)
        self.N2deconv4_bn = nn.BatchNorm2d(featdim * 8)
        self.N2deconv5 = nn.ConvTranspose2d(256 + featdim * 8, featdim * 4, 4, 2, 1)
        self.N2deconv5_bn = nn.BatchNorm2d(featdim * 4)
        self.N2deconv6 = nn.ConvTranspose2d(128 + featdim * 4, featdim * 2, 4, 2, 1)
        self.N2deconv6_bn = nn.BatchNorm2d(featdim * 2)
        self.N2deconv7 = nn.ConvTranspose2d(64 + featdim * 2, featdim * 2, 4, 2, 1)
        self.N2deconv7_bn = nn.BatchNorm2d(featdim * 2)
        self.N2deconv8 = nn.ConvTranspose2d(64 + featdim * 2, 3, 4, 2, 1)

    def forward(self, structfeats, normalfeats):
        catN1feat = torch.cat([structfeats[4], normalfeats], 1) # e5 8*8
        N2d4 = self.N2deconv4_bn(self.N2deconv4(F.relu(catN1feat)))
        N2d4 = torch.cat([N2d4, structfeats[3]], 1)
        N2d5 = self.N2deconv5_bn(self.N2deconv5(F.relu(N2d4)))
        N2d5 = torch.cat([N2d5, structfeats[2]], 1)
        N2d6 = self.N2deconv6_bn(self.N2deconv6(F.relu(N2d5)))
        N2d6 = torch.cat([N2d6, structfeats[1]], 1)
        N2d7 = self.N2deconv7_bn(self.N2deconv7(F.relu(N2d6)))
        N2d7 = torch.cat([N2d7, structfeats[0]], 1)
        N2d8 = self.N2deconv8(F.relu(N2d7))
        findNorm = F.normalize(torch.tanh(N2d8))     
        return findNorm
   

class CasFNE_4N(nn.Module):
    # initializers
    def __init__(self, featdim=128):
        super(CasFNE_4N, self).__init__()

        #based network
        self.BasedNormalNet = CasFNE_Based(featdim=featdim)
        # CascadedNet_1 encoder
        self.NormalEncoder = CasNormalEncoder(featdim=featdim)
        # CascadedNet_1 decoder
        self.NormalDecoder = CasNormalDecoder(featdim=featdim)

               
    # forward method
    def forward(self, input):
        preNorm_1, structfeats = self.BasedNormalNet(input)

        # cascade_1, e7
        casNormFeat_1 = self.NormalEncoder(preNorm_1)
        preNorm_2 = self.NormalDecoder(structfeats, casNormFeat_1)
        # cascade_2, e7
        casNormFeat_2 = self.NormalEncoder(preNorm_2)
        preNorm_3 = self.NormalDecoder(structfeats, casNormFeat_2)
          # cascade_2, e7
        casNormFeat_3 = self.NormalEncoder(preNorm_3)
        preNorm_4 = self.NormalDecoder(structfeats, casNormFeat_3)                  
        return preNorm_1, preNorm_2, preNorm_3, preNorm_4
    
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class CasFNE_5N(nn.Module):
    # initializers
    def __init__(self, featdim=128):
        super(CasFNE_5N, self).__init__()
        #based network
        self.BasedNormalNet = CasFNE_Based(featdim=featdim)
        # CascadedNet_1 encoder
        self.NormalEncoder = CasNormalEncoder(featdim=featdim)
        # CascadedNet_1 decoder
        self.NormalDecoder = CasNormalDecoder(featdim=featdim)

               
    # forward method
    def forward(self, input):
        preNorm_1, structfeats = self.BasedNormalNet(input)

        # cascade_1, e7
        casNormFeat_1 = self.NormalEncoder(preNorm_1)
        preNorm_2 = self.NormalDecoder(structfeats, casNormFeat_1)
        # cascade_2, e7
        casNormFeat_2 = self.NormalEncoder(preNorm_2)
        preNorm_3 = self.NormalDecoder(structfeats, casNormFeat_2)
          # cascade_2, e7
        casNormFeat_3 = self.NormalEncoder(preNorm_3)
        preNorm_4 = self.NormalDecoder(structfeats, casNormFeat_3) 

        casNormFeat_4 = self.NormalEncoder(preNorm_4)
        preNorm_5 = self.NormalDecoder(structfeats, casNormFeat_4)                
        return preNorm_1, preNorm_2, preNorm_3, preNorm_4, preNorm_5
    
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)



class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
    
import time
if __name__ == "__main__":

    Model = CasFNE_5N(featdim=32).cuda()     # 38.2G, 29.3M

    x = torch.rand((1, 3, 256, 256)).cuda()
    flops_1, params_1 = profile(Model, inputs=(x, ))
    print('FLOPs = ' + str((flops_1)/1000**3) + 'G')
    print('Params = ' + str((params_1)/1000**2) + 'M')
    

