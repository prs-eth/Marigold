


from random import randrange
from re import I

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from tqdm import tqdm

INPUT_DIM = 4
FEATURE_DIM = 64

class GADBase(nn.Module):
    """
    Source: https://github.com/prs-eth/Diffusion-Super-Resolution/blob/main/model/gad_base.py
    Paper: https://arxiv.org/abs/2211.11592
    """    
    def __init__(
            self, feature_extractor='UNet',
            Npre=8000, Ntrain=0,
            gamma=1.0,
    ):
        super().__init__()

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
        self.gamma = gamma
 
        if feature_extractor=='none': 
            # RGB verion of DADA does not need a deep feature extractor
            self.feature_extractor = None
            self.Ntrain = 0
            self.logk = torch.log(torch.tensor(0.03))

        elif feature_extractor=='UNet':
            # Learned verion of DADA
            self.feature_extractor =  torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bicubic'),
                # torch.nn.Identity(),
                smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=INPUT_DIM),
                torch.nn.AvgPool2d(kernel_size=2, stride=2)
                # torch.nn.Identity(),
            )
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')
             

    # def forward(self, sample, train=False, deps=0.1):
    def forward(self, source, guide, mask_lr=None, train=False, deps=0.1,
                Nsteps=None, gamma=None):
        # guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']

        mask_lr = mask_lr if mask_lr is not None else torch.ones_like(source)
        init_depth = torch.nn.functional.interpolate(source, guide.shape[-2:], mode='bilinear', align_corners=False)
        Npre = self.Npre if Nsteps is None else Nsteps
        gamma = self.gamma if train else gamma

        # assert that all values are positive, otherwise shift depth map to positives
        if source.min()<=deps:
            # print("Warning: The forward function was called with negative depth values. Values were temporarly shifted. Consider using unnormalized depth values for stability.")
            source += deps
            init_depth += deps
            shifted = True
        else:
            shifted = False

        y_pred, aux = self.diffuse(init_depth.clone(), guide.clone(), source, mask_lr < 0.5,
                 K=torch.exp(self.logk), verbose=False, train=train, Npre=Npre, gamma=gamma)

        # revert the shift
        if shifted:
            y_pred -= deps

        # return {'y_pred': y_pred} | aux
        return {**{'y_pred': y_pred}, **aux}


    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False,
        Npre=8000, gamma=1.0):

        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape

        # Define Downsampling operations that depend on the input size
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        # Deep Learning version or RGB version to calucalte the coefficients
        if self.feature_extractor is None:
            guide_feats = guide
        else:
            
            # Concatenate the guide and the image
            cat_input = torch.cat([guide, img-img.mean((1,2,3), keepdim=True) ], 1)

            # Pad the input to be a multiple of 32
            height, width = img.shape[2], img.shape[3]
            pad_height = (32 - height % 32) % 32
            pad_width = (32 - width % 32) % 32
            padded_img = F.pad(cat_input, (0, pad_width, 0, pad_height), mode='constant', value=0)

            # Extract features
            guide_feats = self.feature_extractor(padded_img)

            # Remove padding
            if pad_height > 0:
                guide_feats = guide_feats[:,:,:-pad_height,:]
            if pad_width > 0:
                guide_feats = guide_feats[:,:,:,:-pad_width]
        
        # Convert the features to8 coefficients with the Perona-Malik edge-detection function
        cv, ch = c(guide_feats, K=K)

        # Iterations without gradient
        if self.Npre>0: 
            with torch.no_grad():
                Npre = randrange(Npre) if train else Npre
                for t in tqdm(range(Npre), disable=False):                     
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, gamma=gamma, eps=1e-8)

        # Iterations with gradient
        if self.Ntrain>0: 
            for t in range(self.Ntrain): 
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, gamma=gamma, eps=1e-8)

        return img, {"cv": cv, "ch": ch}


# @torch.jit.script
def c(I, K: float=0.03):
    # apply function to both dimensions
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]), 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]), 1), 1), K)
    return cv, ch

# @torch.jit.script
def g(x, K: float=0.03):
    # Perona-Malik edge detection
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))

@torch.jit.script
def diffuse_step(cv, ch, I, l: float=0.24):
    # Anisotropic Diffusion implmentation, Eq. (1) in paper.

    # calculate diffusion update as increments
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv 

    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th 
    
    return I

def adjust_step(img, source, mask_inv, upsample, downsample, gamma=1.0, eps=1e-8):
    # Implementation of the adjustment step. Eq (3) in paper.

    # Iss = subsample img
    img_ss = downsample(img)

    # Rss = source / Iss
    ratio_ss = source / (img_ss + eps)
    ratio_ss[mask_inv] = 1

    # R = NN upsample r
    ratio = upsample(ratio_ss)

    # ratio = torch.sqrt(ratio)
    # img = img * R
    # img = img * (1 + (R-1)*gamma)
    img = img + img * (ratio-1) * gamma

    return img
    # return img * ratio 