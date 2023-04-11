# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
NAFSSR: Stereo Image Super-Resolution Using NAFNet

@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology

from basicsr.archs.NAFNet_arch import LayerNorm2d, NAFBlock
from basicsr.archs.arch_util import MySequential
from basicsr.archs.local_arch import Local_Base
from basicsr.utils.registry import ARCH_REGISTRY


def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)
    return torch.from_numpy(mask_np).float().to(device)


class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.criterion = nn.L1Loss()
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward_(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

    def forward(self, x_l, x_r, LR_left, LR_right, loss):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, W (transposed)
        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        b, h, w, c = Q_l.shape
        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale
        attention_T = attention.permute(0, 1, 3, 2)
        M_right_to_left = torch.softmax(attention, dim=-1)
        M_left_to_right = torch.softmax(attention_T, dim=-1)

        V_left_to_right = torch.sum(M_left_to_right.detach(), 2) > 0.1
        V_left_to_right = morphologic_process(V_left_to_right.view(b, 1, h, w))
        V_right_to_left = torch.sum(M_right_to_left.detach(), 2) > 0.1
        V_right_to_left = morphologic_process(V_right_to_left.view(b, 1, h, w))

        M_left_right_left = torch.matmul(M_right_to_left, M_left_to_right)
        M_right_left_right = torch.matmul(M_left_to_right, M_right_to_left)

        F_r2l = torch.matmul(M_right_to_left, V_r)  #B, H, W, c
        F_l2r = torch.matmul(M_left_to_right, V_l)  #B, H, W, c
        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma

        ### loss_smoothness
        loss_h = self.criterion(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                    self.criterion(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
        loss_w = self.criterion(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                    self.criterion(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
        loss_smooth = loss_w + loss_h

        ### loss_cycle
        Identity = torch.autograd.Variable(torch.eye(w, w).repeat(b, h, 1, 1).to(Q_l.device), requires_grad=False)
        loss_cycle = self.criterion(M_left_right_left * V_left_to_right.permute(0, 2, 1, 3), Identity * V_left_to_right.permute(0, 2, 1, 3)) + \
                        self.criterion(M_right_left_right * V_right_to_left.permute(0, 2, 1, 3), Identity * V_right_to_left.permute(0, 2, 1, 3))

        ### loss_photometric
        LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), LR_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, 3))
        LR_right_warped = LR_right_warped.view(b, h, w, 3).contiguous().permute(0, 3, 1, 2)
        LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, 3))
        LR_left_warped = LR_left_warped.view(b, h, w, 3).contiguous().permute(0, 3, 1, 2)

        loss_photo = self.criterion(LR_left * V_left_to_right, LR_right_warped * V_left_to_right) + \
                        self.criterion(LR_right * V_right_to_left, LR_left_warped * V_right_to_left)

        loss += 0.0025 * (loss_photo + 0.1 * loss_smooth + loss_cycle)
        return x_l + F_r2l, x_r + F_l2r, LR_left, LR_right, loss

class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats
        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.
        r1, r2, r3, r4, r5 = new_feats
        if self.training and factor != 1.:
            r1, r2 = tuple([x+factor*(new_x-x) for x, new_x in zip(feats[:2], new_feats[:2])])
        return r1, r2, r3, r4, r5

class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        new_feats = tuple([self.blk(x) for x in feats[:2]])
        if self.fusion:
            new_feats = self.fusion(*new_feats, *feats[2:])
        return new_feats

class NAFNetSR(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate,
                NAFBlockSR(
                    width,
                    fusion=(fusion_from <= i and i <= fusion_to),
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )

        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale

    def forward(self, inp):
        # print(inp.device)
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        scloss = 0.
        feats = self.body(*feats, *inp, scloss)
        out = torch.cat([self.up(x) for x in feats[:2]], dim=1)
        out = out + inp_hr
        return out, feats[-1]

@ARCH_REGISTRY.register()
class NAFSSRsc(Local_Base, NAFNetSR):
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':
    # num_blks = 128
    # width = 128
    num_blks = 32
    width = 32
    droppath=0.1
    train_size = (1, 6, 30, 90)

    net = NAFSSRsc(up_scale=2,train_size=train_size, fast_imp=True, width=width, num_blks=num_blks, drop_path_rate=droppath)

    inp_shape = (6, 64, 64)
    net(torch.rand(train_size))
    # from ptflops import get_model_complexity_info
    # FLOPS = 0
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # # params = float(params[:-4])
    # print(params)
    # macs = float(macs[:-4]) + FLOPS / 10 ** 9

    # print('mac', macs, params)

    # from basicsr.models.archs.arch_util import measure_inference_speed
    # net = net.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(net, (data,))




