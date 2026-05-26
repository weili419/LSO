import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
try:
    from model.edsr import make_edsr_baseline
except ModuleNotFoundError:
    # Allows running this file directly: `python model/LSO.py`
    import os
    import sys

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from model.edsr import make_edsr_baseline


################################################################
# Multiscale modules 2D
################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Upsample(mode='bilinear',scale_factor=1/2),
            # nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        in_channels = in_channels + 3

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1, x2, rgb):
        x1 = self.up(x1)
        x = torch.cat([x1, x2, rgb], dim=1)
        
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),

        )
    def forward(self, x):
        return self.conv(x)


################################################################
# Patchify and Neural Spectral Block
################################################################
class NeuralSpectralBlock2d(nn.Module):
    def __init__(self, width, num_basis, patch_size=[3, 3], num_token=4):
        super(NeuralSpectralBlock2d, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # # basis - 注册为 buffer，自动移动到正确设备
        # self.register_buffer('modes_list', 
        #                     (1.0 / float(num_basis)) * torch.tensor([i for i in range(num_basis)],
        #                                                             dtype=torch.float))
        self.register_buffer('modes_list', torch.arange(1, num_basis+1, dtype=torch.float))
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float))
        # latent
        self.head = 8
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width)) * torch.rand(self.head, self.num_token, width // self.head, dtype=torch.float))
        self.encoder_attn = nn.Conv2d(self.width, self.width * 2, kernel_size=1, stride=1)
        self.decoder_attn = nn.Conv2d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        d = q.shape[-1]
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k) / math.sqrt(d))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        # x: B C H W
        B, C, H, W = x.shape
        L = H * W
        latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        x_tmp = self.encoder_attn(x).view(B, C * 2, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head, 2).permute(4, 0, 2, 1, 3).contiguous()
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C L
        x_init = x
        B, C, H, W = x.shape
        L = H * W
        latent_token = latent_token.view(B, self.head, C // self.head, self.num_token).permute(0, 1, 3, 2).contiguous()
        x_tmp = self.decoder_attn(x).view(B, C, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head).permute(0, 2, 1, 3).contiguous()
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, H, W) + x_init  # B H L C/H
        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        x_cos = torch.cos(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        res = x
        B, C, H, W = x.shape
        # 检查 patch_size 是否能整除 H 和 W
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            # 如果不能整除，进行 padding
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, [0, pad_w, 0, pad_h])
            H_padded = H + pad_h
            W_padded = W + pad_w
        else:
            H_padded, W_padded = H, W
        
        # patchify
        x = x.view(B, C,
                   H_padded // self.patch_size[0], self.patch_size[0], 
                   W_padded // self.patch_size[1], self.patch_size[1]).contiguous() \
            .permute(0, 2, 4, 1, 3, 5).contiguous() \
            .view(B * (H_padded // self.patch_size[0]) * (W_padded // self.patch_size[1]), C,
                  self.patch_size[0], self.patch_size[1])
        # Neural Spectral
        # (1) encoder
        latent_token = self.latent_encoder_attn(x)
        # (2) transition
        latent_token = torch.sigmoid(latent_token) * math.pi
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)
        # de-patchify
        x = x.view(B, (H_padded // self.patch_size[0]), (W_padded // self.patch_size[1]), C, 
                   self.patch_size[0], self.patch_size[1]).permute(0, 3, 1, 4, 2, 5).contiguous() \
            .view(B, C, H_padded, W_padded).contiguous()
        
        # 如果进行了 padding，需要裁剪回原始尺寸
        if H_padded != H or W_padded != W:
            x = x[:, :, :H, :W]
        
        return x


class LSOModel(nn.Module):
    def __init__(self, args=None, bilinear=True, n_select_bands=3, n_bands=31, sf= 4):
        super(LSOModel, self).__init__()
        
        patch_size = [args.patch_size, args.patch_size]
        width = args.width 
        num_token = args.num_token
        num_basis = args.num_basis
        # 添加缺失的参数
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        
        self.guide_dim = args.guide_dim # 128
        self.n_resblocks = args.n_resblocks # 4

        self.sf = sf # 4
        
        # multiscale modules - 只保留2层下采样
        self.inc = DoubleConv(self.guide_dim+2, width) # lift 

        self.down1 = Down(width, width)
        self.down2 = Down(width, width)
        # self.down3 = Down(width, width)

        # Up 的 in_channels 应该是 concat 后的通道数：x1_channels + x2_channels
        # up1: process4(x4) [512] + process3(x3) [256] = 768 -> 256
        self.up1 = Up(width + width, width)  # 从 x4 开始上采样
        # up2: up1输出 [256] + process2(x2) [128] = 384 -> 128
        self.up2 = Up(width + width, width)
        # up3: up2输出 [128] + process1(x1) [64] = 192 -> 64
        # self.up3 = Up(width + width, width)

        self.outc = OutConv(width, self.n_bands)

        # Patchified Neural Spectral Blocks - 删除 process5
        self.process1 = NeuralSpectralBlock2d(width, num_basis, patch_size, num_token)
        self.process2 = NeuralSpectralBlock2d(width, num_basis, patch_size, num_token)
        self.process3 = NeuralSpectralBlock2d(width, num_basis, patch_size, num_token)
        # self.process4 = NeuralSpectralBlock2d(width, num_basis, patch_size, num_token)

        # spatial encoder - 修复参数 # 默认就是 n_resblocks = 4, n_feats = 128
        self.spatial_encoder = make_edsr_baseline(n_resblocks=self.n_resblocks, n_feats=self.guide_dim, n_colors=self.n_select_bands + self.n_bands)
        

    def forward(self, HR_MSI, LR_HSI):
        # print(HR_MSI.shape)
        # print(lms.shape)
        # print(LR_HSI.shape)
        lms = F.interpolate(LR_HSI, scale_factor=self.sf, mode='bicubic', align_corners=False)

        feat = torch.cat([HR_MSI, lms], dim=1)
        x = self.spatial_encoder(feat)  # [B, guide_dim, H, W]
        B, C, H, W = x.shape
        # 生成坐标网格
        grid = self.get_grid((B, H, W), x.device)  # [B, H, W, 2]
        grid = grid.permute(0, 3, 1, 2)
        x = torch.cat((x, grid), dim=1)
        
        x1 = self.inc(x) # width

        x2 = self.down1(x1)
        x3 = self.down2(x2)

        HR_MSI2 = F.interpolate(HR_MSI, scale_factor=0.5, mode='bicubic', align_corners=False)
        
        NS_x1 = self.process1(x1)
        NS_x2 = self.process2(x2)
        NS_x3 = self.process3(x3)

        y = self.up1(NS_x3, NS_x2, HR_MSI2)+x2
        y = self.up2(y, NS_x1, HR_MSI)+x1
        
        resnet = self.outc(y)

        return lms + resnet

    def get_grid(self, shape, device):
        """生成坐标网格 [B, H, W, 2] - 优化版本，直接在 GPU 上创建"""
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        # 直接在 GPU 上创建，避免 CPU-GPU 传输
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.view(1, size_x, 1, 1).expand(batchsize, -1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
        gridy = gridy.view(1, 1, size_y, 1).expand(batchsize, size_x, -1, 1)
        return torch.cat((gridx, gridy), dim=-1)

# Compatibility alias for code that imports Model from this module.
Model = LSOModel
