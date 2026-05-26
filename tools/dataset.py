import os
import torch
import torch.nn.functional as F
import h5py
import torch.utils.data as data
import cv2
import numpy as np
# Prefer package-relative import so it works when called from project root (e.g. scripts import `tools.dataset`)
try:
    from . import Pypher  # type: ignore
except Exception:
    # Fallbacks for running this file directly or unusual PYTHONPATH setups
    try:
        from tools import Pypher  # type: ignore
    except Exception:
        import Pypher  # type: ignore

def para_setting(kernel_type, ksize, sz, sigma):
    if kernel_type == 'uniform_blur':
        psf = np.ones([ksize, ksize]) / (ksize * ksize)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(ksize, sigma), (cv2.getGaussianKernel(ksize, sigma)).T)

    fft_B = Pypher.psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B, fft_BT

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, dataset_type):
        super(DatasetFromHdf5, self).__init__()
        # Load once into memory as float32 torch tensors (faster indexing, no per-sample numpy->torch)
        with h5py.File(file_path, 'r') as f:
            print(f.keys())
            keys = set(f.keys())
            self.GT = torch.from_numpy(np.array(f["GT"], dtype=np.float32)).float()
            print(self.GT.shape)  # e.g. (N, C, H, W)
            self.RGB = torch.from_numpy(np.array(f["RGB"], dtype=np.float32)).float()
            self.LRHSI = None

            # Optional fields (some h5 may not contain these)
            # 不用自带的UP和LRHSI，用自己生成的
            # self.UP = None
            # self.LRHSI = None
            # if "HSI_up" in keys:
            #     self.UP = torch.from_numpy(np.array(f["HSI_up"], dtype=np.float32)).float()
            # if "LRHSI" in keys:
            #     self.LRHSI = torch.from_numpy(np.array(f["LRHSI"], dtype=np.float32)).float()

        self.dataset_type = dataset_type
        # ---- infer downsample factor from dataset_type (case-insensitive) ----
        dt = str(self.dataset_type).lower()
        if dt in ("cave_x8", "harvard_x8"):
            self.ds_factor = 8
        elif dt in ("cave_x4", "harvard_x4"):
            self.ds_factor = 4
        elif dt in ("cave_x12", "harvard_x12"):
            self.ds_factor = 12
        elif dt in ("cave_x16", "harvard_x16"):
            self.ds_factor = 16
        elif dt in ("cave_x32", "harvard_x32"):
            self.ds_factor = 32
        else:
            self.ds_factor = None

        if self.ds_factor is not None:
            print('用 H_z 做 “先模糊，再下采样”')
            sigma = 0.5 * self.ds_factor # 你也可以改成自己想要的模糊强度
            H = self.GT.shape[-2]
            W = self.GT.shape[-1]
            sz = [H, W]
            ksize  = 2 * int(np.ceil(3*sigma)) + 1
            fft_B, _ = para_setting('gaussian_blur', ksize, sz, sigma)

            # fft_B: numpy complex (H, W) -> torch float (H, W, 2)  [real, imag]
            # self.fft_B = torch.stack(
            #     (torch.from_numpy(np.real(fft_B)).float(),
            #      torch.from_numpy(np.imag(fft_B)).float()),
            #     dim=-1
            # )  # (H, W, 2)
            Bc = torch.from_numpy(fft_B.copy()).to(torch.complex64)
            self.Bc = Bc[None, None, ...]

            # Precompute LRHSI from GT once (fast __getitem__)
            print("[Precompute] building LRHSI from GT via H_z ...")
            N = self.GT.shape[0]
            with torch.no_grad():
                lr0 = self.H_z(self.GT[0], self.ds_factor, self.Bc)  # (C,h,w)
                C, h_lr, w_lr = lr0.shape
                lr_all = torch.empty((N, C, h_lr, w_lr), dtype=torch.float32)
                lr_all[0] = lr0
                for i in range(1, N):
                    lr_all[i] = self.H_z(self.GT[i], self.ds_factor, self.Bc)
                    if i % 200 == 0:
                        print(f"[Precompute] {i}/{N} done")

            self.LRHSI = lr_all
            print(f"[Precompute] LRHSI overwritten: shape={tuple(self.LRHSI.shape)}")
        else:
            # If we can't infer ds_factor, we must rely on LRHSI provided by the h5 file.
            if self.LRHSI is None:
                raise KeyError(
                    f"Missing 'LRHSI' in h5 and cannot infer ds_factor from dataset_type={self.dataset_type!r}."
                )

        # Ensure UP exists: use provided HSI_up if present; otherwise generate by upsampling LRHSI.
        # if self.UP is None:
        #     # LRHSI: (N,C,h,w) -> UP: (N,C,H,W)
        #     self.UP = F.interpolate(
        #         self.LRHSI,
        #         size=self.GT.shape[-2:],
        #         mode='bicubic',
        #         align_corners=False
        #     )
            
            
    def H_z(self, z, factor, Bc):
        """
        z: (C,H,W) 或 (B,C,H,W) 或 (H,W,C) —— 都尽量兼容
        Bc: (1,1,H,W) complex
        返回：下采样后的实部
        """
        # 统一到 (B,C,H,W)
        restore_hwc = False
        if z.dim() == 3:
            # (H,W,C) 情况：把它转成 (C,H,W)
            if z.shape[-1] in (31, 32) and z.shape[0] == z.shape[1]:
                z = z.permute(2, 0, 1).contiguous()
                restore_hwc = True
            z = z.unsqueeze(0)  # (1,C,H,W)
        elif z.dim() != 4:
            raise ValueError(f"Unexpected z shape: {tuple(z.shape)}")

        # B, C, H, W = z.shape
        # Bc = torch.complex(fft_B[..., 0], fft_B[..., 1])  # (H,W) complex
        # Bc = Bc.view(1, 1, H, W)  # broadcast to (B,C,H,W)

        f = torch.fft.fft2(z, dim=(-2, -1))      # (B,C,H,W) complex
        Hz = torch.fft.ifft2(f * Bc, dim=(-2, -1))  # 卷积后回空间域

        # start = int(factor // 2) - 1
        start = 0
        x = Hz[:, :, start::factor, start::factor]  # 关键：stride 抽取 = 下采样
        x = x.real  # 只取实部

        # 如果原来是 HWC，就转回 HWC（可选）
        if restore_hwc:
            x = x.squeeze(0).permute(1, 2, 0).contiguous()  # (h,w,c)
            return x
        else:
            return x.squeeze(0)  # (C,h,w)

    #####必要函数
    def __getitem__(self, index):
        input_rgb = self.RGB[index]
        input_lr = self.LRHSI[index]
        # input_lr_u = self.UP[index]
        target = self.GT[index]

        # return input_rgb, input_lr, input_lr_u, target
        return input_rgb, input_lr, target

    def __len__(self):
        return self.GT.shape[0]
