"""
dataset.py - Dataset and Loss Functions for Mora SP Cup 2026
------------------------------------------------------------
Implements:
1. DenoisePatchDataset: Memory-efficient random patch extraction with data augmentation
2. SSIMLoss / CombinedL1SSIMLoss: Loss aligning with the competition evaluation metric
"""

import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DenoisePatchDataset(Dataset):
    """
    Dataset that extracts random cropped patches from noisy & ground-truth image pairs.
    Includes data augmentation (random horizontal/vertical flips, 90-degree rotations).
    """
    def __init__(self,
                 noisy_dir: Path,
                 gt_dir: Path,
                 image_ids: List[str],
                 patch_size: int = 128,
                 patches_per_image: int = 64,
                 is_training: bool = True):
        self.noisy_dir = Path(noisy_dir)
        self.gt_dir = Path(gt_dir)
        self.image_ids = image_ids
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.image_ids) * self.patches_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx = idx // self.patches_per_image
        image_id = self.image_ids[img_idx]

        noisy_path = self.noisy_dir / f"{image_id}_noise.png"
        gt_path = self.gt_dir / f"{image_id}.png"

        # Load images as float32 in [0, 1]
        with Image.open(noisy_path) as n_img, Image.open(gt_path) as g_img:
            noisy = np.asarray(n_img.convert("RGB"), dtype=np.float32) / 255.0
            gt = np.asarray(g_img.convert("RGB"), dtype=np.float32) / 255.0

        h, w, _ = noisy.shape

        if self.is_training:
            # Random crop
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)

            p_noisy = noisy[top:top + self.patch_size, left:left + self.patch_size]
            p_gt = gt[top:top + self.patch_size, left:left + self.patch_size]

            # Augmentation: Random Flips & Rotations
            if random.random() > 0.5:
                p_noisy = np.fliplr(p_noisy)
                p_gt = np.fliplr(p_gt)
            if random.random() > 0.5:
                p_noisy = np.flipud(p_noisy)
                p_gt = np.flipud(p_gt)

            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                p_noisy = np.rot90(p_noisy, rot)
                p_gt = np.rot90(p_gt, rot)
        else:
            # Deterministic center crop for validation patches
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2
            p_noisy = noisy[top:top + self.patch_size, left:left + self.patch_size]
            p_gt = gt[top:top + self.patch_size, left:left + self.patch_size]

        # Convert to Channel-First Tensor (C, H, W)
        tensor_noisy = torch.from_numpy(p_noisy.copy()).permute(2, 0, 1)
        tensor_gt = torch.from_numpy(p_gt.copy()).permute(2, 0, 1)

        return tensor_noisy, tensor_gt


def gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor([
        np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def ssim_tensor(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int, channel: int) -> torch.Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class CombinedL1SSIMLoss(nn.Module):
    """
    Combined Loss: L1 + alpha * (1 - SSIM)
    Directly maximizes PSNR while preserving structural edge integrity.
    """
    def __init__(self, alpha: float = 0.4, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", create_window(window_size, channels))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = F.l1_loss(pred, target)
        ssim_val = ssim_tensor(pred, target, self.window, self.window_size, self.channels)
        return (1.0 - self.alpha) * l1_loss + self.alpha * (1.0 - ssim_val)
