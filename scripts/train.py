"""
train.py - GPU Training Pipeline for Mora SP Cup 2026
------------------------------------------------------
Trains lightweight neural denoiser on RTX 5060 Laptop GPU:
- Mixed Precision (FP16/AMP) for 2x faster training
- Combined L1 + SSIM loss
- Validation checkpointing based on Mora SP Cup Composite Score
- Full training progress logging
"""

import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from model import build_model
from dataset import DenoisePatchDataset, CombinedL1SSIMLoss

GT_DIR = Path("competition_data/public/ground_truth")
NOISY_DIR = Path("competition_data/public/noisy")


def composite_metric(pred: np.ndarray, gt: np.ndarray, noisy: np.ndarray):
    p_pred = sk_psnr(gt, pred, data_range=1.0)
    p_noisy = sk_psnr(gt, noisy, data_range=1.0)
    s_pred = sk_ssim(gt, pred, data_range=1.0, channel_axis=2, win_size=7)
    s_noisy = sk_ssim(gt, noisy, data_range=1.0, channel_axis=2, win_size=7)

    delta_psnr = p_pred - p_noisy
    delta_ssim = s_pred - s_noisy

    n = np.clip(delta_psnr / 15.0, 0.0, 1.0)
    s = max(delta_ssim, 0.0)
    return 0.6 * n + 0.4 * s, delta_psnr, delta_ssim


def evaluate_val_set(model: torch.nn.Module, val_ids: list, device: torch.device):
    model.eval()
    scores, d_psnrs, d_ssims = [], [], []

    with torch.no_grad():
        for sid in val_ids:
            from PIL import Image
            gt = np.asarray(Image.open(GT_DIR / f"{sid}.png").convert("RGB"), dtype=np.float32) / 255.0
            noisy = np.asarray(Image.open(NOISY_DIR / f"{sid}_noise.png").convert("RGB"), dtype=np.float32) / 255.0

            # Tensor conversion
            tensor_noisy = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                pred_tensor = model(tensor_noisy)

            pred = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)

            score, dp, ds = composite_metric(pred, gt, noisy)
            scores.append(score)
            d_psnrs.append(dp)
            d_ssims.append(ds)

    return float(np.mean(scores)), float(np.mean(d_psnrs)), float(np.mean(d_ssims))


def main():
    parser = argparse.ArgumentParser(description="Train Neural Denoiser")
    parser.add_argument("--model", default="unet", choices=["unet", "dncnn"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch crop size")
    parser.add_argument("--patches_per_image", type=int, default=16, help="Patches extracted per image per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--val_split", type=int, default=60, help="Number of validation images")
    parser.add_argument("--save_path", default="scripts/best_model.pth", help="Path to save best checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"TRAINING INITIALIZATION: {args.model.upper()}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Patch Size: {args.patch_size}")
    print("=" * 60)

    # 460 images split into 400 train and 60 val
    all_ids = [f"{i:03d}" for i in range(1, 461)]
    # Fixed deterministic split: take every ~7th image for 60 val, rest 400 for train
    val_ids = all_ids[::len(all_ids) // args.val_split][:args.val_split]
    train_ids = [sid for sid in all_ids if sid not in val_ids]

    print(f"Training images: {len(train_ids)} | Validation images: {len(val_ids)}")

    train_dataset = DenoisePatchDataset(
        noisy_dir=NOISY_DIR,
        gt_dir=GT_DIR,
        image_ids=train_ids,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        is_training=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows-friendly
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {param_count:,}")

    criterion = CombinedL1SSIMLoss(alpha=0.4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_score = 0.0
    start_all = time.time()

    print("\nStarting Training Loop...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (noisy, gt) in enumerate(train_loader, 1):
            noisy = noisy.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                pred = model(noisy)
                loss = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Validation every epoch
        val_score, val_dp, val_ds = evaluate_val_set(model, val_ids, device)

        is_best = val_score > best_score
        if is_best:
            best_score = val_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "composite_score": best_score,
                "delta_psnr": val_dp,
                "delta_ssim": val_ds,
                "model_name": args.model,
            }, args.save_path)
            star = " * [BEST SAVED]"
        else:
            star = ""

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Comp Score: {val_score:.4f} | "
              f"dPSNR: {val_dp:+.2f}dB | dSSIM: {val_ds:+.4f}{star}", flush=True)

    total_time = time.time() - start_all
    print("=" * 60)
    print(f"TRAINING COMPLETE in {total_time/60:.2f} minutes!")
    print(f"Best Validation Composite Score: {best_score:.4f}")
    print(f"Best model saved to: {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
