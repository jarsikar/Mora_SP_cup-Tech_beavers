#!/usr/bin/env python3
"""
denoise.py - Official Denoising Script for Mora SP Cup 2026
-----------------------------------------------------------
Championship Grand Ensemble Pipeline (Score: ~0.4046):
- Stage 1: Defect / hot pixel removal (Local median filter)
- Stage 2: Classical Multi-scale Wavelet shrinkage (BayesShrink in YCbCr) + Collaborative NLM
- Stage 3: Deep Residual Neural Network (Lightweight UNet)
- Stage 4: Optimal Dual-Domain Ensemble (50% UNet + 50% Classical DSP)
- Seamless CPU fallback: Fully functional offline on any mid-range CPU

Usage:
    python scripts/denoise.py --noise_dir <input_dir> --denoised_dir <output_dir>
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_wavelet

# Add scripts directory to path for local imports
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NOISE_SUFFIX = "_noise"
MODEL_CHECKPOINT_PATH = SCRIPTS_DIR / "best_model.pth"

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def correct_defect_pixels(img_float: np.ndarray, threshold: float = 0.20, kernel_size: int = 3) -> np.ndarray:
    """Removes extreme impulse / defect / hot pixels using median outlier rejection."""
    corrected = img_float.copy()
    for c in range(img_float.shape[2]):
        channel_u8 = (img_float[..., c] * 255).astype(np.uint8)
        median_u8 = cv2.medianBlur(channel_u8, kernel_size)
        median = median_u8.astype(np.float32) / 255.0
        deviation = np.abs(img_float[..., c] - median)
        outlier_mask = deviation > threshold
        corrected[..., c] = np.where(outlier_mask, median, img_float[..., c])
    return corrected


def denoise_classical_stage(img_float: np.ndarray) -> np.ndarray:
    """Wavelet BayesShrink soft-thresholding in YCbCr followed by collaborative NLM."""
    wav = denoise_wavelet(
        img_float,
        method="BayesShrink",
        mode="soft",
        wavelet="sym4",
        channel_axis=-1,
        convert2ycbcr=True,
        rescale_sigma=True,
    )
    wav_u8 = (np.clip(wav, 0.0, 1.0) * 255).astype(np.uint8)
    nlm_u8 = cv2.fastNlMeansDenoisingColored(wav_u8, None, 8.0, 12.0, 7, 21)
    return nlm_u8.astype(np.float32) / 255.0


def load_neural_model(checkpoint_path: Path):
    if not TORCH_AVAILABLE or not checkpoint_path.exists():
        return None, None

    try:
        from model import build_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_name = ckpt.get("model_name", "unet")
        model = build_model(model_name).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, device
    except Exception as err:
        print(f"[Warning] Failed to load neural model: {err}. Using classical pipeline.")
        return None, None


def denoise_neural_stage(img_float: np.ndarray, model, device) -> np.ndarray:
    """Inference through lightweight neural denoiser."""
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                out_tensor = model(tensor)
        else:
            out_tensor = model(tensor)

    out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out_np, 0.0, 1.0)


def strip_noise_suffix(stem: str) -> str:
    if stem.lower().endswith(NOISE_SUFFIX):
        return stem[:-len(NOISE_SUFFIX)]
    return stem


def main():
    parser = argparse.ArgumentParser(description="Mora SP Cup 2026 Grand Ensemble Denoiser")
    parser.add_argument("--noise_dir", "--input_dir", dest="input_dir", required=True, type=Path,
                        help="Path to folder of noisy input images")
    parser.add_argument("--denoised_dir", "--output_dir", dest="output_dir", required=True, type=Path,
                        help="Path to folder where denoised images will be saved")
    parser.add_argument("--model_path", default=MODEL_CHECKPOINT_PATH, type=Path,
                        help="Path to neural model weights (.pth)")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in args.input_dir.glob("*") if p.suffix.lower() in VALID_EXTENSIONS])

    if not image_paths:
        raise SystemExit(f"No valid image files found in {args.input_dir}")

    # Load Neural Model
    model, device = load_neural_model(args.model_path)
    if model is not None and args.force_cpu:
        device = torch.device("cpu")
        model = model.to(device)

    mode = "Grand Ensemble (50% Classical DSP + 50% Neural UNet)" if model is not None else "Pure Classical DSP (Wavelet + NLM)"
    print("=" * 65)
    print(f"PIPELINE: {mode}")
    print(f"Device:   {device if model is not None else 'CPU'}")
    print(f"Inputs:   {args.input_dir}")
    print(f"Outputs:  {args.output_dir}")
    print(f"Total:    {len(image_paths)} images")
    print("=" * 65)

    total_start = time.time()
    for idx, path in enumerate(image_paths, 1):
        img_start = time.time()
        with Image.open(path) as raw_img:
            img_rgb = raw_img.convert("RGB")
            img_float = np.asarray(img_rgb, dtype=np.float32) / 255.0

        # Stage 1: Defect outlier removal
        cleaned = correct_defect_pixels(img_float, threshold=0.20, kernel_size=3)

        # Stage 2: Classical Wavelet + NLM
        classical_out = denoise_classical_stage(cleaned)

        if model is not None:
            # Stage 3: Neural Denoiser
            neural_out = denoise_neural_stage(cleaned, model, device)
            # Stage 4: Dual-domain Grand Ensemble
            final_out = 0.50 * classical_out + 0.50 * neural_out
        else:
            final_out = classical_out

        out_u8 = (np.clip(final_out, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        out_id = strip_noise_suffix(path.stem)
        out_path = args.output_dir / f"{out_id}.png"
        Image.fromarray(out_u8).save(out_path, format="PNG")

        elapsed = time.time() - img_start
        print(f"[{idx:02d}/{len(image_paths):02d}] {path.name} -> {out_path.name} ({elapsed:.2f}s)")

    total_time = time.time() - total_start
    print("=" * 65)
    print(f"All {len(image_paths)} images processed in {total_time:.2f}s ({total_time/len(image_paths):.2f}s/image).")
    print("=" * 65)


if __name__ == "__main__":
    main()
