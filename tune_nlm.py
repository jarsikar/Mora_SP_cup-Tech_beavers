"""
Day 3: Find the best single 'h' value for Non-Local Means denoising.
Reuses the baseline's own functions so results are directly comparable.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Let Python find and import functions from baseline/denoise.py
sys.path.append(str(Path(__file__).parent / "baseline"))
from denoise import correct_defect_pixels, denoise_nlm

GT_DIR = Path("competition_data/public/ground_truth")
NOISY_DIR = Path("competition_data/public/noisy")

# Testing all 460 images for every h value would be slow.
# Every 15th image gives a spread-out sample of ~31 images -- fast to test,
# still representative since it's spread across the whole dataset.
all_ids = [f"{i:03d}" for i in range(1, 461)]
val_ids = all_ids[::15]
print(f"Validating on {len(val_ids)} images\n")


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def composite_score(noisy, pred, gt):
    """Reproduces the exact formula from evaluation/evaluate.py"""
    p_noisy = psnr(gt, noisy, data_range=1.0)
    p_pred = psnr(gt, pred, data_range=1.0)
    s_noisy = ssim(gt, noisy, data_range=1.0, channel_axis=2)
    s_pred = ssim(gt, pred, data_range=1.0, channel_axis=2)

    delta_psnr = p_pred - p_noisy
    delta_ssim = s_pred - s_noisy

    n = np.clip(delta_psnr / 15.0, 0, 1)
    s = max(delta_ssim, 0)
    return 0.6 * n + 0.4 * s


# Load all validation images into memory once -- avoids re-reading
# the same files from disk for every h value we test
images = []
for sid in val_ids:
    clean = load_image(GT_DIR / f"{sid}.png")
    noisy = load_image(NOISY_DIR / f"{sid}_noise.png")
    images.append((sid, clean, noisy))

# The values we'll test -- spread around the baseline's default of 10
h_candidates = [4, 6, 8, 10, 12, 14, 16, 20]
results = {}

for h in h_candidates:
    scores = []
    for sid, clean, noisy in images:
        corrected = correct_defect_pixels(noisy, threshold=0.25)
        img_uint8 = (corrected * 255).astype(np.uint8)
        denoised_uint8 = denoise_nlm(img_uint8, h=h, h_color=h)
        denoised = denoised_uint8.astype(np.float32) / 255.0
        scores.append(composite_score(noisy, denoised, clean))

    avg_score = np.mean(scores)
    results[h] = avg_score
    print(f"h={h:>4}: composite score = {avg_score:.4f}")

best_h = max(results, key=results.get)
print(f"\nBest h found: {best_h} (score={results[best_h]:.4f})")