"""
tune_classical.py
-----------------
Explores advanced classical signal processing techniques beyond the baseline:
1. Defect-pixel threshold tuning
2. Decoupled Luminance/Chrominance filtering (Lab / YCrCb)
3. Multi-scale Wavelet Denoising (BayesShrink / VisuShrink)
4. Combined Wavelet + NLM
"""

import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_wavelet

GT_DIR = Path("competition_data/public/ground_truth")
NOISY_DIR = Path("competition_data/public/noisy")

# Representative 31 validation images across the 460 dataset
all_ids = [f"{i:03d}" for i in range(1, 461)]
val_ids = all_ids[::15]


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def composite_score(noisy, pred, gt):
    p_noisy = psnr(gt, noisy, data_range=1.0)
    p_pred = psnr(gt, pred, data_range=1.0)
    s_noisy = ssim(gt, noisy, data_range=1.0, channel_axis=2, win_size=7)
    s_pred = ssim(gt, pred, data_range=1.0, channel_axis=2, win_size=7)

    delta_psnr = p_pred - p_noisy
    delta_ssim = s_pred - s_noisy

    n = np.clip(delta_psnr / 15.0, 0.0, 1.0)
    s = max(delta_ssim, 0.0)
    return 0.6 * n + 0.4 * s, delta_psnr, delta_ssim


def correct_defect_pixels(img_float: np.ndarray, threshold: float = 0.25, kernel_size: int = 3) -> np.ndarray:
    corrected = img_float.copy()
    for c in range(img_float.shape[2]):
        channel_uint8 = (img_float[..., c] * 255).astype(np.uint8)
        median_uint8 = cv2.medianBlur(channel_uint8, kernel_size)
        median = median_uint8.astype(np.float32) / 255.0
        deviation = np.abs(img_float[..., c] - median)
        outlier_mask = deviation > threshold
        corrected[..., c] = np.where(outlier_mask, median, img_float[..., c])
    return corrected


def denoise_wavelet_rgb(img_float: np.ndarray, method: str = "BayesShrink", wavelet: str = "db4") -> np.ndarray:
    return denoise_wavelet(
        img_float,
        method=method,
        mode="soft",
        wavelet=wavelet,
        channel_axis=-1,
        convert2ycbcr=True,
        rescale_sigma=True,
    )


def run_benchmarks():
    print(f"Loading {len(val_ids)} validation images...", flush=True)
    images = []
    for sid in val_ids:
        clean = load_image(GT_DIR / f"{sid}.png")
        noisy = load_image(NOISY_DIR / f"{sid}_noise.png")
        images.append((sid, clean, noisy))

    print(f"Loaded {len(images)} images successfully.\n", flush=True)

    experiments = [
        (
            "Baseline (defect th=0.25, NLM h=10, h_color=10)",
            lambda noisy: cv2.fastNlMeansDenoisingColored(
                (correct_defect_pixels(noisy, 0.25) * 255).astype(np.uint8), None, 10, 10, 7, 21
            ).astype(np.float32) / 255.0,
        ),
        (
            "NLM Tuned (defect th=0.25, NLM h=16, h_color=16)",
            lambda noisy: cv2.fastNlMeansDenoisingColored(
                (correct_defect_pixels(noisy, 0.25) * 255).astype(np.uint8), None, 16, 16, 7, 21
            ).astype(np.float32) / 255.0,
        ),
        (
            "NLM Decoupled Chroma (defect th=0.20, h_luma=13, h_chroma=20)",
            lambda noisy: cv2.fastNlMeansDenoisingColored(
                (correct_defect_pixels(noisy, 0.20) * 255).astype(np.uint8), None, 13, 20, 7, 21
            ).astype(np.float32) / 255.0,
        ),
        (
            "NLM Decoupled Chroma (defect th=0.20, h_luma=14, h_chroma=22)",
            lambda noisy: cv2.fastNlMeansDenoisingColored(
                (correct_defect_pixels(noisy, 0.20) * 255).astype(np.uint8), None, 14, 22, 7, 21
            ).astype(np.float32) / 255.0,
        ),
        (
            "Wavelet BayesShrink (db4, convert2ycbcr=True)",
            lambda noisy: denoise_wavelet_rgb(correct_defect_pixels(noisy, 0.20), method="BayesShrink", wavelet="db4"),
        ),
        (
            "Wavelet VisuShrink (db4, convert2ycbcr=True)",
            lambda noisy: denoise_wavelet_rgb(correct_defect_pixels(noisy, 0.20), method="VisuShrink", wavelet="db4"),
        ),
        (
            "Hybrid Wavelet + Soft NLM (Wavelet db4 -> NLM h=8, h_color=12)",
            lambda noisy: (
                cv2.fastNlMeansDenoisingColored(
                    (np.clip(denoise_wavelet_rgb(correct_defect_pixels(noisy, 0.20), method="BayesShrink", wavelet="db4"), 0, 1) * 255).astype(np.uint8),
                    None, 8, 12, 7, 21
                ).astype(np.float32) / 255.0
            ),
        ),
    ]

    print(f"{'Method':<55} | {'Comp Score':<10} | {'dPSNR (dB)':<10} | {'dSSIM':<10} | {'Time/img (s)':<12}", flush=True)
    print("-" * 105, flush=True)

    for name, fn in experiments:
        scores = []
        d_psnrs = []
        d_ssims = []
        start_time = time.time()
        for sid, clean, noisy in images:
            pred = np.clip(fn(noisy), 0.0, 1.0)
            score, dp, ds = composite_score(noisy, pred, clean)
            scores.append(score)
            d_psnrs.append(dp)
            d_ssims.append(ds)
        elapsed = (time.time() - start_time) / len(images)
        mean_score = np.mean(scores)
        mean_dp = np.mean(d_psnrs)
        mean_ds = np.mean(d_ssims)
        print(f"{name:<55} | {mean_score:<10.4f} | {mean_dp:<+10.4f} | {mean_ds:<+10.4f} | {elapsed:<12.3f}", flush=True)


if __name__ == "__main__":
    run_benchmarks()
