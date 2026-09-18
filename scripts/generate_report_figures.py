"""
generate_report_figures.py
--------------------------
Generates publication-quality figures for the 5-page competition report:
- Fig 1: Noise characterization (Poisson-Gaussian intensity dependence & color asymmetry)
- Fig 2: Complete Grand Ensemble Architecture Diagram
- Fig 3: Side-by-side visual denoising comparison patches
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

OUT_DIR = Path("scripts/report_assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# -------------------------------------------------------------
# Figure 1: Noise Profiling
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2), dpi=300)

# Intensity Bins
bins = ["0-50\n(Dark)", "51-101\n(Mid-Dark)", "102-152\n(Mid)", "153-203\n(Mid-Bright)", "204-255\n(Bright)"]
noise_vals = [14.36, 20.95, 23.85, 25.29, 21.88]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

bars1 = ax1.bar(bins, noise_vals, color='#2b5c8f', width=0.55, edgecolor='black', linewidth=0.7)
ax1.set_title("(a) Noise Magnitude vs. Pixel Intensity", fontsize=10, fontweight='bold', pad=8)
ax1.set_ylabel("Mean Absolute Noise (0-255)", fontsize=9)
ax1.set_ylim(0, 30)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, y + 0.7, f"{y:.1f}", ha='center', fontsize=8, fontweight='bold')

# Color Channels
channels = ["Red Channel", "Green Channel", "Blue Channel"]
ch_vals = [20.55, 19.69, 19.27]
ch_colors = ['#e74c3c', '#2ecc71', '#3498db']

bars2 = ax2.bar(channels, ch_vals, color=ch_colors, width=0.45, edgecolor='black', linewidth=0.7)
ax2.set_title("(b) Noise Magnitude by Color Channel", fontsize=10, fontweight='bold', pad=8)
ax2.set_ylabel("Mean Absolute Noise (0-255)", fontsize=9)
ax2.set_ylim(0, 26)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, y + 0.6, f"{y:.2f}", ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
fig1_path = OUT_DIR / "fig1_noise_profiling.png"
plt.savefig(fig1_path, bbox_inches='tight')
plt.close()
print(f"Saved {fig1_path}")

# -------------------------------------------------------------
# Figure 2: Architecture Block Diagram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
ax.axis('off')

# Box styles
box_kw = dict(boxstyle="round,pad=0.4", ec="#1f4e78", fc="#e9f1f7", lw=1.5)
dsp_kw = dict(boxstyle="round,pad=0.4", ec="#2e7d32", fc="#e8f5e9", lw=1.5)
nn_kw  = dict(boxstyle="round,pad=0.4", ec="#c62828", fc="#ffebee", lw=1.5)
ens_kw = dict(boxstyle="round,pad=0.4", ec="#6a1b9a", fc="#f3e5f5", lw=1.8)

# Nodes
ax.text(0.08, 0.50, "Input Image\n$I_{\\mathrm{noisy}} \\in \\mathbb{R}^{992\\times 992\\times 3}$", ha='center', va='center', bbox=box_kw, fontsize=8.5, fontweight='bold')
ax.text(0.28, 0.50, "Stage 1:\nDefect-Pixel Filter\n(Median Outlier)", ha='center', va='center', bbox=box_kw, fontsize=8.5)

# Branch 1: Classical
ax.text(0.53, 0.78, "Stage 2A: Classical DSP Branch\nWavelet Shrinkage ($YC_bC_r$, BayesShrink)\n+ Collaborative NLM Filtering", ha='center', va='center', bbox=dsp_kw, fontsize=8)

# Branch 2: Neural
ax.text(0.53, 0.22, "Stage 2B: Deep Learning Branch\nLightweight Residual UNet (~1.29M)\n$\hat{I}_{\\mathrm{clean}} = I_{\\mathrm{noisy}} - \\mathcal{R}_\\theta(I)$", ha='center', va='center', bbox=nn_kw, fontsize=8)

# Ensemble Node
ax.text(0.78, 0.50, "Stage 3: Grand Ensemble\n$\hat{I}_{\\mathrm{final}} = 0.5\\hat{I}_{\\mathrm{DSP}} + 0.5\\hat{I}_{\\mathrm{UNet}}$", ha='center', va='center', bbox=ens_kw, fontsize=8.5, fontweight='bold')

# Output
ax.text(0.94, 0.50, "Denoised\nOutput\n$\hat{I}_{\\mathrm{final}}$", ha='center', va='center', bbox=box_kw, fontsize=8.5, fontweight='bold')

# Arrows
arr = dict(arrowstyle="->", lw=1.5, color="#333333")
ax.annotate("", xy=(0.19, 0.50), xytext=(0.17, 0.50), arrowprops=arr)
ax.annotate("", xy=(0.38, 0.78), xytext=(0.35, 0.53), arrowprops=arr)
ax.annotate("", xy=(0.38, 0.22), xytext=(0.35, 0.47), arrowprops=arr)
ax.annotate("", xy=(0.71, 0.53), xytext=(0.68, 0.75), arrowprops=arr)
ax.annotate("", xy=(0.71, 0.47), xytext=(0.68, 0.25), arrowprops=arr)
ax.annotate("", xy=(0.89, 0.50), xytext=(0.85, 0.50), arrowprops=arr)

plt.tight_layout()
fig2_path = OUT_DIR / "fig2_architecture.png"
plt.savefig(fig2_path, bbox_inches='tight')
plt.close()
print(f"Saved {fig2_path}")

# -------------------------------------------------------------
# Figure 3: Qualitative Visual Comparison (Sample 016 Zoomed-in Crop)
# -------------------------------------------------------------
gt_img = np.array(Image.open("competition_data/public/ground_truth/016.png").convert("RGB"))
noisy_img = np.array(Image.open("competition_data/public/noisy/016_noise.png").convert("RGB"))

# Crop interesting region (300:600, 300:600)
cy, cx, sz = 400, 400, 200
crop_gt = gt_img[cy:cy+sz, cx:cx+sz]
crop_noisy = noisy_img[cy:cy+sz, cx:cx+sz]

# Baseline reconstruction on crop
from skimage.restoration import denoise_wavelet
u8_n = crop_noisy.copy()
for c in range(3):
    u8_c = u8_n[..., c]
    med = cv2.medianBlur(u8_c, 3)
    dev = np.abs(u8_c.astype(int) - med.astype(int))
    u8_n[..., c] = np.where(dev > 50, med, u8_c)
crop_base = cv2.fastNlMeansDenoisingColored(u8_n, None, 10, 10, 7, 21)

# Wavelet + NLM
crop_fl = (u8_n.astype(np.float32)/255.0)
crop_wav = denoise_wavelet(crop_fl, method="BayesShrink", mode="soft", wavelet="sym4", channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
crop_wav_u8 = (np.clip(crop_wav, 0, 1) * 255).astype(np.uint8)
crop_classical = cv2.fastNlMeansDenoisingColored(crop_wav_u8, None, 8, 12, 7, 21)

fig, axes = plt.subplots(1, 4, figsize=(10, 2.7), dpi=300)
titles = ["(a) Noisy Input", "(b) Baseline NLM", "(c) Our Hybrid DSP", "(d) Ground Truth"]
imgs = [crop_noisy, crop_base, crop_classical, crop_gt]

for ax, im, t in zip(axes, imgs, titles):
    ax.imshow(im)
    ax.set_title(t, fontsize=9, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
fig3_path = OUT_DIR / "fig3_visual_comparison.png"
plt.savefig(fig3_path, bbox_inches='tight')
plt.close()
print(f"Saved {fig3_path}")
