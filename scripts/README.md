# Mora SP Cup 2026 - Official Submission Documentation
**Team: Tech_beavers**

This repository contains the official preliminary round submission code, model checkpoints, and reproduction instructions for the **Mora SP Cup 2026 Low-Light Image Denoising & Enhancement Challenge**.

---

## Architecture: Dual-Domain Grand Ensemble

Our system unites multi-scale classical signal processing with a deep residual neural network to maximize both structural fidelity (&Delta;SSIM) and peak signal-to-noise ratio (&Delta;PSNR):
1. **Defect Pixel Correction:** Spatial median outlier filter targeting sensor impulse/hot pixels.
2. **Classical Transform Branch:** Discrete Wavelet Decomposition using Symlets (`sym4`) in $YC_bC_r$ perceptual color space with BayesShrink soft-thresholding, followed by collaborative Non-Local Means (NLM).
3. **Deep Neural Branch:** Lightweight Residual UNet (~1.29M parameters) trained under a residual formulation $\hat{I} = I - \mathcal{R}_\theta(I)$ with mixed-precision on an NVIDIA RTX 5060 Laptop GPU.
4. **Dual-Domain Blending:** Convex ensemble ($\hat{I}_{\text{final}} = 0.50 \hat{I}_{\text{DSP}} + 0.50 \hat{I}_{\text{UNet}}$) canceling uncorrelated residual noise.
5. **Autonomous CPU Fallback:** If GPU acceleration is disabled or PyTorch is absent, the script automatically triggers the high-performance Classical Wavelet+NLM pipeline (scoring 0.3803) without raising exceptions.

---

## Instructions to Run

### 1. Requirements
- Python 3.10+ or 3.11+
- Install dependencies:
  ```bash
  pip install -r scripts/requirements.txt
  ```

### 2. Execution Command
Run the official denoising script by pointing to the input noisy folder and target output folder:
```bash
python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised
```
*(Aliases `--input_dir` and `--output_dir` are also supported).*

### 3. CPU Benchmark & Verification
To verify execution strictly on CPU:
```bash
python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised --force_cpu
```
- **Observed CPU Runtime:** 20.52 seconds for all 20 images (**1.03 s/image**).
- **Offline Compliance:** Zero remote network or cloud calls.

---

## Submission Package & Verification Record

- **Team Name:** `Tech_beavers`
- **Preliminary ZIP Package:** `Tech_beavers.zip` (Contains exactly `461.png` &ndash; `480.png`)
- **Preliminary Report:** `Tech_beavers_Report.pdf` (Strictly 5 pages, 1-inch margins, 12pt body)
- **Model Checkpoint File:** `scripts/best_model.pth`
- **Model SHA-256 Checksum:** `5a63b7fb5876bdba7797b76cbbb7140eefb404297e9f7f9a213b8763b3d3a242`
- **Validation Composite Score:** **0.4046** (+84% over 0.2201 baseline)
- **Average Validation $\Delta$PSNR:** **+7.10 dB** (up to +9.73 dB)
- **Average Validation $\Delta$SSIM:** **+0.3018**
