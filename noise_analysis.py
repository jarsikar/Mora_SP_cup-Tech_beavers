"""
Day 2: Visual noise comparison
Saves side-by-side clean / noisy / amplified-noise images for a few samples.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

GT_DIR = Path("competition_data/public/ground_truth")
NOISY_DIR = Path("competition_data/public/noisy")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

sample_ids = ["001", "050", "150", "300"]

def load_image(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32)

for sid in sample_ids:
    clean = load_image(GT_DIR / f"{sid}.png")
    noisy = load_image(NOISY_DIR / f"{sid}_noise.png")
    noise = noisy - clean

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean.astype(np.uint8))
    axes[0].set_title(f"{sid} - Clean")
    axes[0].axis("off")

    axes[1].imshow(noisy.astype(np.uint8))
    axes[1].set_title(f"{sid} - Noisy")
    axes[1].axis("off")

    # Raw noise is subtle (small pixel differences), so amplify x3 to actually see the pattern
    noise_vis = np.clip(noise * 3 + 128, 0, 255).astype(np.uint8)
    axes[2].imshow(noise_vis)
    axes[2].set_title(f"{sid} - Noise (amplified 3x)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{sid}_comparison.png", dpi=100)
    plt.close()
    print(f"Saved comparison for {sid}")

print("\nDone. Open the 'analysis_outputs' folder and look at each image.")

"""
Day 2 part 2: Quantitative noise statistics
Checks: does noise strength depend on brightness? Is it different per color channel?
"""

# We now look across ALL 460 images, not just 4 samples, for a reliable answer
all_ids = [f"{i:03d}" for i in range(1, 461)]  # "001", "002", ... "460"

# We'll bin pixels by how bright they are in the clean image,
# then measure how noisy each brightness bin is
bins = [0, 51, 102, 153, 204, 255]  # 5 brightness ranges: very dark -> very bright
bin_labels = ["0-50 (darkest)", "51-101", "102-152", "153-203", "204-255 (brightest)"]
bin_noise_sums = [0.0] * 5
bin_pixel_counts = [0] * 5

channel_noise_sums = [0.0, 0.0, 0.0]  # R, G, B
channel_pixel_counts = [0, 0, 0]

# Only check every 20th image for speed (20 images is still a solid sample)
sample_for_stats = all_ids[::20]
print(f"Analyzing {len(sample_for_stats)} images for noise statistics...")

for sid in sample_for_stats:
    clean = load_image(GT_DIR / f"{sid}.png")
    noisy = load_image(NOISY_DIR / f"{sid}_noise.png")
    noise = noisy - clean

    # Brightness = average of R,G,B at each pixel
    brightness = clean.mean(axis=2)

    for i in range(5):
        low, high = bins[i], bins[i + 1]
        mask = (brightness >= low) & (brightness < high)
        if mask.sum() > 0:
            # noise[mask] grabs only the noise values where this brightness range applies
            bin_noise_sums[i] += np.abs(noise[mask]).sum()
            bin_pixel_counts[i] += mask.sum() * 3  # 3 color channels per pixel

    for c in range(3):  # 0=Red, 1=Green, 2=Blue
        channel_noise_sums[c] += np.abs(noise[:, :, c]).sum()
        channel_pixel_counts[c] += noise[:, :, c].size

print("\n--- Noise strength by brightness level ---")
for i in range(5):
    if bin_pixel_counts[i] > 0:
        avg_noise = bin_noise_sums[i] / bin_pixel_counts[i]
        print(f"{bin_labels[i]}: average noise magnitude = {avg_noise:.3f}")
    else:
        print(f"{bin_labels[i]}: no pixels found in this range")

print("\n--- Noise strength by color channel ---")
channel_names = ["Red", "Green", "Blue"]
for c in range(3):
    avg_noise = channel_noise_sums[c] / channel_pixel_counts[c]
    print(f"{channel_names[c]}: average noise magnitude = {avg_noise:.3f}")