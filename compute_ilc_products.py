#!/usr/bin/env python3
"""
Compute ILC products for UNet training.

This script:
1. Loads B-mode foreground patches (all-scale) at 4 frequencies
2. Loads CMB realization patches (frequency-independent)
3. Applies ILC patch-by-patch to compute:
   - ILC CMB reconstructions
   - ILC foreground reconstructions (per frequency)
   - ILC residuals (ILC_cmb - true_cmb)
4. Saves all products for later UNet training

UNet Training Goal:
------------------
Train a UNet to predict ILC residuals using 8-channel input:
  - Channels 1-4: ILC foreground reconstructions at 95, 145, 220, 270 GHz
  - Channels 5-8: Small-scale B-modes (ℓ > 200) at 95, 145, 220, 270 GHz
  - Target: ILC residuals (single channel)

The UNet learns to correct systematic ILC errors based on foreground characteristics.

IMPORTANT: This script preserves patch ordering throughout all operations!
"""

import numpy as np
import os
import sys
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from apply_ilc_core import apply_ilc_to_single_patch
import paths_config

BASE_DIR_B = paths_config.B_SMALL_DIR
OUTPUT_DIR = paths_config.ILC_DIR

# Frequencies
FREQUENCIES = [95, 145, 220, 270]

print("="*80)
print("COMPUTING ILC PRODUCTS FOR UNET TRAINING")
print("="*80)
print(f"Input directory: {BASE_DIR_B}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Frequencies: {FREQUENCIES}")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA (PRESERVING ORDER!)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load B-mode foreground patches (all-scale) for ILC
print("\nLoading B-mode foreground patches (all-scale) for ILC...")
B_patches = {}
for freq in FREQUENCIES:
    file_path = f"{BASE_DIR_B}/sim1-150_freq{freq}_B_patches.npy"
    print(f"  Loading {freq} GHz: {file_path}")
    B_patches[freq] = np.load(file_path)
    print(f"    Shape: {B_patches[freq].shape}")

# Load CMB patches (frequency-independent, from 220 GHz file)
cmb_file = f"{BASE_DIR_B}/sim1-150_freq220_cmb_draw_b_all_scales.npy"
print(f"\nLoading CMB patches: {cmb_file}")
CMB_patches = np.load(cmb_file)
print(f"  Shape: {CMB_patches.shape}")

# Verify all patches have same number
n_patches = B_patches[95].shape[0]
print(f"\nVerifying patch counts...")
for freq in FREQUENCIES:
    assert B_patches[freq].shape[0] == n_patches, f"Mismatch at {freq} GHz"
assert CMB_patches.shape[0] == n_patches, "Mismatch in CMB patches"
print(f"  ✓ All datasets have {n_patches} patches")

# Get patch dimensions
patch_height, patch_width = B_patches[95].shape[1], B_patches[95].shape[2]
print(f"  Patch dimensions: {patch_height} x {patch_width}")

# ============================================================================
# STEP 2: APPLY ILC PATCH-BY-PATCH (PRESERVING ORDER!)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: APPLYING ILC TO ALL PATCHES")
print("="*80)
print(f"Processing {n_patches} patches...")
print("Computing:")
print("  1. ILC CMB reconstructions")
print("  2. ILC foreground reconstructions (4 frequencies)")
print("  3. ILC residuals (ILC_cmb - true_cmb)")

# Initialize storage arrays (maintaining order!)
ILC_cmb_all = np.zeros((n_patches, patch_height, patch_width), dtype=np.float32)
ILC_residuals_all = np.zeros((n_patches, patch_height, patch_width), dtype=np.float32)
ILC_foregrounds_all = {freq: np.zeros((n_patches, patch_height, patch_width), dtype=np.float32) 
                       for freq in FREQUENCIES}
ILC_weights_all = np.zeros((n_patches, len(FREQUENCIES)), dtype=np.float32)

# Process each patch
for patch_idx in tqdm(range(n_patches), desc="Applying ILC"):
    # Stack frequency patches for this spatial location (n_freq, height, width)
    # Order: 95, 145, 220, 270 GHz
    freq_patches = np.stack([
        B_patches[95][patch_idx],
        B_patches[145][patch_idx],
        B_patches[220][patch_idx],
        B_patches[270][patch_idx]
    ], axis=0)
    
    # Get CMB patch for this location
    cmb_patch = CMB_patches[patch_idx]
    
    # Apply ILC to this patch
    ilc_cmb, ilc_foregrounds, weights, ilc_residuals = apply_ilc_to_single_patch(
        freq_patches, cmb_patch, get_residuals=True
    )
    
    # Store results (maintaining order!)
    ILC_cmb_all[patch_idx] = ilc_cmb
    ILC_residuals_all[patch_idx] = ilc_residuals
    ILC_weights_all[patch_idx] = weights
    
    # Store foregrounds for each frequency (order preserved)
    for i, freq in enumerate(FREQUENCIES):
        ILC_foregrounds_all[freq][patch_idx] = ilc_foregrounds[i]

print(f"\n✓ ILC processing complete!")

# ============================================================================
# STEP 3: SAVE ILC PRODUCTS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: SAVING ILC PRODUCTS")
print("="*80)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save ILC CMB reconstructions
ilc_cmb_file = f"{OUTPUT_DIR}/ILC_cmb_b.npy"
np.save(ilc_cmb_file, ILC_cmb_all)
print(f"✓ Saved ILC CMB: {ilc_cmb_file}")
print(f"  Shape: {ILC_cmb_all.shape}")

# Save ILC residuals (UNet target)
ilc_residuals_file = f"{OUTPUT_DIR}/ilc_residuals_b.npy"
np.save(ilc_residuals_file, ILC_residuals_all)
print(f"✓ Saved ILC residuals: {ilc_residuals_file}")
print(f"  Shape: {ILC_residuals_all.shape}")

# Save ILC foreground reconstructions (per frequency)
for freq in FREQUENCIES:
    ilc_fg_file = f"{OUTPUT_DIR}/ILC_foregrounds_b_{freq}.npy"
    np.save(ilc_fg_file, ILC_foregrounds_all[freq])
    print(f"✓ Saved ILC foregrounds ({freq} GHz): {ilc_fg_file}")
    print(f"  Shape: {ILC_foregrounds_all[freq].shape}")

# Save ILC weights for analysis
ilc_weights_file = f"{OUTPUT_DIR}/ILC_weights_b.npy"
np.save(ilc_weights_file, ILC_weights_all)
print(f"✓ Saved ILC weights: {ilc_weights_file}")
print(f"  Shape: {ILC_weights_all.shape}")

# ============================================================================
# STEP 4: COMPUTE AND PRINT STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: ILC PRODUCT STATISTICS")
print("="*80)

print("\nILC CMB Reconstruction:")
print(f"  Mean: {ILC_cmb_all.mean():.6e}")
print(f"  Std:  {ILC_cmb_all.std():.6e}")
print(f"  Min:  {ILC_cmb_all.min():.6e}")
print(f"  Max:  {ILC_cmb_all.max():.6e}")

print("\nILC Residuals (ILC_cmb - true_cmb):")
print(f"  Mean: {ILC_residuals_all.mean():.6e}")
print(f"  Std:  {ILC_residuals_all.std():.6e}")
print(f"  Min:  {ILC_residuals_all.min():.6e}")
print(f"  Max:  {ILC_residuals_all.max():.6e}")
print(f"  Mean |residual|: {np.abs(ILC_residuals_all).mean():.6e}")

print("\nILC Foreground Reconstructions:")
for freq in FREQUENCIES:
    fg = ILC_foregrounds_all[freq]
    print(f"  {freq} GHz:")
    print(f"    Mean: {fg.mean():.6e}, Std: {fg.std():.6e}")
    print(f"    Range: [{fg.min():.6e}, {fg.max():.6e}]")

print("\nILC Weights (should sum to ~1.0 for CMB preservation):")
for i, freq in enumerate(FREQUENCIES):
    weights = ILC_weights_all[:, i]
    print(f"  {freq} GHz:")
    print(f"    Mean: {weights.mean():.4f} ± {weights.std():.4f}")
    print(f"    Range: [{weights.min():.4f}, {weights.max():.4f}]")

# Check weight sum
weight_sums = np.sum(ILC_weights_all, axis=1)
print(f"\nWeight sum statistics (should be ~1.0):")
print(f"  Mean: {weight_sums.mean():.6f} ± {weight_sums.std():.6f}")
print(f"  Range: [{weight_sums.min():.6f}, {weight_sums.max():.6f}]")

# Compute spatial correlation between ILC CMB and true CMB
print("\nSpatial Correlation (ILC CMB vs True CMB):")
correlations = []
for patch_idx in range(min(1000, n_patches)):  # Sample first 1000 patches
    ilc_flat = ILC_cmb_all[patch_idx].flatten()
    true_flat = CMB_patches[patch_idx].flatten()
    corr = np.corrcoef(ilc_flat, true_flat)[0, 1]
    if not np.isnan(corr):
        correlations.append(corr)
correlations = np.array(correlations)
print(f"  Mean correlation: {correlations.mean():.4f} ± {correlations.std():.4f}")
print(f"  Median correlation: {np.median(correlations):.4f}")
print(f"  Range: [{correlations.min():.4f}, {correlations.max():.4f}]")
print(f"  Patches with corr > 0.9: {np.sum(correlations > 0.9)}/{len(correlations)} ({100*np.sum(correlations > 0.9)/len(correlations):.1f}%)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PROCESSING COMPLETE!")
print("="*80)
print(f"\nAll ILC products saved to: {OUTPUT_DIR}")
print(f"\nFiles created:")
print(f"  - ILC_cmb_b.npy              (ILC CMB reconstructions)")
print(f"  - ilc_residuals_b.npy        (UNet target: ILC errors)")
print(f"  - ILC_foregrounds_b_95.npy   (ILC foregrounds at 95 GHz)")
print(f"  - ILC_foregrounds_b_145.npy  (ILC foregrounds at 145 GHz)")
print(f"  - ILC_foregrounds_b_220.npy  (ILC foregrounds at 220 GHz)")
print(f"  - ILC_foregrounds_b_270.npy  (ILC foregrounds at 270 GHz)")
print(f"  - ILC_weights_b.npy          (ILC weights for analysis)")

print("\n" + "="*80)
print("NEXT STEPS: ASSEMBLE UNET INPUT/TARGET")
print("="*80)
print("""
To create UNet training data, combine ILC products with small-scale B-modes:

# Load small-scale B-mode patches (original foregrounds, no CMB)
B_small_95 = np.load(f"{BASE_DIR_B}/sim1-150_freq95_B_patches_small.npy")
B_small_145 = np.load(f"{BASE_DIR_B}/sim1-150_freq145_B_patches_small.npy")
B_small_220 = np.load(f"{BASE_DIR_B}/sim1-150_freq220_B_patches_small.npy")
B_small_270 = np.load(f"{BASE_DIR_B}/sim1-150_freq270_B_patches_small.npy")

# Load ILC products
ILC_foregrounds_95 = np.load(f"{OUTPUT_DIR}/ILC_foregrounds_b_95.npy")
ILC_foregrounds_145 = np.load(f"{OUTPUT_DIR}/ILC_foregrounds_b_145.npy")
ILC_foregrounds_220 = np.load(f"{OUTPUT_DIR}/ILC_foregrounds_b_220.npy")
ILC_foregrounds_270 = np.load(f"{OUTPUT_DIR}/ILC_foregrounds_b_270.npy")
ILC_residuals = np.load(f"{OUTPUT_DIR}/ilc_residuals_b.npy")

# UNet Input: 8 channels (4 ILC foregrounds + 4 small-scale B)
UNet_input = np.stack([
    ILC_foregrounds_95, ILC_foregrounds_145, ILC_foregrounds_220, ILC_foregrounds_270,
    B_small_95, B_small_145, B_small_220, B_small_270
], axis=1)  # Shape: (n_patches, 8, height, width)

# UNet Target: 1 channel (ILC residuals)
UNet_target = ILC_residuals[:, np.newaxis, :, :]  # Shape: (n_patches, 1, height, width)

# Save for training
np.save(f"{OUTPUT_DIR}/UNet_input_8channel.npy", UNet_input)
np.save(f"{OUTPUT_DIR}/UNet_target_residuals.npy", UNet_target)

print(f"UNet input shape: {UNet_input.shape}")
print(f"UNet target shape: {UNet_target.shape}")
""")
print("="*80)

