#!/usr/bin/env python3
"""
Evaluate effectiveness of UNet-enhanced ILC vs vanilla ILC for CMB reconstruction.

This script:
1. Loads multi-frequency B-mode patches (95, 145, 220, 270 GHz)
2. Loads primary CMB and scales it by a factor (default 0.001)
3. Adds scaled CMB to each frequency channel
4. Constructs synthetic UNet channel: scaled_cmb + b_small + unet_predicted_b_large
5. Compares vanilla ILC (4 frequencies) vs enhanced ILC (4 frequencies + UNet channel)

Usage:
    python eval_ilc_enhancement.py --model-path <path> --output-dir <dir> [--cmb-scale 0.01]
    
    Example:
    python eval_ilc_enhancement.py \
    --model-path /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/best_model_153600_19200_nonorm.pt \
    --output-dir /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/ilc_enhancement_results \
    --cmb-scale 0.01 \
    --n-test 500

    python eval_ilc_enhancement.py \
    --model-path /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_cmb_scaled/normalized_b_only/best_model_153600_19200_norm.pt \
    --output-dir /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_cmb_scaled/normalized_b_only/ilc_enhancement_results \
    --cmb-scale 0.01 \
    --b-only \
    --normalize
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import pickle

# Local code-release imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import plot_params
plot_params.setup_latex_path()
plot_params.patch_texmanager()
plot_params.setup_latex_preamble()
plt.rcParams.update(plot_params.params)

import architecture

try:
    from cmb_spectrum_2d import calculate_2d_spectrum
except Exception as e:
    print(f"Warning: Error importing calculate_2d_spectrum: {e}")
    calculate_2d_spectrum = None

from apply_ilc_core import apply_ilc_to_single_patch, apply_ilc_to_single_patch_with_unet
import paths_config

# =============================================================================
# CONFIGURATION
# =============================================================================

B_DIR = paths_config.B_SMALL_DIR
T_E_DIR = paths_config.T_E_DIR
STATS_FILE = paths_config.STATS_FILE_SINGLEFREQ

# Multi-frequency channels for ILC
FREQUENCIES = [95, 145, 220, 270]  # GHz
REFERENCE_FREQ = 220  # GHz - frequency for UNet model

# Power spectrum parameters
import healpy as hp
NSIDE = 1024
PIX_SIZE = int(hp.nside2resol(NSIDE, arcmin=True))  # arcminutes
DELTA_ELL = 50
ELL_MAX = 2000  # Maximum ell for cross-power spectrum analysis
PATCH_SIZE = 128

# UNet configuration
CONFIG = {
    "batch_size": 32,
    "in_channels": 3,
    "out_channels": 1,
    "feature_dims": [32, 64],
    "negative_slope": 0.01,
}


def load_multifreq_data(frequencies, test_indices):
    """
    Load multi-frequency B-mode patches for specified frequencies.
    
    Args:
        frequencies: List of frequencies in GHz (e.g., [95, 145, 220, 270])
        test_indices: Indices for test set
        
    Returns:
        freq_patches_dict: Dictionary mapping frequency to patches array
    """
    print("\n" + "="*80)
    print("LOADING MULTI-FREQUENCY B-MODE PATCHES")
    print("="*80)
    
    freq_patches_dict = {}
    
    for freq in frequencies:
        # Load B-mode patches (foreground only, no CMB yet)
        b_file = f"{B_DIR}/sim1-150_freq{freq}_B_patches.npy"
        print(f"\nLoading {freq} GHz B-mode patches: {b_file}")
        
        if not os.path.exists(b_file):
            raise FileNotFoundError(f"B-mode file not found: {b_file}")
        
        b_patches = np.load(b_file)
        print(f"  Shape: {b_patches.shape}")
        
        # Extract test set
        test_patches = b_patches[test_indices]
        freq_patches_dict[freq] = test_patches
        
        print(f"  Test set shape: {test_patches.shape}")
        print(f"  Range: [{test_patches.min():.3e}, {test_patches.max():.3e}]")
    
    return freq_patches_dict


def load_primary_cmb(test_indices, reference_freq=220):
    """
    Load primary CMB B-mode patches (same for all frequencies).
    
    Args:
        test_indices: Indices for test set
        reference_freq: Reference frequency (default 220 GHz)
        
    Returns:
        cmb_patches: Primary CMB B-mode patches for test set
    """
    print("\n" + "="*80)
    print("LOADING PRIMARY CMB B-MODE PATCHES")
    print("="*80)
    
    cmb_file = f"{B_DIR}/sim1-150_freq{reference_freq}_cmb_draw_b_all_scales.npy"
    print(f"\nLoading primary CMB: {cmb_file}")
    
    if not os.path.exists(cmb_file):
        raise FileNotFoundError(f"CMB file not found: {cmb_file}")
    
    cmb_patches = np.load(cmb_file)
    print(f"  Shape: {cmb_patches.shape}")
    
    # Extract test set
    test_cmb = cmb_patches[test_indices]
    print(f"  Test set shape: {test_cmb.shape}")
    print(f"  Range: [{test_cmb.min():.3e}, {test_cmb.max():.3e}]")
    print(f"  Mean: {test_cmb.mean():.3e}, Std: {test_cmb.std():.3e}")
    
    return test_cmb


def load_unet_components(test_indices, reference_freq=220):
    """
    Load UNet components: predictions, small-scale foregrounds.
    
    Args:
        test_indices: Indices for test set
        reference_freq: Reference frequency (default 220 GHz)
        
    Returns:
        unet_predictions: UNet predictions of large-scale B-mode foregrounds
        b_small: Small-scale B-mode foregrounds (UNet input)
    """
    print("\n" + "="*80)
    print("LOADING UNET COMPONENTS")
    print("="*80)
    
    # Load small-scale B-mode foregrounds
    b_small_file = f"{B_DIR}/sim1-150_freq{reference_freq}_B_patches_small.npy"
    print(f"\nLoading small-scale B-mode foregrounds: {b_small_file}")
    
    if not os.path.exists(b_small_file):
        raise FileNotFoundError(f"Small-scale file not found: {b_small_file}")
    
    b_small = np.load(b_small_file)
    test_b_small = b_small[test_indices]
    print(f"  Shape: {test_b_small.shape}")
    print(f"  Range: [{test_b_small.min():.3e}, {test_b_small.max():.3e}]")
    
    return test_b_small


def load_model_and_generate_predictions(model_path, test_indices, normalize=False, 
                                        use_te=True, reference_freq=220):
    """
    Load trained UNet model and generate predictions.
    
    Args:
        model_path: Path to trained model checkpoint
        test_indices: Indices for test set
        normalize: Whether model uses normalization
        use_te: Whether to use T and E channels
        reference_freq: Reference frequency (default 220 GHz)
        
    Returns:
        predictions: UNet predictions of large-scale B-mode foregrounds
    """
    print("\n" + "="*80)
    print("LOADING UNET MODEL AND GENERATING PREDICTIONS")
    print("="*80)
    
    # Load test data for UNet
    print(f"\nLoading test data for UNet...")
    
    # Load B small-scale (input channel 0)
    b_small_file = f"{B_DIR}/sim1-150_freq{reference_freq}_B_patches_small.npy"
    b_small = np.load(b_small_file)[test_indices]
    
    # Load B large-scale (target)
    b_large_file = f"{B_DIR}/sim1-150_freq{reference_freq}_B_patches_large.npy"
    b_large = np.load(b_large_file)[test_indices]
    
    # Stack input channels
    input_channels = [b_small]
    
    if use_te:
        # Load T and E
        t_file = f"{T_E_DIR}/sim1-150_freq{reference_freq}_T_patches.npy"
        t_all = np.load(t_file)[test_indices]
        
        e_file = f"{T_E_DIR}/sim1-150_freq{reference_freq}_E_patches.npy"
        e_all = np.load(e_file)[test_indices]
        
        input_channels.extend([t_all, e_all])
    
    test_X = np.stack(input_channels, axis=1)  # Shape: (n_patches, n_channels, H, W)
    test_y = b_large[:, np.newaxis, :, :]  # Shape: (n_patches, 1, H, W)
    
    print(f"  UNet input shape: {test_X.shape}")
    print(f"  UNet target shape: {test_y.shape}")
    
    # Normalize if needed
    norm_stats = None
    if normalize:
        print("\nNormalizing test data...")
        stats = np.load(STATS_FILE)
        
        input_mean = stats['input_mean']
        input_std = stats['input_std']
        target_mean = stats['target_mean']
        target_std = stats['target_std']
        
        if not use_te:
            input_mean = input_mean[:1]
            input_std = input_std[:1]
        
        # Normalize
        n_channels = test_X.shape[1]
        for i in range(n_channels):
            test_X[:, i] = (test_X[:, i] - input_mean[i]) / (input_std[i] + 1e-8)
        
        test_y = (test_y - target_mean) / (target_std + 1e-8)
        
        norm_stats = {
            'input_mean': input_mean,
            'input_std': input_std,
            'target_mean': target_mean,
            'target_std': target_std
        }
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get model config
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        in_channels = saved_config.get('in_channels', 3 if use_te else 1)
        model_config = {
            'in_channels': in_channels,
            'out_channels': saved_config.get('out_channels', 1),
            'feature_dims': saved_config.get('feature_dims', CONFIG["feature_dims"]),
            'negative_slope': saved_config.get('negative_slope', CONFIG["negative_slope"]),
        }
    else:
        in_channels = 3 if use_te else 1
        model_config = CONFIG.copy()
        model_config['in_channels'] = in_channels
    
    # Create model
    mean_init_path = None
    if not normalize:
        model_dir = os.path.dirname(model_path)
        mean_init_path = os.path.join(model_dir, "mean_init_stats.pkl")
        if not os.path.exists(mean_init_path):
            mean_init_path = None
    
    model = architecture.UNET(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        feature_dims=model_config['feature_dims'],
        negative_slope=model_config['negative_slope'],
        mean_init=mean_init_path
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded on device: {device}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    from torch.utils.data import DataLoader, TensorDataset
    
    test_dataset = TensorDataset(
        torch.FloatTensor(test_X),
        torch.FloatTensor(test_y)
    )
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Predicting"):
            x = x.to(device)
            pred = model(x)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Denormalize if needed
    if norm_stats is not None:
        print("Denormalizing predictions...")
        predictions = predictions * norm_stats['target_std'] + norm_stats['target_mean']
    
    # Squeeze channel dimension
    if predictions.ndim == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0, :, :]
    
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Range: [{predictions.min():.3e}, {predictions.max():.3e}]")
    
    return predictions


def apply_vanilla_ilc(freq_patches_dict, scaled_cmb_patches, frequencies):
    """
    Apply vanilla ILC to multi-frequency observations.
    
    Args:
        freq_patches_dict: Dictionary mapping frequency to foreground patches
        scaled_cmb_patches: Scaled primary CMB patches
        frequencies: List of frequencies
        
    Returns:
        vanilla_ilc_cmb: ILC-reconstructed CMB (n_patches, H, W)
        vanilla_ilc_weights: ILC weights (n_patches, n_freq)
        vanilla_ilc_residuals: ILC residuals (n_patches, H, W)
    """
    print("\n" + "="*80)
    print("APPLYING VANILLA ILC")
    print("="*80)
    
    n_patches = scaled_cmb_patches.shape[0]
    n_freq = len(frequencies)
    
    # Stack frequency patches: shape (n_patches, n_freq, H, W)
    freq_patch_stack = np.stack([freq_patches_dict[freq] for freq in frequencies], axis=1)
    
    print(f"\nProcessing {n_patches} patches with {n_freq} frequencies...")
    print(f"  Frequency patch stack shape: {freq_patch_stack.shape}")
    print(f"  Scaled CMB shape: {scaled_cmb_patches.shape}")
    
    # Initialize output arrays
    vanilla_ilc_cmb = np.zeros_like(scaled_cmb_patches)
    vanilla_ilc_weights = np.zeros((n_patches, n_freq))
    vanilla_ilc_residuals = np.zeros_like(scaled_cmb_patches)
    vanilla_ilc_variances = np.zeros(n_patches)
    
    # Process each patch
    for i in tqdm(range(n_patches), desc="Vanilla ILC"):
        freq_patch = freq_patch_stack[i]  # Shape: (n_freq, H, W)
        cmb_patch = scaled_cmb_patches[i]  # Shape: (H, W)
        
        # Apply ILC
        ilc_cmb, ilc_fg, weights, residuals = apply_ilc_to_single_patch(
            freq_patch, cmb_patch, get_residuals=True
        )
        
        # Compute actual variance of ILC CMB reconstruction
        variance = np.var(ilc_cmb)
        
        vanilla_ilc_cmb[i] = ilc_cmb
        vanilla_ilc_weights[i] = weights
        vanilla_ilc_residuals[i] = residuals
        vanilla_ilc_variances[i] = variance
    
    print(f"\n  ✓ Vanilla ILC completed")
    print(f"    ILC CMB shape: {vanilla_ilc_cmb.shape}")
    print(f"    ILC weights shape: {vanilla_ilc_weights.shape}")
    print(f"    ILC residuals shape: {vanilla_ilc_residuals.shape}")
    print(f"    ILC variances shape: {vanilla_ilc_variances.shape}")
    print(f"    Mean variance: {np.mean(vanilla_ilc_variances):.6e}")
    
    return vanilla_ilc_cmb, vanilla_ilc_weights, vanilla_ilc_residuals, vanilla_ilc_variances


def apply_enhanced_ilc(freq_patches_dict, scaled_cmb_patches, unet_predictions, 
                      b_small, frequencies):
    """
    Apply UNet-enhanced ILC to multi-frequency observations + UNet synthetic channel.
    
    Args:
        freq_patches_dict: Dictionary mapping frequency to foreground patches
        scaled_cmb_patches: Scaled primary CMB patches
        unet_predictions: UNet predictions of large-scale foregrounds
        b_small: Small-scale B-mode foregrounds
        frequencies: List of frequencies
        
    Returns:
        enhanced_ilc_cmb: ILC-reconstructed CMB (n_patches, H, W)
        enhanced_ilc_weights: ILC weights (n_patches, n_freq + 1)
        enhanced_ilc_residuals: ILC residuals (n_patches, H, W)
        unet_synthetic_channel: Synthetic UNet channel (n_patches, H, W)
    """
    print("\n" + "="*80)
    print("APPLYING UNET-ENHANCED ILC")
    print("="*80)
    
    n_patches = scaled_cmb_patches.shape[0]
    n_freq = len(frequencies)
    
    # Stack frequency patches: shape (n_patches, n_freq, H, W)
    freq_patch_stack = np.stack([freq_patches_dict[freq] for freq in frequencies], axis=1)
    
    # Construct UNet synthetic channel: scaled_cmb + b_small + unet_predicted_b_large
    print(f"\nConstructing UNet synthetic channel...")
    print(f"  Formula: synthetic_channel = scaled_cmb + b_small + unet_predicted_b_large")
    # unet_synthetic_channel = scaled_cmb_patches + b_small + unet_predictions
    unet_synthetic_channel = scaled_cmb_patches + unet_predictions
    
    print(f"  Synthetic channel shape: {unet_synthetic_channel.shape}")
    print(f"  Range: [{unet_synthetic_channel.min():.3e}, {unet_synthetic_channel.max():.3e}]")
    
    print(f"\nProcessing {n_patches} patches with {n_freq} frequencies + 1 UNet channel...")
    
    # Initialize output arrays
    enhanced_ilc_cmb = np.zeros_like(scaled_cmb_patches)
    enhanced_ilc_weights = np.zeros((n_patches, n_freq + 1))
    enhanced_ilc_residuals = np.zeros_like(scaled_cmb_patches)
    enhanced_ilc_variances = np.zeros(n_patches)
    
    # Process each patch
    for i in tqdm(range(n_patches), desc="Enhanced ILC"):
        freq_patch = freq_patch_stack[i]  # Shape: (n_freq, H, W)
        cmb_patch = scaled_cmb_patches[i]  # Shape: (H, W)
        unet_channel = unet_synthetic_channel[i]  # Shape: (H, W)
        
        # Apply ILC with UNet
        ilc_cmb, ilc_fg, ilc_unet, weights = apply_ilc_to_single_patch_with_unet(
            freq_patch, cmb_patch, unet_channel
        )
        
        # Compute residuals
        residuals = ilc_cmb - cmb_patch
        
        # Compute actual variance of ILC CMB reconstruction
        variance = np.var(ilc_cmb)
        
        enhanced_ilc_cmb[i] = ilc_cmb
        enhanced_ilc_weights[i] = weights
        enhanced_ilc_residuals[i] = residuals
        enhanced_ilc_variances[i] = variance
    
    print(f"\n  ✓ Enhanced ILC completed")
    print(f"    ILC CMB shape: {enhanced_ilc_cmb.shape}")
    print(f"    ILC weights shape: {enhanced_ilc_weights.shape}")
    print(f"    ILC residuals shape: {enhanced_ilc_residuals.shape}")
    print(f"    ILC variances shape: {enhanced_ilc_variances.shape}")
    print(f"    Mean variance: {np.mean(enhanced_ilc_variances):.6e}")
    
    return enhanced_ilc_cmb, enhanced_ilc_weights, enhanced_ilc_residuals, unet_synthetic_channel, enhanced_ilc_variances


def compute_mse(reconstruction, target):
    """Compute MSE between reconstruction and target."""
    return np.mean((reconstruction - target)**2)


def compute_spatial_correlation(map1, map2):
    """Compute spatial Pearson correlation."""
    flat1 = map1.flatten()
    flat2 = map2.flatten()
    corr = np.corrcoef(flat1, flat2)[0, 1]
    return 0.0 if np.isnan(corr) else corr


def compute_normalized_cross_spectrum(map1, map2, delta_ell=DELTA_ELL, ell_max=ELL_MAX, 
                                     pix_size=PIX_SIZE, N=PATCH_SIZE):
    """
    Compute normalized cross-power spectrum between two 2D maps.
    
    Args:
        map1, map2: 2D numpy arrays (H, W) - input maps
        delta_ell: bin width for power spectra
        ell_max: maximum ell for analysis
        pix_size: pixel size in arcminutes
        N: patch size in pixels
        
    Returns:
        ell_array: array of ell values
        normalized_cross_ps: normalized cross-power spectrum
    """
    if calculate_2d_spectrum is None:
        raise ValueError("calculate_2d_spectrum not available - cannot compute cross-spectrum")
    
    # Compute cross-power spectrum
    ell_array, cl_cross = calculate_2d_spectrum(
        map1, map2,
        delta_ell=delta_ell, ell_max=ell_max,
        pix_size=pix_size, N=N, lib_mode='NumPy'
    )
    
    # Compute auto-power spectra for normalization
    _, cl_map1_map1 = calculate_2d_spectrum(
        map1, map1,
        delta_ell=delta_ell, ell_max=ell_max,
        pix_size=pix_size, N=N, lib_mode='NumPy'
    )
    _, cl_map2_map2 = calculate_2d_spectrum(
        map2, map2,
        delta_ell=delta_ell, ell_max=ell_max,
        pix_size=pix_size, N=N, lib_mode='NumPy'
    )
    
    # Normalize cross-power spectrum
    normalized_cross_ps = cl_cross / np.sqrt(cl_map1_map1 * cl_map2_map2)
    
    return ell_array, normalized_cross_ps


def analyze_results(vanilla_ilc_cmb, vanilla_ilc_residuals, vanilla_ilc_weights, vanilla_ilc_variances,
                   enhanced_ilc_cmb, enhanced_ilc_residuals, enhanced_ilc_weights, enhanced_ilc_variances,
                   primary_cmb, frequencies):
    """
    Analyze and compare vanilla vs enhanced ILC results.
    
    Returns:
        results_dict: Dictionary containing analysis results
    """
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    n_patches = primary_cmb.shape[0]
    
    # Compute metrics for each patch
    vanilla_mse = []
    enhanced_mse = []
    vanilla_corr = []
    enhanced_corr = []
    improvement_ratio = []
    
    for i in tqdm(range(n_patches), desc="Computing metrics"):
        # MSE
        v_mse = compute_mse(vanilla_ilc_cmb[i], primary_cmb[i])
        e_mse = compute_mse(enhanced_ilc_cmb[i], primary_cmb[i])
        vanilla_mse.append(v_mse)
        enhanced_mse.append(e_mse)
        
        # Correlation
        v_corr = compute_spatial_correlation(vanilla_ilc_cmb[i], primary_cmb[i])
        e_corr = compute_spatial_correlation(enhanced_ilc_cmb[i], primary_cmb[i])
        vanilla_corr.append(v_corr)
        enhanced_corr.append(e_corr)
        
        # Improvement ratio
        if v_mse > 0:
            improvement_ratio.append(v_mse / e_mse)
        else:
            improvement_ratio.append(1.0)
    
    vanilla_mse = np.array(vanilla_mse)
    enhanced_mse = np.array(enhanced_mse)
    vanilla_corr = np.array(vanilla_corr)
    enhanced_corr = np.array(enhanced_corr)
    improvement_ratio = np.array(improvement_ratio)
    
    # Generate statistics text
    stats_lines = []
    stats_lines.append(f"\n{'='*80}")
    stats_lines.append("SUMMARY STATISTICS")
    stats_lines.append(f"{'='*80}")
    
    stats_lines.append(f"\nMSE (Mean Squared Error):")
    stats_lines.append(f"  Vanilla ILC:  {np.mean(vanilla_mse):.6e} ± {np.std(vanilla_mse):.6e}")
    stats_lines.append(f"  Enhanced ILC: {np.mean(enhanced_mse):.6e} ± {np.std(enhanced_mse):.6e}")
    stats_lines.append(f"  Improvement:  {(1 - np.mean(enhanced_mse)/np.mean(vanilla_mse))*100:.2f}%")
    
    stats_lines.append(f"\nSpatial Correlation:")
    stats_lines.append(f"  Vanilla ILC:  {np.mean(vanilla_corr):.6f} ± {np.std(vanilla_corr):.6f}")
    stats_lines.append(f"  Enhanced ILC: {np.mean(enhanced_corr):.6f} ± {np.std(enhanced_corr):.6f}")
    stats_lines.append(f"  Improvement:  {np.mean(enhanced_corr) - np.mean(vanilla_corr):+.6f}")
    
    stats_lines.append(f"\nImprovement Ratio (vanilla_MSE / enhanced_MSE):")
    stats_lines.append(f"  Mean: {np.mean(improvement_ratio):.4f}x")
    stats_lines.append(f"  Median: {np.median(improvement_ratio):.4f}x")
    stats_lines.append(f"  Patches with improvement (ratio > 1): {np.sum(improvement_ratio > 1)}/{n_patches} ({100*np.sum(improvement_ratio > 1)/n_patches:.1f}%)")
    
    stats_lines.append(f"\nILC CMB Reconstruction Variance:")
    stats_lines.append(f"  Vanilla ILC:  {np.mean(vanilla_ilc_variances):.6e} ± {np.std(vanilla_ilc_variances):.6e}")
    stats_lines.append(f"  Enhanced ILC: {np.mean(enhanced_ilc_variances):.6e} ± {np.std(enhanced_ilc_variances):.6e}")
    variance_reduction = (1 - np.mean(enhanced_ilc_variances)/np.mean(vanilla_ilc_variances)) * 100
    stats_lines.append(f"  Variance reduction: {variance_reduction:.2f}%")
    stats_lines.append(f"  Enhanced ILC ≤ Vanilla ILC: {np.sum(enhanced_ilc_variances <= vanilla_ilc_variances)}/{n_patches} ({100*np.sum(enhanced_ilc_variances <= vanilla_ilc_variances)/n_patches:.1f}%)")
    
    stats_lines.append(f"\nILC Weights Statistics:")
    
    # Verify that weights sum to 1 for each patch
    vanilla_weight_sums = np.sum(vanilla_ilc_weights, axis=1)
    enhanced_weight_sums = np.sum(enhanced_ilc_weights, axis=1)
    
    stats_lines.append(f"  Weight sum verification (should be ~1.0 for each patch):")
    stats_lines.append(f"    Vanilla ILC: mean={np.mean(vanilla_weight_sums):.6f}, std={np.std(vanilla_weight_sums):.6f}, range=[{np.min(vanilla_weight_sums):.6f}, {np.max(vanilla_weight_sums):.6f}]")
    stats_lines.append(f"    Enhanced ILC: mean={np.mean(enhanced_weight_sums):.6f}, std={np.std(enhanced_weight_sums):.6f}, range=[{np.min(enhanced_weight_sums):.6f}, {np.max(enhanced_weight_sums):.6f}]")
    
    stats_lines.append(f"\n  Vanilla ILC ({len(frequencies)} frequencies):")
    for i, freq in enumerate(frequencies):
        stats_lines.append(f"    {freq} GHz: {np.mean(vanilla_ilc_weights[:, i]):.4f} ± {np.std(vanilla_ilc_weights[:, i]):.4f}")
    # Show sum of mean weights (should be ~1.0)
    vanilla_mean_sum = np.sum([np.mean(vanilla_ilc_weights[:, i]) for i in range(len(frequencies))])
    stats_lines.append(f"    Sum of mean weights: {vanilla_mean_sum:.6f} (should be ~1.0)")
    
    stats_lines.append(f"\n  Enhanced ILC ({len(frequencies)} frequencies + UNet):")
    for i, freq in enumerate(frequencies):
        stats_lines.append(f"    {freq} GHz: {np.mean(enhanced_ilc_weights[:, i]):.4f} ± {np.std(enhanced_ilc_weights[:, i]):.4f}")
    stats_lines.append(f"    UNet channel: {np.mean(enhanced_ilc_weights[:, -1]):.4f} ± {np.std(enhanced_ilc_weights[:, -1]):.4f}")
    # Show sum of mean weights (should be ~1.0)
    enhanced_mean_sum = np.sum([np.mean(enhanced_ilc_weights[:, i]) for i in range(len(frequencies) + 1)])
    stats_lines.append(f"    Sum of mean weights: {enhanced_mean_sum:.6f} (should be ~1.0)")
    
    # Print statistics
    stats_text = "\n".join(stats_lines)
    print(stats_text)
    
    # Return results dictionary
    results = {
        'vanilla_ilc_cmb': vanilla_ilc_cmb,
        'enhanced_ilc_cmb': enhanced_ilc_cmb,
        'vanilla_ilc_residuals': vanilla_ilc_residuals,
        'enhanced_ilc_residuals': enhanced_ilc_residuals,
        'vanilla_ilc_weights': vanilla_ilc_weights,
        'enhanced_ilc_weights': enhanced_ilc_weights,
        'vanilla_ilc_variances': vanilla_ilc_variances,
        'enhanced_ilc_variances': enhanced_ilc_variances,
        'vanilla_mse': vanilla_mse,
        'enhanced_mse': enhanced_mse,
        'vanilla_corr': vanilla_corr,
        'enhanced_corr': enhanced_corr,
        'improvement_ratio': improvement_ratio,
        'primary_cmb': primary_cmb,
        'stats_text': stats_text,  # Include formatted statistics text
    }
    
    return results


def compute_cross_spectra_analysis(vanilla_ilc_cmb, enhanced_ilc_cmb, primary_cmb, scaled_cmb,
                                   freq_patches_dict, reference_freq=220):
    """
    Compute normalized cross-spectra between reconstructed foregrounds and true foregrounds.
    
    This function:
    1. Computes residuals: vanilla_ilc_resid and enhanced_ilc_resid (with respect to scaled_cmb)
    2. Computes observed_cmb = scaled_cmb + b_fg_220
    3. Computes reconstructed foregrounds: vanilla_ilc_fg and enhanced_ilc_fg
    4. Computes normalized cross-spectra between reconstructed fg and true fg (all_scales_b_fg)
    
    Args:
        vanilla_ilc_cmb: Vanilla ILC reconstructed CMB (n_patches, H, W)
        enhanced_ilc_cmb: Enhanced ILC reconstructed CMB (n_patches, H, W)
        primary_cmb: Primary (true, unscaled) CMB B-mode patches (n_patches, H, W)
        scaled_cmb: Scaled CMB patches used in ILC (n_patches, H, W)
        freq_patches_dict: Dictionary mapping frequency to foreground patches
        reference_freq: Reference frequency for foreground patches (default: 220 GHz)
        
    Returns:
        results_dict: Dictionary containing:
            - vanilla_ilc_resid: Residuals (n_patches, H, W) - computed with respect to scaled_cmb
            - enhanced_ilc_resid: Residuals (n_patches, H, W) - computed with respect to scaled_cmb
            - observed_cmb: Observed CMB (n_patches, H, W)
            - vanilla_ilc_fg: Reconstructed foregrounds (n_patches, H, W)
            - enhanced_ilc_fg: Reconstructed foregrounds (n_patches, H, W)
            - all_scales_b_fg: True all-scales B-mode foregrounds (n_patches, H, W)
            - vanilla_cross_spectra_array: Cross-spectra array (n_patches, n_ell_bins)
            - enhanced_cross_spectra_array: Cross-spectra array (n_patches, n_ell_bins)
            - ell_array: Array of ell values
            - mean_vanilla_cross_ps: Mean cross-spectrum across patches
            - std_vanilla_cross_ps: Std cross-spectrum across patches
            - mean_enhanced_cross_ps: Mean cross-spectrum across patches
            - std_enhanced_cross_ps: Std cross-spectrum across patches
    """
    print("\n" + "="*80)
    print("COMPUTING CROSS-SPECTRA ANALYSIS")
    print("="*80)
    
    n_patches = primary_cmb.shape[0]
    
    # Step 1: Compute residuals (with respect to scaled_cmb, which is what ILC is trying to reconstruct)
    print("\nStep 1: Computing residuals (with respect to scaled CMB)...")
    vanilla_ilc_resid = vanilla_ilc_cmb - scaled_cmb
    enhanced_ilc_resid = enhanced_ilc_cmb - scaled_cmb
    
    print(f"  Vanilla ILC residuals shape: {vanilla_ilc_resid.shape}")
    print(f"    Mean: {np.mean(vanilla_ilc_resid):.6e}, Std: {np.std(vanilla_ilc_resid):.6e}")
    print(f"  Enhanced ILC residuals shape: {enhanced_ilc_resid.shape}")
    print(f"    Mean: {np.mean(enhanced_ilc_resid):.6e}, Std: {np.std(enhanced_ilc_resid):.6e}")
    
    # Step 2: Compute observed_cmb = scaled_cmb + b_fg_220
    print(f"\nStep 2: Computing observed CMB = scaled_cmb + b_fg_{reference_freq}...")
    if reference_freq not in freq_patches_dict:
        raise ValueError(f"Reference frequency {reference_freq} GHz not found in freq_patches_dict")
    
    b_fg_220 = freq_patches_dict[reference_freq]  # All-scales B-mode foreground at 220 GHz
    observed_cmb = scaled_cmb + b_fg_220
    
    # print(f"  b_fg_{reference_freq} shape: {b_fg_220.shape}")
    # print(f"  observed_cmb shape: {observed_cmb.shape}")
    print(f"    Mean: {np.mean(observed_cmb):.6e}, Std: {np.std(observed_cmb):.6e}")
    print(f"    Mean scaled cmb: {np.mean(scaled_cmb):.6e}, Std: {np.std(scaled_cmb):.6e}")
    print(f"    Mean fg 220: {np.mean(b_fg_220):.6e}, Std: {np.std(b_fg_220):.6e}")
    
    # Step 3: Compute reconstructed foregrounds
    print("\nStep 3: Computing reconstructed foregrounds...")
    enhanced_ilc_fg = observed_cmb - enhanced_ilc_cmb
    vanilla_ilc_fg = observed_cmb - vanilla_ilc_cmb
    
    # Verify logic: enhanced_ilc_fg should equal b_fg_220 - enhanced_ilc_resid
    # Note: enhanced_ilc_resid = enhanced_ilc_cmb - scaled_cmb, so:
    # enhanced_ilc_fg = observed_cmb - enhanced_ilc_cmb
    #                 = (scaled_cmb + b_fg_220) - enhanced_ilc_cmb
    #                 = b_fg_220 + (scaled_cmb - enhanced_ilc_cmb)
    #                 = b_fg_220 - (enhanced_ilc_cmb - scaled_cmb)
    #                 = b_fg_220 - enhanced_ilc_resid
    print("\n  Verifying logic for enhanced_ilc_fg...")
    # The correct relationship is: enhanced_ilc_fg = b_fg_220 - enhanced_ilc_resid
    expected_enhanced_fg = b_fg_220 - enhanced_ilc_resid
    diff_enhanced = np.abs(enhanced_ilc_fg - expected_enhanced_fg)
    max_diff_enhanced = np.max(diff_enhanced)
    print(f"    enhanced_ilc_fg = observed_cmb - enhanced_ilc_cmb")
    print(f"    = (scaled_cmb + b_fg_220) - enhanced_ilc_cmb")
    print(f"    = b_fg_220 + (scaled_cmb - enhanced_ilc_cmb)")
    print(f"    = b_fg_220 - enhanced_ilc_resid")
    print(f"    Max difference from expected: {max_diff_enhanced:.6e}")
    if max_diff_enhanced > 1e-10:
        print(f"    ⚠ WARNING: Logic verification failed! Difference is larger than expected.")
    else:
        print(f"    ✓ Logic verified!")
    
    print(f"  Vanilla ILC foreground shape: {vanilla_ilc_fg.shape}")
    print(f"    Mean: {np.mean(vanilla_ilc_fg):.6e}, Std: {np.std(vanilla_ilc_fg):.6e}")
    print(f"  Enhanced ILC foreground shape: {enhanced_ilc_fg.shape}")
    print(f"    Mean: {np.mean(enhanced_ilc_fg):.6e}, Std: {np.std(enhanced_ilc_fg):.6e}")
    
    # all_scales_b_fg is the same as b_fg_220 (all-scales B-mode foreground at 220 GHz)
    all_scales_b_fg = b_fg_220
    
    # Step 4: Compute normalized cross-spectra
    print("\nStep 4: Computing normalized cross-spectra...")
    print(f"  Computing for {n_patches} patches...")
    print(f"  Parameters: delta_ell={DELTA_ELL}, ell_max={ELL_MAX}, pix_size={PIX_SIZE}, N={PATCH_SIZE}")
    
    if calculate_2d_spectrum is None:
        print("  ⚠ WARNING: calculate_2d_spectrum not available - skipping cross-spectra computation")
        # Still compute MSE even if cross-spectra not available
        print("\nStep 5: Computing MSE between reconstructed foregrounds and true foregrounds...")
        vanilla_fg_mse_list = []
        enhanced_fg_mse_list = []
        
        for patch_idx in tqdm(range(n_patches), desc="Foreground MSE"):
            try:
                vanilla_fg_patch = vanilla_ilc_fg[patch_idx]
                enhanced_fg_patch = enhanced_ilc_fg[patch_idx]
                true_fg_patch = all_scales_b_fg[patch_idx]
                
                # Compute MSE: vanilla_ilc_fg vs all_scales_b_fg
                vanilla_mse = compute_mse(vanilla_fg_patch, true_fg_patch)
                vanilla_fg_mse_list.append(vanilla_mse)
                
                # Compute MSE: enhanced_ilc_fg vs all_scales_b_fg
                enhanced_mse = compute_mse(enhanced_fg_patch, true_fg_patch)
                enhanced_fg_mse_list.append(enhanced_mse)
                
            except Exception as e:
                print(f"    Warning: Error computing MSE for patch {patch_idx}: {e}")
                vanilla_fg_mse_list.append(np.nan)
                enhanced_fg_mse_list.append(np.nan)
        
        vanilla_fg_mse_array = np.array(vanilla_fg_mse_list)
        enhanced_fg_mse_array = np.array(enhanced_fg_mse_list)
        
        # Compute mean and std across patches
        mean_vanilla_fg_mse = np.nanmean(vanilla_fg_mse_array)
        std_vanilla_fg_mse = np.nanstd(vanilla_fg_mse_array)
        mean_enhanced_fg_mse = np.nanmean(enhanced_fg_mse_array)
        std_enhanced_fg_mse = np.nanstd(enhanced_fg_mse_array)
        
        print(f"\n  ✓ Computed foreground MSE for {n_patches} patches")
        print(f"    Vanilla ILC foreground MSE:")
        print(f"      Mean: {mean_vanilla_fg_mse:.6e}, Std: {std_vanilla_fg_mse:.6e}")
        print(f"    Enhanced ILC foreground MSE:")
        print(f"      Mean: {mean_enhanced_fg_mse:.6e}, Std: {std_enhanced_fg_mse:.6e}")
        
        return {
            'vanilla_ilc_resid': vanilla_ilc_resid,
            'enhanced_ilc_resid': enhanced_ilc_resid,
            'observed_cmb': observed_cmb,
            'vanilla_ilc_fg': vanilla_ilc_fg,
            'enhanced_ilc_fg': enhanced_ilc_fg,
            'all_scales_b_fg': all_scales_b_fg,
            'vanilla_cross_spectra_array': None,
            'enhanced_cross_spectra_array': None,
            'ell_array': None,
            'mean_vanilla_cross_ps': None,
            'std_vanilla_cross_ps': None,
            'mean_enhanced_cross_ps': None,
            'std_enhanced_cross_ps': None,
            'vanilla_fg_mse_array': vanilla_fg_mse_array,
            'enhanced_fg_mse_array': enhanced_fg_mse_array,
            'mean_vanilla_fg_mse': mean_vanilla_fg_mse,
            'std_vanilla_fg_mse': std_vanilla_fg_mse,
            'mean_enhanced_fg_mse': mean_enhanced_fg_mse,
            'std_enhanced_fg_mse': std_enhanced_fg_mse,
        }
    
    vanilla_cross_spectra_list = []
    enhanced_cross_spectra_list = []
    ell_array_ref = None
    
    for patch_idx in tqdm(range(n_patches), desc="Cross-spectra"):
        try:
            vanilla_fg_patch = vanilla_ilc_fg[patch_idx]
            enhanced_fg_patch = enhanced_ilc_fg[patch_idx]
            true_fg_patch = all_scales_b_fg[patch_idx]
            
            # Compute cross-spectrum: vanilla_ilc_fg × all_scales_b_fg
            ell_array, vanilla_cross_ps = compute_normalized_cross_spectrum(
                vanilla_fg_patch, true_fg_patch
            )
            vanilla_cross_spectra_list.append(vanilla_cross_ps)
            
            # Compute cross-spectrum: enhanced_ilc_fg × all_scales_b_fg
            _, enhanced_cross_ps = compute_normalized_cross_spectrum(
                enhanced_fg_patch, true_fg_patch
            )
            enhanced_cross_spectra_list.append(enhanced_cross_ps)
            
            # Store ell_array reference from first successful computation
            if ell_array_ref is None:
                ell_array_ref = ell_array.copy()
                
        except Exception as e:
            print(f"    Warning: Error computing cross-spectrum for patch {patch_idx}: {e}")
            # Append NaN arrays to maintain alignment
            if ell_array_ref is not None:
                nan_array = np.full_like(ell_array_ref, np.nan)
            else:
                # Estimate size from parameters
                estimated_nbins = int(ELL_MAX / DELTA_ELL)
                nan_array = np.full(estimated_nbins, np.nan, dtype=float)
            
            vanilla_cross_spectra_list.append(nan_array.copy())
            enhanced_cross_spectra_list.append(nan_array.copy())
    
    vanilla_cross_spectra_array = np.array(vanilla_cross_spectra_list)
    enhanced_cross_spectra_array = np.array(enhanced_cross_spectra_list)
    
    # Compute mean and std across patches
    mean_vanilla_cross_ps = np.nanmean(vanilla_cross_spectra_array, axis=0)
    std_vanilla_cross_ps = np.nanstd(vanilla_cross_spectra_array, axis=0)
    mean_enhanced_cross_ps = np.nanmean(enhanced_cross_spectra_array, axis=0)
    std_enhanced_cross_ps = np.nanstd(enhanced_cross_spectra_array, axis=0)
    
    print(f"\n  ✓ Computed cross-spectra for {n_patches} patches")
    print(f"    Vanilla cross-spectra shape: {vanilla_cross_spectra_array.shape}")
    print(f"    Enhanced cross-spectra shape: {enhanced_cross_spectra_array.shape}")
    print(f"    Mean vanilla cross-spectrum (across ell): {np.nanmean(mean_vanilla_cross_ps):.6e}")
    print(f"    Mean enhanced cross-spectrum (across ell): {np.nanmean(mean_enhanced_cross_ps):.6e}")
    
    # Step 5: Compute MSE between reconstructed foregrounds and true foregrounds
    print("\nStep 5: Computing MSE between reconstructed foregrounds and true foregrounds...")
    vanilla_fg_mse_list = []
    enhanced_fg_mse_list = []
    
    for patch_idx in tqdm(range(n_patches), desc="Foreground MSE"):
        try:
            vanilla_fg_patch = vanilla_ilc_fg[patch_idx]
            enhanced_fg_patch = enhanced_ilc_fg[patch_idx]
            true_fg_patch = all_scales_b_fg[patch_idx]
            
            # Compute MSE: vanilla_ilc_fg vs all_scales_b_fg
            vanilla_mse = compute_mse(vanilla_fg_patch, true_fg_patch)
            vanilla_fg_mse_list.append(vanilla_mse)
            
            # Compute MSE: enhanced_ilc_fg vs all_scales_b_fg
            enhanced_mse = compute_mse(enhanced_fg_patch, true_fg_patch)
            enhanced_fg_mse_list.append(enhanced_mse)
            
        except Exception as e:
            print(f"    Warning: Error computing MSE for patch {patch_idx}: {e}")
            vanilla_fg_mse_list.append(np.nan)
            enhanced_fg_mse_list.append(np.nan)
    
    vanilla_fg_mse_array = np.array(vanilla_fg_mse_list)
    enhanced_fg_mse_array = np.array(enhanced_fg_mse_list)
    
    # Compute mean and std across patches
    mean_vanilla_fg_mse = np.nanmean(vanilla_fg_mse_array)
    std_vanilla_fg_mse = np.nanstd(vanilla_fg_mse_array)
    mean_enhanced_fg_mse = np.nanmean(enhanced_fg_mse_array)
    std_enhanced_fg_mse = np.nanstd(enhanced_fg_mse_array)
    
    print(f"\n  ✓ Computed foreground MSE for {n_patches} patches")
    print(f"    Vanilla ILC foreground MSE:")
    print(f"      Mean: {mean_vanilla_fg_mse:.6e}, Std: {std_vanilla_fg_mse:.6e}")
    print(f"    Enhanced ILC foreground MSE:")
    print(f"      Mean: {mean_enhanced_fg_mse:.6e}, Std: {std_enhanced_fg_mse:.6e}")
    
    return {
        'vanilla_ilc_resid': vanilla_ilc_resid,
        'enhanced_ilc_resid': enhanced_ilc_resid,
        'observed_cmb': observed_cmb,
        'vanilla_ilc_fg': vanilla_ilc_fg,
        'enhanced_ilc_fg': enhanced_ilc_fg,
        'all_scales_b_fg': all_scales_b_fg,
        'vanilla_cross_spectra_array': vanilla_cross_spectra_array,
        'enhanced_cross_spectra_array': enhanced_cross_spectra_array,
        'ell_array': ell_array_ref,
        'mean_vanilla_cross_ps': mean_vanilla_cross_ps,
        'std_vanilla_cross_ps': std_vanilla_cross_ps,
        'mean_enhanced_cross_ps': mean_enhanced_cross_ps,
        'std_enhanced_cross_ps': std_enhanced_cross_ps,
        'vanilla_fg_mse_array': vanilla_fg_mse_array,
        'enhanced_fg_mse_array': enhanced_fg_mse_array,
        'mean_vanilla_fg_mse': mean_vanilla_fg_mse,
        'std_vanilla_fg_mse': std_vanilla_fg_mse,
        'mean_enhanced_fg_mse': mean_enhanced_fg_mse,
        'std_enhanced_fg_mse': std_enhanced_fg_mse,
    }


def save_cross_spectra_statistics(cross_spectra_results, output_dir, model_identifier):
    """
    Save cross-spectra statistics to text files.
    
    Args:
        cross_spectra_results: Dictionary from compute_cross_spectra_analysis()
        output_dir: Output directory for saving files
        model_identifier: Model identifier string for filename
    """
    print("\n" + "="*80)
    print("SAVING CROSS-SPECTRA STATISTICS")
    print("="*80)
    
    if cross_spectra_results['ell_array'] is None:
        print("  ⚠ WARNING: No cross-spectra data available - skipping save")
        return
    
    ell_array = cross_spectra_results['ell_array']
    vanilla_cross_spectra_array = cross_spectra_results['vanilla_cross_spectra_array']
    enhanced_cross_spectra_array = cross_spectra_results['enhanced_cross_spectra_array']
    mean_vanilla_cross_ps = cross_spectra_results['mean_vanilla_cross_ps']
    std_vanilla_cross_ps = cross_spectra_results['std_vanilla_cross_ps']
    mean_enhanced_cross_ps = cross_spectra_results['mean_enhanced_cross_ps']
    std_enhanced_cross_ps = cross_spectra_results['std_enhanced_cross_ps']
    
    # Save vanilla ILC cross-spectrum statistics
    vanilla_file = os.path.join(output_dir, f'{model_identifier}_vanilla_ilc_cross_spectrum_stats.txt')
    print(f"\nSaving vanilla ILC cross-spectrum statistics to: {vanilla_file}")
    
    with open(vanilla_file, 'w') as f:
        f.write(f"Vanilla ILC Foreground Cross-Power Spectrum Statistics\n")
        f.write(f"Model: {model_identifier}\n")
        f.write(f"Cross-spectrum: vanilla_ilc_fg × all_scales_b_fg\n")
        f.write(f"Number of patches: {len(vanilla_cross_spectra_array)}\n")
        f.write(f"Ell bins: {len(mean_vanilla_cross_ps)}\n")
        f.write(f"\n{'='*80}\n\n")
        
        # Include ell array for plotting
        f.write("Ell (multipole) array:\n")
        f.write("  [")
        for i, ell in enumerate(ell_array):
            if i < len(ell_array) - 1:
                f.write(f"{ell:.1f}, ")
            else:
                f.write(f"{ell:.1f}")
            # Add line breaks for readability (every 8 values)
            if (i + 1) % 8 == 0 and i < len(ell_array) - 1:
                f.write("\n   ")
        f.write("]\n")
        f.write(f"\n{'='*80}\n\n")
        
        # Mean and std across patches for each ell bin
        f.write("Mean cross-spectrum (per ell bin, averaged across patches):\n")
        f.write(f"  Mean across ell: {np.nanmean(mean_vanilla_cross_ps):.6e}\n")
        f.write(f"  Std across ell: {np.nanstd(mean_vanilla_cross_ps):.6e}\n")
        
        f.write(f"\nPer-ell-bin statistics (mean ± std across patches):\n")
        for i, ell in enumerate(ell_array):
            if not np.isnan(mean_vanilla_cross_ps[i]):
                f.write(f"  ell={ell:6.1f}: {mean_vanilla_cross_ps[i]:.6e} ± {std_vanilla_cross_ps[i]:.6e}\n")
        
        f.write(f"\n\nStd cross-spectrum (per ell bin, std across patches):\n")
        f.write(f"  Mean std (across ell): {np.nanmean(std_vanilla_cross_ps):.6e}\n")
        f.write(f"  Std of std (across ell): {np.nanstd(std_vanilla_cross_ps):.6e}\n")
        
        f.write(f"\n\nOverall statistics (across all patches and ell bins):\n")
        f.write(f"  Mean cross-spectrum: {np.nanmean(vanilla_cross_spectra_array):.6e}\n")
        f.write(f"  Std cross-spectrum: {np.nanstd(vanilla_cross_spectra_array):.6e}\n")
        
        # Also save in a format that's easy to parse for plotting
        f.write(f"\n\n{'='*80}\n")
        f.write("Data for plotting (ell, mean_cross_ps, std_cross_ps):\n")
        f.write("Format: ell  mean_cross_ps  std_cross_ps\n")
        f.write(f"{'='*80}\n")
        for i, ell in enumerate(ell_array):
            if not np.isnan(mean_vanilla_cross_ps[i]):
                f.write(f"{ell:6.1f}  {mean_vanilla_cross_ps[i]:.6e}  {std_vanilla_cross_ps[i]:.6e}\n")
    
    print(f"  ✓ Saved vanilla ILC cross-spectrum statistics")
    
    # Save enhanced ILC cross-spectrum statistics
    enhanced_file = os.path.join(output_dir, f'{model_identifier}_enhanced_ilc_cross_spectrum_stats.txt')
    print(f"\nSaving enhanced ILC cross-spectrum statistics to: {enhanced_file}")
    
    with open(enhanced_file, 'w') as f:
        f.write(f"Enhanced ILC (UNet-ILC) Foreground Cross-Power Spectrum Statistics\n")
        f.write(f"Model: {model_identifier}\n")
        f.write(f"Cross-spectrum: enhanced_ilc_fg × all_scales_b_fg\n")
        f.write(f"Number of patches: {len(enhanced_cross_spectra_array)}\n")
        f.write(f"Ell bins: {len(mean_enhanced_cross_ps)}\n")
        f.write(f"\n{'='*80}\n\n")
        
        # Include ell array for plotting
        f.write("Ell (multipole) array:\n")
        f.write("  [")
        for i, ell in enumerate(ell_array):
            if i < len(ell_array) - 1:
                f.write(f"{ell:.1f}, ")
            else:
                f.write(f"{ell:.1f}")
            # Add line breaks for readability (every 8 values)
            if (i + 1) % 8 == 0 and i < len(ell_array) - 1:
                f.write("\n   ")
        f.write("]\n")
        f.write(f"\n{'='*80}\n\n")
        
        # Mean and std across patches for each ell bin
        f.write("Mean cross-spectrum (per ell bin, averaged across patches):\n")
        f.write(f"  Mean across ell: {np.nanmean(mean_enhanced_cross_ps):.6e}\n")
        f.write(f"  Std across ell: {np.nanstd(mean_enhanced_cross_ps):.6e}\n")
        
        f.write(f"\nPer-ell-bin statistics (mean ± std across patches):\n")
        for i, ell in enumerate(ell_array):
            if not np.isnan(mean_enhanced_cross_ps[i]):
                f.write(f"  ell={ell:6.1f}: {mean_enhanced_cross_ps[i]:.6e} ± {std_enhanced_cross_ps[i]:.6e}\n")
        
        f.write(f"\n\nStd cross-spectrum (per ell bin, std across patches):\n")
        f.write(f"  Mean std (across ell): {np.nanmean(std_enhanced_cross_ps):.6e}\n")
        f.write(f"  Std of std (across ell): {np.nanstd(std_enhanced_cross_ps):.6e}\n")
        
        f.write(f"\n\nOverall statistics (across all patches and ell bins):\n")
        f.write(f"  Mean cross-spectrum: {np.nanmean(enhanced_cross_spectra_array):.6e}\n")
        f.write(f"  Std cross-spectrum: {np.nanstd(enhanced_cross_spectra_array):.6e}\n")
        
        # Also save in a format that's easy to parse for plotting
        f.write(f"\n\n{'='*80}\n")
        f.write("Data for plotting (ell, mean_cross_ps, std_cross_ps):\n")
        f.write("Format: ell  mean_cross_ps  std_cross_ps\n")
        f.write(f"{'='*80}\n")
        for i, ell in enumerate(ell_array):
            if not np.isnan(mean_enhanced_cross_ps[i]):
                f.write(f"{ell:6.1f}  {mean_enhanced_cross_ps[i]:.6e}  {std_enhanced_cross_ps[i]:.6e}\n")
    
    print(f"  ✓ Saved enhanced ILC cross-spectrum statistics")


def save_fg_mse_statistics(cross_spectra_results, output_dir, model_identifier):
    """
    Save foreground MSE statistics to text files.
    
    Args:
        cross_spectra_results: Dictionary from compute_cross_spectra_analysis()
        output_dir: Output directory for saving files
        model_identifier: Model identifier string for filename
    """
    print("\n" + "="*80)
    print("SAVING FOREGROUND MSE STATISTICS")
    print("="*80)
    
    if 'vanilla_fg_mse_array' not in cross_spectra_results or cross_spectra_results['vanilla_fg_mse_array'] is None:
        print("  ⚠ WARNING: No foreground MSE data available - skipping save")
        return
    
    vanilla_fg_mse_array = cross_spectra_results['vanilla_fg_mse_array']
    enhanced_fg_mse_array = cross_spectra_results['enhanced_fg_mse_array']
    mean_vanilla_fg_mse = cross_spectra_results['mean_vanilla_fg_mse']
    std_vanilla_fg_mse = cross_spectra_results['std_vanilla_fg_mse']
    mean_enhanced_fg_mse = cross_spectra_results['mean_enhanced_fg_mse']
    std_enhanced_fg_mse = cross_spectra_results['std_enhanced_fg_mse']
    
    # Save vanilla ILC foreground MSE statistics
    vanilla_file = os.path.join(output_dir, f'{model_identifier}_vanilla_ilc_fg_mse_stats.txt')
    print(f"\nSaving vanilla ILC foreground MSE statistics to: {vanilla_file}")
    
    with open(vanilla_file, 'w') as f:
        f.write(f"Vanilla ILC Foreground MSE Statistics\n")
        f.write(f"Model: {model_identifier}\n")
        f.write(f"MSE: vanilla_ilc_fg vs all_scales_b_fg\n")
        f.write(f"Number of patches: {len(vanilla_fg_mse_array)}\n")
        f.write(f"\n{'='*80}\n\n")
        
        f.write(f"Mean MSE: {mean_vanilla_fg_mse:.6e}\n")
        f.write(f"Std MSE: {std_vanilla_fg_mse:.6e}\n")
        f.write(f"\nMin MSE: {np.nanmin(vanilla_fg_mse_array):.6e}\n")
        f.write(f"Max MSE: {np.nanmax(vanilla_fg_mse_array):.6e}\n")
        f.write(f"Median MSE: {np.nanmedian(vanilla_fg_mse_array):.6e}\n")
    
    print(f"  ✓ Saved vanilla ILC foreground MSE statistics")
    
    # Save enhanced ILC foreground MSE statistics
    enhanced_file = os.path.join(output_dir, f'{model_identifier}_enhanced_ilc_fg_mse_stats.txt')
    print(f"\nSaving enhanced ILC foreground MSE statistics to: {enhanced_file}")
    
    with open(enhanced_file, 'w') as f:
        f.write(f"Enhanced ILC (UNet-ILC) Foreground MSE Statistics\n")
        f.write(f"Model: {model_identifier}\n")
        f.write(f"MSE: enhanced_ilc_fg vs all_scales_b_fg\n")
        f.write(f"Number of patches: {len(enhanced_fg_mse_array)}\n")
        f.write(f"\n{'='*80}\n\n")
        
        f.write(f"Mean MSE: {mean_enhanced_fg_mse:.6e}\n")
        f.write(f"Std MSE: {std_enhanced_fg_mse:.6e}\n")
        f.write(f"\nMin MSE: {np.nanmin(enhanced_fg_mse_array):.6e}\n")
        f.write(f"Max MSE: {np.nanmax(enhanced_fg_mse_array):.6e}\n")
        f.write(f"Median MSE: {np.nanmedian(enhanced_fg_mse_array):.6e}\n")
        
        # Also include improvement ratio if vanilla MSE is available
        if mean_vanilla_fg_mse > 0:
            improvement_ratio = mean_vanilla_fg_mse / mean_enhanced_fg_mse
            f.write(f"\nImprovement ratio (vanilla_MSE / enhanced_MSE): {improvement_ratio:.4f}x\n")
            f.write(f"Enhanced ILC MSE is {improvement_ratio:.2f}x better (lower) than vanilla ILC\n")
    
    print(f"  ✓ Saved enhanced ILC foreground MSE statistics")


def visualize_comparison(results, output_dir, sample_indices=None):
    """
    Create visualization comparing vanilla vs enhanced ILC.
    
    Args:
        results: Dictionary from analyze_results()
        output_dir: Output directory for plots
        sample_indices: List of sample indices to plot (default: first 5)
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    if sample_indices is None:
        sample_indices = list(range(min(5, len(results['primary_cmb']))))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sample comparison plots
    print(f"\nCreating sample comparison plots for {len(sample_indices)} samples...")
    for idx in sample_indices:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        primary = results['primary_cmb'][idx]
        vanilla = results['vanilla_ilc_cmb'][idx]
        enhanced = results['enhanced_ilc_cmb'][idx]
        
        v_mse = results['vanilla_mse'][idx]
        e_mse = results['enhanced_mse'][idx]
        v_corr = results['vanilla_corr'][idx]
        e_corr = results['enhanced_corr'][idx]
        
        # Primary CMB
        im0 = axes[0].imshow(primary, cmap='RdBu_r', origin='lower')
        axes[0].set_title(f'Primary CMB\nSample {idx}', fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # Vanilla ILC
        im1 = axes[1].imshow(vanilla, cmap='RdBu_r', origin='lower')
        axes[1].set_title(f'ILC\nMSE: {v_mse:.3e}\nCorr: {v_corr:.4f}', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Enhanced ILC
        im2 = axes[2].imshow(enhanced, cmap='RdBu_r', origin='lower')
        axes[2].set_title(f'UNet-ILC\nMSE: {e_mse:.3e}\nCorr: {e_corr:.4f}', fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        # Difference (Vanilla - Enhanced)
        diff = vanilla - enhanced
        im3 = axes[3].imshow(diff, cmap='RdBu_r', origin='lower')
        axes[3].set_title(f'Difference\n(ILC - UNet-ILC)', fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_{idx}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. MSE comparison scatter plot
    print("\nCreating MSE comparison scatter plot...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    v_mse = results['vanilla_mse']
    e_mse = results['enhanced_mse']
    
    # Filter out zero or negative values for log scale
    valid_mask = (v_mse > 0) & (e_mse > 0)
    v_mse_valid = v_mse[valid_mask]
    e_mse_valid = e_mse[valid_mask]
    
    if len(v_mse_valid) == 0:
        print("  Warning: No valid MSE values > 0 for log scale plot")
        v_mse_valid = v_mse
        e_mse_valid = e_mse
    
    # Set log scale on both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Color points by whether they're above or below diagonal (improved vs not improved)
    improved_mask = e_mse_valid < v_mse_valid
    not_improved_mask = ~improved_mask
    
    # Plot points - improved in green, not improved in red
    if np.any(improved_mask):
        ax.scatter(v_mse_valid[improved_mask], e_mse_valid[improved_mask], 
                   alpha=0.5, s=30, c='green', edgecolors='darkgreen', linewidths=0.3,
                   label=fr'UNet-ILC $\leq$ ILC (n={np.sum(improved_mask)})', zorder=2)
    
    if np.any(not_improved_mask):
        ax.scatter(v_mse_valid[not_improved_mask], e_mse_valid[not_improved_mask], 
                   alpha=0.5, s=30, c='red', edgecolors='darkred', linewidths=0.3,
                   label=fr'UNet-ILC $\geq$ ILC (n={np.sum(not_improved_mask)})', zorder=2)
    
    # Diagonal line (y=x) on log scale
    mse_min = max(v_mse_valid.min(), e_mse_valid.min(), 1e-10)
    mse_max = max(v_mse_valid.max(), e_mse_valid.max())
    ax.plot([mse_min, mse_max], [mse_min, mse_max], 'k--', lw=2, 
            label='y=x (no change)', alpha=0.8, zorder=3)
    
    ax.set_xlabel('ILC MSE', fontsize=14)
    ax.set_ylabel('UNet-ILC MSE', fontsize=14)
    ax.set_title('MSE Comparison: ILC vs UNet-ILC', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=14)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)  # Both major and minor grid for log scale
    ax.set_aspect('equal', adjustable='box')
    
    # Add improvement text
    improvement_pct = (1 - np.mean(e_mse)/np.mean(v_mse)) * 100
    n_improved = np.sum(e_mse < v_mse)
    n_total = len(e_mse)
    ax.text(0.96, 0.02, rf'UNet-ILC $\leq$ ILC: {100*n_improved/n_total:.1f}$\%$',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mse_comparison_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation comparison
    print("\nCreating correlation comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    v_corr = results['vanilla_corr']
    e_corr = results['enhanced_corr']
    
    # Histograms with log scales for better visualization
    # Filter out any invalid correlations
    v_corr_valid = v_corr[~np.isnan(v_corr) & ~np.isinf(v_corr)]
    e_corr_valid = e_corr[~np.isnan(e_corr) & ~np.isinf(e_corr)]
    
    # Use more bins for better resolution
    n_bins = 50
    corr_min = min(v_corr_valid.min(), e_corr_valid.min())
    corr_max = max(v_corr_valid.max(), e_corr_valid.max())
    
    # Create bins - use linear bins since correlations are already in reasonable range
    bins = np.linspace(corr_min, corr_max, n_bins)
    
    # Use histogram step plot with thick lines for clearer visualization
    # Use hist() with step style and log y-scale
    axes[0].hist(v_corr_valid, bins=bins, alpha=0.8, histtype='step', 
                 label='ILC', color='blue', linewidth=3, log=True)
    axes[0].hist(e_corr_valid, bins=bins, alpha=0.8, histtype='step', 
                 label='UNet-ILC', color='green', linewidth=3, log=True)
    
    # Mark mean values with vertical lines
    v_mean = np.mean(v_corr_valid)
    e_mean = np.mean(e_corr_valid)
    axes[0].axvline(v_mean, color='blue', linestyle='--', lw=2, 
                    label=f'ILC mean: {v_mean:.4f}', alpha=0.8)
    axes[0].axvline(e_mean, color='green', linestyle='--', lw=2, 
                    label=f'UNet-ILC mean: {e_mean:.4f}', alpha=0.8)
    
    axes[0].set_xlabel('Spatial Correlation', fontsize=14)
    axes[0].set_ylabel('Frequency', fontsize=14)
    axes[0].set_title('Correlation Distribution', fontsize=14, fontweight='bold')
    axes[0].set_yscale('log')  # Log scale on y-axis
    axes[0].legend(fontsize=14)
    axes[0].grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)  # Both major and minor grid for log scale
    
    # Scatter plot with finer-grained hexbin
    # gridsize controls resolution: 50 (current), 75 (finer), 100 (very fine), 150+ (extremely fine)
    hb = axes[1].hexbin(v_corr, e_corr, gridsize=120, cmap='Blues', mincnt=1,
                        linewidths=0, edgecolors='none')
    plt.colorbar(hb, ax=axes[1], label='Count')
    corr_min = min(v_corr.min(), e_corr.min())
    corr_max = max(v_corr.max(), e_corr.max())
    # Set axis limits to start at 0.995 for better visualization
    axes[1].set_xlim(0.995, corr_max)
    axes[1].set_ylim(0.995, corr_max)
    axes[1].plot([0.995, corr_max], [0.995, corr_max], 'r--', lw=2, label='y=x', alpha=0.8)
    axes[1].set_xlabel('ILC Correlation', fontsize=14)
    axes[1].set_ylabel('UNet-ILC Correlation', fontsize=14)
    axes[1].set_title('Correlation Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. ILC weights comparison
    print("\nCreating ILC weights comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    v_weights = results['vanilla_ilc_weights']
    e_weights = results['enhanced_ilc_weights']
    
    n_freq = v_weights.shape[1]
    
    # Vanilla weights boxplot
    axes[0].boxplot([v_weights[:, i] for i in range(n_freq)], 
                    labels=[f'{FREQUENCIES[i]} GHz' for i in range(n_freq)])
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('ILC Weight', fontsize=14)
    axes[0].set_title('ILC Weights', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Enhanced weights boxplot
    labels_enhanced = [f'{FREQUENCIES[i]} GHz' for i in range(n_freq)] + ['UNet']
    axes[1].boxplot([e_weights[:, i] for i in range(n_freq + 1)], labels=labels_enhanced)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('UNet-ILC Weight', fontsize=14)
    axes[1].set_title('UNet-ILC Weights', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ilc_weights_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Variance comparison (actual variance of ILC CMB reconstruction)
    print("\nCreating variance comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    v_var = results['vanilla_ilc_variances']
    e_var = results['enhanced_ilc_variances']
    
    # Scatter plot comparing variances
    ax = axes[0]
    valid_mask = (v_var > 0) & (e_var > 0)
    v_var_valid = v_var[valid_mask]
    e_var_valid = e_var[valid_mask]
    
    # Set log scale on both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Color points by whether enhanced has lower variance
    improved_mask = e_var_valid < v_var_valid
    equal_or_worse_mask = ~improved_mask
    
    if np.any(improved_mask):
        ax.scatter(v_var_valid[improved_mask], e_var_valid[improved_mask], 
                   alpha=0.5, s=30, c='green', edgecolors='darkgreen', linewidths=0.3,
                   label=fr'Enhanced < Vanilla (n={np.sum(improved_mask)})', zorder=2)
    
    if np.any(equal_or_worse_mask):
        ax.scatter(v_var_valid[equal_or_worse_mask], e_var_valid[equal_or_worse_mask], 
                   alpha=0.5, s=30, c='red', edgecolors='darkred', linewidths=0.3,
                   label=fr'Enhanced ≥ Vanilla (n={np.sum(equal_or_worse_mask)})', zorder=2)
    
    # Diagonal line (y=x)
    var_min = max(v_var_valid.min(), e_var_valid.min(), 1e-15)
    var_max = max(v_var_valid.max(), e_var_valid.max())
    ax.plot([var_min, var_max], [var_min, var_max], 'k--', lw=2, 
            label='y=x (no change)', alpha=0.8, zorder=3)
    
    ax.set_xlabel('Vanilla ILC CMB Variance', fontsize=14)
    ax.set_ylabel('Enhanced ILC CMB Variance', fontsize=14)
    ax.set_title('ILC CMB Reconstruction Variance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics text
    n_lower = np.sum(e_var < v_var)
    n_equal = np.sum(np.abs(e_var - v_var) < 1e-10 * np.maximum(e_var, v_var))
    n_higher = np.sum(e_var > v_var)
    n_total = len(e_var)
    mean_reduction = (1 - np.mean(e_var)/np.mean(v_var)) * 100
    
    stats_text = (f'Enhanced < Vanilla: {n_lower}/{n_total} ({100*n_lower/n_total:.1f}%)\n'
                  f'Enhanced = Vanilla: {n_equal}/{n_total} ({100*n_equal/n_total:.1f}%)\n'
                  f'Enhanced > Vanilla: {n_higher}/{n_total} ({100*n_higher/n_total:.1f}%)\n'
                  f'Mean reduction: {mean_reduction:.2f}%')
    ax.text(0.96, 0.02, stats_text, transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)
    
    # Histogram of variance ratios
    ax = axes[1]
    variance_ratio = e_var / v_var
    variance_ratio_valid = variance_ratio[valid_mask]
    
    bins = np.logspace(np.log10(variance_ratio_valid.min()), np.log10(variance_ratio_valid.max()), 50)
    ax.hist(variance_ratio_valid, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(1.0, color='red', linestyle='--', lw=2, label='No change (ratio=1)', zorder=3)
    ax.axvline(np.mean(variance_ratio_valid), color='green', linestyle='--', lw=2, 
               label=f'Mean ratio: {np.mean(variance_ratio_valid):.4f}', zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('Variance Ratio (Enhanced / Vanilla)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Distribution of Variance Ratios', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ilc_variance_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate UNet-enhanced ILC vs vanilla ILC')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained UNet model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--cmb-scale', type=float, default=0.01,
                       help='Scaling factor for primary CMB (default: 0.01)')
    parser.add_argument('--normalize', action='store_true',
                       help='Use normalized model (default: unnormalized)')
    parser.add_argument('--b-only', action='store_true',
                       help='Use B-only model (1 channel), skip T and E channels')
    parser.add_argument('--n-test', type=int, default=None,
                       help='Number of test samples to process (default: all)')
    parser.add_argument('--sample-indices', type=str, default=None,
                       help='Comma-separated list of sample indices to visualize (default: first 5)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("UNET-ENHANCED ILC EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  CMB scaling factor: {args.cmb_scale}")
    print(f"  Normalization: {'Yes' if args.normalize else 'No'}")
    print(f"  Input channels: {'B only (1 channel)' if args.b_only else 'B + T + E (3 channels)'}")
    print(f"  Frequencies: {FREQUENCIES}")
    
    # Load test indices
    stats = np.load(STATS_FILE)
    test_indices = stats['test_indices']
    
    if args.n_test is not None:
        test_indices = test_indices[:args.n_test]
    
    print(f"\n  Test samples: {len(test_indices)}")
    
    # Load data
    freq_patches_dict = load_multifreq_data(FREQUENCIES, test_indices)
    primary_cmb = load_primary_cmb(test_indices, REFERENCE_FREQ)
    b_small = load_unet_components(test_indices, REFERENCE_FREQ)
    unet_predictions = load_model_and_generate_predictions(
        args.model_path, test_indices, 
        normalize=args.normalize,
        use_te=(not args.b_only),
        reference_freq=REFERENCE_FREQ
    )
    
    # Scale primary CMB
    print(f"\nScaling primary CMB by factor: {args.cmb_scale}")
    scaled_cmb = primary_cmb * args.cmb_scale
    print(f"  Scaled CMB range: [{scaled_cmb.min():.3e}, {scaled_cmb.max():.3e}]")
    
    # Apply vanilla ILC
    vanilla_ilc_cmb, vanilla_ilc_weights, vanilla_ilc_residuals, vanilla_ilc_variances = apply_vanilla_ilc(
        freq_patches_dict, scaled_cmb, FREQUENCIES
    )
    
    # Apply enhanced ILC
    enhanced_ilc_cmb, enhanced_ilc_weights, enhanced_ilc_residuals, unet_synthetic, enhanced_ilc_variances = apply_enhanced_ilc(
        freq_patches_dict, scaled_cmb, unet_predictions, b_small, FREQUENCIES
    )
    
    # Analyze results
    results = analyze_results(
        vanilla_ilc_cmb, vanilla_ilc_residuals, vanilla_ilc_weights, vanilla_ilc_variances,
        enhanced_ilc_cmb, enhanced_ilc_residuals, enhanced_ilc_weights, enhanced_ilc_variances,
        scaled_cmb, FREQUENCIES
    )
    
    # Compute cross-spectra analysis
    cross_spectra_results = compute_cross_spectra_analysis(
        vanilla_ilc_cmb, enhanced_ilc_cmb, primary_cmb, scaled_cmb,
        freq_patches_dict, reference_freq=REFERENCE_FREQ
    )
    
    # Save results
    print(f"\nSaving results to: {args.output_dir}")
    results_file = os.path.join(args.output_dir, 'ilc_comparison_results.npz')
    np.savez(
        results_file,
        vanilla_ilc_cmb=vanilla_ilc_cmb,
        enhanced_ilc_cmb=enhanced_ilc_cmb,
        vanilla_ilc_residuals=vanilla_ilc_residuals,
        enhanced_ilc_residuals=enhanced_ilc_residuals,
        vanilla_ilc_weights=vanilla_ilc_weights,
        enhanced_ilc_weights=enhanced_ilc_weights,
        vanilla_mse=results['vanilla_mse'],
        enhanced_mse=results['enhanced_mse'],
        vanilla_corr=results['vanilla_corr'],
        enhanced_corr=results['enhanced_corr'],
        improvement_ratio=results['improvement_ratio'],
        vanilla_ilc_variances=results['vanilla_ilc_variances'],
        enhanced_ilc_variances=results['enhanced_ilc_variances'],
        primary_cmb=primary_cmb,
        scaled_cmb=scaled_cmb,
        unet_synthetic_channel=unet_synthetic,
        unet_predictions=unet_predictions,
        b_small=b_small,
        test_indices=test_indices,
        frequencies=FREQUENCIES,
        cmb_scale=args.cmb_scale
    )
    print(f"  ✓ Results saved to: {results_file}")
    
    # Save statistics to text file
    stats_file = os.path.join(args.output_dir, 'ilc_comparison_statistics.txt')
    with open(stats_file, 'w') as f:
        # Write header with configuration
        f.write("="*80 + "\n")
        f.write("UNET-ENHANCED ILC EVALUATION STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  Model path: {args.model_path}\n")
        f.write(f"  CMB scaling factor: {args.cmb_scale}\n")
        f.write(f"  Normalization: {'Yes' if args.normalize else 'No'}\n")
        f.write(f"  Input channels: {'B only (1 channel)' if args.b_only else 'B + T + E (3 channels)'}\n")
        f.write(f"  Frequencies: {FREQUENCIES}\n")
        f.write(f"  Test samples: {len(test_indices)}\n")
        f.write("\n")
        # Write statistics
        f.write(results['stats_text'])
        f.write("\n")
    print(f"  ✓ Statistics saved to: {stats_file}")
    
    # Create model identifier from model path
    model_filename = os.path.basename(args.model_path)  # e.g., "best_model_153600_19200_nonorm.pt"
    model_basename = os.path.splitext(model_filename)[0]  # Remove .pt extension
    norm_suffix = "normalized" if args.normalize else "unnormalized"
    channel_suffix = "b_only" if args.b_only else "teb"
    model_identifier = f"{norm_suffix}_{channel_suffix}_{model_basename}"
    
    # Save cross-spectra statistics
    if cross_spectra_results['ell_array'] is not None:
        save_cross_spectra_statistics(cross_spectra_results, args.output_dir, model_identifier)
    
    # Save foreground MSE statistics
    save_fg_mse_statistics(cross_spectra_results, args.output_dir, model_identifier)
    
    # Create visualizations
    if args.sample_indices:
        sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
    else:
        sample_indices = None
    
    visualize_comparison(results, args.output_dir, sample_indices)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - ilc_comparison_results.npz: All numerical results")
    print(f"  - ilc_comparison_statistics.txt: Summary statistics")
    print(f"  - sample_*_comparison.png: Individual sample comparisons")
    print(f"  - mse_comparison_scatter.png: MSE scatter plot")
    print(f"  - correlation_comparison.png: Correlation comparison")
    print(f"  - ilc_weights_comparison.png: ILC weights comparison")
    print(f"  - ilc_variance_comparison.png: ILC CMB reconstruction variance comparison")


if __name__ == '__main__':
    main()

