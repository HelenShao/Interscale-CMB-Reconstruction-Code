#!/usr/bin/env python3
"""
Plot to validate signal preservation: show that cross-power spectrum (reconstructed × true)
equals auto-power spectrum (true × true), demonstrating unbiased CMB recovery.

This validates the statement: "The signal-preserving property of this framework ensures 
that the improved foreground removal does not bias the CMB reconstruction."
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from cmb_spectrum_2d import calculate_2d_spectrum
import paths_config

# Constants
DELTA_ELL = 50
ELL_CUTOFF = 200
PIX_SIZE = 3.4  # arcminutes (for NSIDE=1024)
PATCH_SIZE = 128

# File paths (override with env vars or edit for your layout)
FILE_8CH = os.environ.get(
    'ILC_ML_TEST_8CH_NPZ',
    os.path.join(paths_config.OUTPUT_DIR_INTERSCALE, 'unnormalized_full', 'test_results.npz'),
)
FILE_16CH = os.environ.get(
    'ILC_ML_TEST_16CH_NPZ',
    os.path.join(
        paths_config.OUTPUT_DIR_INTERSCALE,
        'unnormalized_full_et/eval_results/test_results.npz',
    ),
)

OUTPUT_DIR = os.environ.get('ILC_ML_FIGURES_DIR', os.path.join(_SCRIPT_DIR, 'figures'))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'signal_preservation_validation.png')

def compute_power_spectrum(map_data, delta_ell=DELTA_ELL, ell_max=ELL_CUTOFF, 
                           pix_size=PIX_SIZE, N=PATCH_SIZE):
    """
    Compute power spectrum of a 2D map.
    
    Args:
        map_data: 2D numpy array (H, W)
        delta_ell: bin width for power spectra
        ell_max: maximum ell for analysis
        pix_size: pixel size in arcminutes
        N: patch size in pixels
        
    Returns:
        ell_array: array of ell values
        cl: power spectrum
    """
    ell_array, cl = calculate_2d_spectrum(
        map_data, map_data,
        delta_ell=delta_ell, ell_max=ell_max,
        pix_size=pix_size, N=N, lib_mode='NumPy'
    )
    return ell_array, cl

def compute_cross_power_spectrum(map1, map2, delta_ell=DELTA_ELL, ell_max=ELL_CUTOFF,
                                 pix_size=PIX_SIZE, N=PATCH_SIZE):
    """
    Compute cross-power spectrum between two 2D maps.
    
    Args:
        map1, map2: 2D numpy arrays (H, W)
        delta_ell: bin width for power spectra
        ell_max: maximum ell for analysis
        pix_size: pixel size in arcminutes
        N: patch size in pixels
        
    Returns:
        ell_array: array of ell values
        cl_cross: cross-power spectrum
    """
    ell_array, cl_cross = calculate_2d_spectrum(
        map1, map2,
        delta_ell=delta_ell, ell_max=ell_max,
        pix_size=pix_size, N=N, lib_mode='NumPy'
    )
    return ell_array, cl_cross

def process_configuration(filepath, config_name):
    """
    Process one configuration (8-channel or 16-channel) and compute power spectra.
    
    Returns:
        ell_array: multipole values
        mean_cross_ps: mean cross-power spectrum (reconstructed × true) across patches
        std_cross_ps: std of cross-power spectrum across patches
        mean_auto_ps: mean auto-power spectrum (true × true) across patches
        std_auto_ps: std of auto-power spectrum across patches
    """
    print(f"\nProcessing {config_name}...")
    print(f"Loading data from: {filepath}")
    
    data = np.load(filepath)
    unet_cmb_reconstructed = data['unet_cmb_reconstructed']
    cmb_draw = data['cmb_draw']
    
    n_patches = len(unet_cmb_reconstructed)
    print(f"Number of patches: {n_patches}")
    
    # Compute cross-power spectrum (reconstructed × true) for each patch
    cross_spectra = []
    auto_spectra = []
    ell_array = None
    
    for i in range(n_patches):
        if (i + 1) % 50 == 0:
            print(f"  Processing patch {i+1}/{n_patches}...")
        
        # Cross-power spectrum: reconstructed × true
        ell_array, cl_cross = compute_cross_power_spectrum(
            unet_cmb_reconstructed[i], cmb_draw[i]
        )
        cross_spectra.append(cl_cross)
        
        # Auto-power spectrum: true × true
        _, cl_auto = compute_power_spectrum(cmb_draw[i])
        auto_spectra.append(cl_auto)
    
    # Convert to arrays
    cross_spectra = np.array(cross_spectra)  # Shape: (n_patches, n_ell_bins)
    auto_spectra = np.array(auto_spectra)    # Shape: (n_patches, n_ell_bins)
    
    # Compute mean and std across patches
    mean_cross_ps = np.mean(cross_spectra, axis=0)
    std_cross_ps = np.std(cross_spectra, axis=0)
    mean_auto_ps = np.mean(auto_spectra, axis=0)
    std_auto_ps = np.std(auto_spectra, axis=0)
    
    print(f"  Mean cross-power spectrum: {np.mean(mean_cross_ps):.6e}")
    print(f"  Mean auto-power spectrum: {np.mean(mean_auto_ps):.6e}")
    print(f"  Ratio (cross/auto): {np.mean(mean_cross_ps) / np.mean(mean_auto_ps):.6f}")
    
    return ell_array, mean_cross_ps, std_cross_ps, mean_auto_ps, std_auto_ps

def plot_signal_preservation_validation():
    """Create the validation plot showing cross-power = auto-power."""
    
    # Process both configurations
    ell_8ch, mean_cross_8ch, std_cross_8ch, mean_auto_8ch, std_auto_8ch = \
        process_configuration(FILE_8CH, "8-channel")
    
    ell_16ch, mean_cross_16ch, std_cross_16ch, mean_auto_16ch, std_auto_16ch = \
        process_configuration(FILE_16CH, "16-channel")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 8-channel configuration
    ax1.plot(ell_8ch, mean_cross_8ch, 'b-', linewidth=2, label=r'$\langle \hat{S} \times S \rangle$ (reconstructed × true)')
    ax1.plot(ell_8ch, mean_auto_8ch, 'r--', linewidth=2, label=r'$\langle S \times S \rangle$ (true × true)')
    ax1.fill_between(ell_8ch, mean_cross_8ch - std_cross_8ch, mean_cross_8ch + std_cross_8ch, 
                     alpha=0.2, color='b', label='Cross-power ±1σ')
    ax1.fill_between(ell_8ch, mean_auto_8ch - std_auto_8ch, mean_auto_8ch + std_auto_8ch, 
                     alpha=0.2, color='r', label='Auto-power ±1σ')
    ax1.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax1.set_ylabel(r'Power Spectrum $C_\ell$ [($\mu$K)$^2$]', fontsize=12)
    ax1.set_title('8-Channel Configuration', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([ell_8ch.min() - 10, ell_8ch.max() + 10])
    
    # Plot 16-channel configuration
    ax2.plot(ell_16ch, mean_cross_16ch, 'b-', linewidth=2, label=r'$\langle \hat{S} \times S \rangle$ (reconstructed × true)')
    ax2.plot(ell_16ch, mean_auto_16ch, 'r--', linewidth=2, label=r'$\langle S \times S \rangle$ (true × true)')
    ax2.fill_between(ell_16ch, mean_cross_16ch - std_cross_16ch, mean_cross_16ch + std_cross_16ch, 
                     alpha=0.2, color='b', label='Cross-power ±1σ')
    ax2.fill_between(ell_16ch, mean_auto_16ch - std_auto_16ch, mean_auto_16ch + std_auto_16ch, 
                     alpha=0.2, color='r', label='Auto-power ±1σ')
    ax2.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax2.set_ylabel(r'Power Spectrum $C_\ell$ [($\mu$K)$^2$]', fontsize=12)
    ax2.set_title('16-Channel Configuration', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([ell_16ch.min() - 10, ell_16ch.max() + 10])
    
    # Add overall title
    fig.suptitle('Signal Preservation Validation: Cross-Power = Auto-Power', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {OUTPUT_FILE}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\n8-Channel Configuration:")
    print(f"  Mean cross-power / Mean auto-power ratio: {np.mean(mean_cross_8ch) / np.mean(mean_auto_8ch):.6f}")
    print(f"  Per-bin ratios: {mean_cross_8ch / mean_auto_8ch}")
    print(f"\n16-Channel Configuration:")
    print(f"  Mean cross-power / Mean auto-power ratio: {np.mean(mean_cross_16ch) / np.mean(mean_auto_16ch):.6f}")
    print(f"  Per-bin ratios: {mean_cross_16ch / mean_auto_16ch}")

if __name__ == "__main__":
    plot_signal_preservation_validation()

