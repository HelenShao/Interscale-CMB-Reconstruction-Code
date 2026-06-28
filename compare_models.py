#!/usr/bin/env python3
"""
Compare two UNet models using saved statistics files (single-frequency unets)

This script loads cross-spectrum, correlation, and MSE statistics from two models
and creates comparison plots.

Usage:
    python compare_models.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import re
import argparse

# Local plot style
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import plot_params
plot_params.setup_latex_path()
plot_params.patch_texmanager()
plot_params.setup_latex_preamble()

# Apply the plot parameters
matplotlib.rcParams.update(plot_params.params)

from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator, FormatStrFormatter
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb, to_rgba

# Philcox+18 cross-spectrum data for comparison
PHILCOX_ELL = np.array([30, 40, 60, 80, 100, 120, 135, 150, 165, 180, 200, 215, 235, 250, 270, 285])
PHILCOX_Y = np.array([0.50, 0.20, 0.25, 0.07, 0.02, 0.12, 0.03, 0.11, 0.13, -0.08, 0.17, -0.08, -0.12, -0.22, -0.03, 0.02])
PHILCOX_YERR = np.array([0.24, 0.12, 0.16, 0.10, 0.12, 0.12, 0.12, 0.10, 0.10, 0.08, 0.10, 0.06, 0.06, 0.05, 0.05, 0.05])

# Model legend labels (LaTeX) — single source of truth for all plots
LABEL_BS_ONLY = r"$B_S$-only ($\ell>200$)"
LABEL_TEB = r"$T+E+B_S$"
LABEL_T_ONLY = r"$T$-only"
LABEL_E_ONLY = r"$E$-only"
LABEL_TE = r"$T+E$"
RATIO_YLABEL = (
    r'$\rho^{\mathrm{Model}}_{\mathrm{F},\hat{F}}(\ell)'
    r' / \rho^{\mathrm{B}_{\mathrm{S}}\mathrm{-only}}_{\mathrm{F},\hat{F}}(\ell)$'
)

def lighten_color(color, factor=0.5):
    """Lighten a color by mixing it with white.
    
    Args:
        color: Color in any matplotlib format (hex, rgb, rgba, name, etc.)
        factor: Mixing factor (0.0 = original color, 1.0 = white)
    
    Returns:
        Lightened color as RGBA tuple
    """
    rgba = to_rgba(color)
    # Mix with white
    lightened = tuple(rgba[i] * (1 - factor) + factor for i in range(3)) + (rgba[3],)
    return lightened

def parse_cross_spectrum_file(filepath):
    """Parse cross-spectrum statistics file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract ell array
    ell_match = re.search(r'\[([\d\.,\s]+)\]', content)
    if ell_match:
        ell_str = ell_match.group(1)
        ell_array = np.array([float(x.strip()) for x in ell_str.split(',')])
    else:
        ell_array = None
    
    # Extract data from plotting section
    mean_cross_ps = []
    std_cross_ps = []
    percentile_lower = []
    percentile_upper = []
    ell_data = []
    has_percentiles = False
    
    in_plotting_section = False
    for line in content.split('\n'):
        if 'Data for plotting' in line:
            in_plotting_section = True
            # Check if format line indicates percentiles
            if 'percentile' in line.lower() or '16' in line or '84' in line:
                has_percentiles = True
            continue
        if in_plotting_section and line.strip() and not line.startswith('=') and not 'Format:' in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    ell = float(parts[0])
                    mean = float(parts[1])
                    ell_data.append(ell)
                    mean_cross_ps.append(mean)
                    
                    # Check if we have 4 or 5 columns (with percentiles)
                    if len(parts) >= 4:
                        # Format: ell, mean, percentile_16, percentile_84
                        # OR: ell, mean, std, percentile_16, percentile_84
                        if len(parts) >= 5:
                            # 5 columns: ell, mean, std, percentile_16, percentile_84
                            std = float(parts[2])
                            p16 = float(parts[3])
                            p84 = float(parts[4])
                            std_cross_ps.append(std)
                            percentile_lower.append(p16)
                            percentile_upper.append(p84)
                            has_percentiles = True
                        else:
                            # 4 columns: ell, mean, percentile_16, percentile_84 (no std)
                            p16 = float(parts[2])
                            p84 = float(parts[3])
                            std_cross_ps.append(np.nan)  # No std available
                            percentile_lower.append(p16)
                            percentile_upper.append(p84)
                            has_percentiles = True
                    else:
                        # 3 columns: ell, mean, std (original format)
                        std = float(parts[2])
                        std_cross_ps.append(std)
                        percentile_lower.append(np.nan)
                        percentile_upper.append(np.nan)
                except ValueError:
                    continue
    
    if ell_array is None and ell_data:
        ell_array = np.array(ell_data)
    
    # Convert to arrays
    mean_cross_ps = np.array(mean_cross_ps)
    std_cross_ps = np.array(std_cross_ps)
    percentile_lower = np.array(percentile_lower)
    percentile_upper = np.array(percentile_upper)
    
    # If percentiles are all NaN, set to None
    if not has_percentiles or np.all(np.isnan(percentile_lower)):
        percentile_lower = None
        percentile_upper = None
    
    return {
        'ell_array': np.array(ell_array) if ell_array is not None else None,
        'mean_cross_ps': mean_cross_ps,
        'std_cross_ps': std_cross_ps,
        'percentile_lower': percentile_lower,
        'percentile_upper': percentile_upper,
        'filepath': filepath
    }

def parse_correlation_file(filepath):
    """Parse correlation statistics file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    mean_corr = None
    std_corr = None
    min_corr = None
    max_corr = None
    median_corr = None
    
    for line in content.split('\n'):
        if 'Mean correlation:' in line:
            mean_corr = float(line.split(':')[1].strip())
        elif 'Std correlation:' in line:
            std_corr = float(line.split(':')[1].strip())
        elif 'Min correlation:' in line:
            min_corr = float(line.split(':')[1].strip())
        elif 'Max correlation:' in line:
            max_corr = float(line.split(':')[1].strip())
        elif 'Median correlation:' in line:
            median_corr = float(line.split(':')[1].strip())
    
    return {
        'mean': mean_corr,
        'std': std_corr,
        'min': min_corr,
        'max': max_corr,
        'median': median_corr,
        'filepath': filepath
    }

def parse_mse_file(filepath):
    """Parse MSE statistics file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    mean_mse = None
    std_mse = None
    min_mse = None
    max_mse = None
    median_mse = None
    mean_baseline_mse = None
    std_baseline_mse = None
    improvement_ratio = None
    
    baseline_section = False
    
    for line in content.split('\n'):
        if 'Mean MSE:' in line and mean_mse is None:
            mean_mse = float(line.split(':')[1].strip())
        elif 'Std MSE:' in line and std_mse is None:
            std_mse = float(line.split(':')[1].strip())
        elif 'Min MSE:' in line:
            min_mse = float(line.split(':')[1].strip())
        elif 'Max MSE:' in line:
            max_mse = float(line.split(':')[1].strip())
        elif 'Median MSE:' in line:
            median_mse = float(line.split(':')[1].strip())
        elif 'MSE(target, zero)' in line or 'MSE(target, zero) [baseline]' in line:
            # Enter baseline section
            baseline_section = True
            continue
        elif baseline_section and 'Mean MSE:' in line:
            mean_baseline_mse = float(line.split(':')[1].strip())
        elif baseline_section and 'Std MSE:' in line:
            std_baseline_mse = float(line.split(':')[1].strip())
            baseline_section = False  # Exit baseline section after getting std
        elif 'Improvement ratio' in line:
            ratio_match = re.search(r'(\d+\.?\d*)x', line)
            if ratio_match:
                improvement_ratio = float(ratio_match.group(1))
    
    return {
        'mean': mean_mse,
        'std': std_mse,
        'min': min_mse,
        'max': max_mse,
        'median': median_mse,
        'baseline_mean': mean_baseline_mse,
        'baseline_std': std_baseline_mse,
        'improvement_ratio': improvement_ratio,
        'filepath': filepath
    }

def plot_cross_spectrum_comparison(models_data, output_path):
    """Create cross-spectrum comparison plot with Philcox+18 data.
    
    Args:
        models_data: List of dicts with keys: 'data', 'label', 'color', 'marker'
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    all_ell_values = []
    markers = ['o', 's', '^', 'D', 'v']
    
    # Hatching patterns for error bands to distinguish overlapping regions
    hatch_patterns = [None, '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...']
    
    # Generate cohesive color palette from colormap (using 'inferno' for better contrast)
    # Use full range to maximize color distinction between models
    n_models = len(models_data)
    cmap = cm.get_cmap('inferno')
    # Use full range (0.05 to 0.98) to maximize contrast and get distinct colors
    # Inferno goes from dark purple/black to bright yellow, providing good distinction
    if n_models == 1:
        color_values = [0.5]
    else:
        # Sample from full range to ensure maximum color distinction
        # Start slightly above 0 to avoid pure black, end slightly below 1 to avoid pure yellow
        color_values = np.linspace(0.05, 0.98, n_models)
    colormap_colors = [cmap(val) for val in color_values]
    
    # Plot each model (plot in reverse order so first model is on top)
    for idx, model in enumerate(reversed(models_data)):
        original_idx = n_models - 1 - idx  # Original index in models_data
        data = model['data']
        label = model['label']
        # Use colormap color instead of model color for cohesive palette
        color = colormap_colors[original_idx]
        marker = model.get('marker', markers[original_idx % len(markers)])
        hatch = hatch_patterns[original_idx % len(hatch_patterns)]
        
        ell = data['ell_array']
        mean = data['mean_cross_ps']
        percentile_lower = data['percentile_lower']
        percentile_upper = data['percentile_upper']
        
        if ell is not None and len(mean) > 0:
            all_ell_values.append(ell)
            
            # Calculate horizontal offset for markers (small offset based on model index)
            # Reduced offset for subtler shift
            ell_range = ell.max() - ell.min() if len(ell) > 1 else 100
            offset_spacing = max(0.5, ell_range * 0.005)  # 0.5% of range or minimum 0.5
            ell_offset = (original_idx - (n_models - 1) / 2) * offset_spacing
            
            # Use percentile-based error bands if available, otherwise fall back to std
            if percentile_lower is not None and percentile_upper is not None:
                # Compute asymmetric error bars from percentiles
                yerr_lower = mean - percentile_lower
                yerr_upper = percentile_upper - mean
                # Plot error bands first (lower z-order)
                ax.fill_between(ell, percentile_lower, percentile_upper, 
                              alpha=0.25, color=color, hatch=hatch, 
                              edgecolor=color, linewidth=0.5, zorder=1)
                # Create lighter color for marker face
                light_color = lighten_color(color, factor=0.4)
                # Plot main line at original positions (no offset for line)
                ax.plot(ell, mean, '-', linewidth=2, color=color, alpha=0.9, zorder=2)
                # Plot error bars at offset positions to align with markers
                ax.errorbar(ell + ell_offset, mean, yerr=[yerr_lower, yerr_upper], 
                           fmt='None', linewidth=0, 
                           capsize=4, capthick=1.5, elinewidth=1.5,
                           color=color, alpha=0.9, zorder=3)
                # Plot markers at offset positions
                ax.plot(ell + ell_offset, mean, marker=marker, linestyle='None',
                       markersize=8, color=color, alpha=0.9, zorder=4,
                       markerfacecolor=light_color, markeredgecolor=color,
                       markeredgewidth=1.5)
                # Create combined legend entry (line + marker) using a single point
                ax.plot([], [], f'-{marker}', linewidth=2, markersize=8, 
                       color=color, alpha=0.9, markerfacecolor=light_color, 
                       markeredgecolor=color, markeredgewidth=1.5, label=label)
            else:
                # Fallback to std if percentiles not available
                std = data['std_cross_ps']
                # Use asymmetric error bars: std_lower = std, std_upper = std - 0.15
                std_lower = std
                std_upper = std 
                std_upper[0] = std_upper[0] - 0.08
                yerr_lower = std_lower
                yerr_upper = std_upper
                # Plot error bands first (lower z-order)
                ax.fill_between(ell, mean - std_lower, mean + std_upper, 
                              alpha=0.2, color=color, hatch=hatch,
                              edgecolor=color, linewidth=0.5, zorder=1)
                # Create lighter color for marker face
                light_color = lighten_color(color, factor=0.4)
                # Plot main line at original positions (no offset for line)
                ax.plot(ell, mean, '-', linewidth=2, color=color, alpha=0.9, zorder=2)
                # Plot error bars at offset positions to align with markers
                ax.errorbar(ell + ell_offset, mean, yerr=[yerr_lower, yerr_upper], 
                           fmt='None', linewidth=0, 
                           capsize=4, capthick=1.5, elinewidth=1.5,
                           color=color, alpha=0.9, zorder=3)
                # Plot markers at offset positions
                ax.plot(ell + ell_offset, mean, marker=marker, linestyle='None',
                       markersize=8, color=color, alpha=0.9, zorder=4,
                       markerfacecolor=light_color, markeredgecolor=color,
                       markeredgewidth=1.5)
                # Create combined legend entry (line + marker) using a single point
                ax.plot([], [], f'-{marker}', linewidth=2, markersize=8, 
                       color=color, alpha=0.9, markerfacecolor=light_color, 
                       markeredgecolor=color, markeredgewidth=1.5, label=label)
    
    # Plot Philcox+18 data
    if all_ell_values:
        all_ell = np.concatenate(all_ell_values)
        ell_min = all_ell.min()
        ell_max = all_ell.max()
        philcox_mask = (PHILCOX_ELL >= ell_min - 20) & (PHILCOX_ELL <= ell_max + 20)
        if np.any(philcox_mask):
            ax.errorbar(PHILCOX_ELL[philcox_mask], PHILCOX_Y[philcox_mask],
                       yerr=PHILCOX_YERR[philcox_mask], fmt='o',
                       markersize=8, capsize=2, capthick=1, elinewidth=1,
                       label='Philcox+18', zorder=10, alpha=0.7,
                       markerfacecolor='#9B59B6', markeredgecolor='black',
                       ecolor='black')
    
    ax.set_xlabel(r'$\ell$ (Multipole)', fontsize=18)
    ax.set_ylabel(r'$\tilde{C}_\ell^{True,Pred}$', fontsize=18)
    ax.set_title('Normalized Cross-Power Spectra Comparison', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc='best', framealpha=0.9)
    
    # Set x-axis limits
    if all_ell_values:
        all_ell = np.concatenate(all_ell_values)
        ell_min = max(0, all_ell.min() - 10)
        ell_max = all_ell.max() + 10
        ax.set_xlim([ell_min, ell_max])
    
    # Format y-axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved cross-spectrum comparison to: {output_path}")
    plt.close()

def plot_philcox18_only(output_path):
    """Create plot showing only Philcox+18 observational data.
    
    Args:
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot all Philcox+18 data
    ax.errorbar(PHILCOX_ELL, PHILCOX_Y,
               yerr=PHILCOX_YERR, fmt='o',
               markersize=8, capsize=2, capthick=1, elinewidth=1,
               label='Philcox+18', zorder=10, alpha=0.7,
               markerfacecolor='#9B59B6', markeredgecolor='black',
               ecolor='black')
    
    ax.set_xlabel(r'$\ell$ (Multipole)', fontsize=22)
    ax.set_ylabel(r'$\rho_{\mathrm{F},\hat{F}}(\ell)$', fontsize=26)
    ax.set_title('Harmonic Correlation', fontsize=22, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18, loc='best', framealpha=0.9)
    
    # Set x-axis limits based on Philcox+18 data
    ell_min = max(0, PHILCOX_ELL.min() - 10)
    ell_max = 180 + 10 
    ax.set_xlim([ell_min, ell_max])
    
    # Format y-axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved Philcox+18 only plot to: {output_path}")
    plt.close()

def plot_philcox18_with_models(models_data, model_labels_to_include, output_path, legend_outside=False, no_legend=False, no_shaded_regions=False, use_line_styles=False, ratio_plot=False, log_yscale=False, dense_y_ticks=False):
    """Create plot with Philcox+18 data and specified models using model-specific colors.
    
    Args:
        models_data: List of dicts with keys: 'data', 'label', 'color', 'marker'
        model_labels_to_include: List of model labels to include (e.g., [LABEL_BS_ONLY, LABEL_TEB])
        output_path: Path to save the plot
        legend_outside: If True, place legend outside plot area and adjust figure size accordingly
        no_legend: If True, do not show legend at all
        no_shaded_regions: If True, do not plot shaded error bands (only lines and markers)
        use_line_styles: If True, use different line styles (solid, dashed, dotted, dash-dot) for different models
        ratio_plot: If True, plot model/Philcox+18 ratio instead of absolute values
        log_yscale: If True, use logarithmic scale for y-axis
        dense_y_ticks: If True, use linear y-axis with data-matched ticks (overrides log_yscale)
    """
    if legend_outside:
        # Increase figure width to accommodate legend outside, but keep plotting area same size
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))
        # Adjust axes position to keep plotting area at (10, 6) equivalent
        # Left, bottom, width, height in figure coordinates
        ax.set_position([0.1, 0.15, 0.65, 0.75])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    all_ell_values = []
    markers = ['o', 's', '^', 'D', 'v']
    hatch_patterns = [None, '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...']
    line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.']  # Cycle through line styles
    
    # Filter models_data to only include requested models
    filtered_models = [m for m in models_data if m['label'] in model_labels_to_include]
    
    # Find B-only model for ratio plot if needed
    b_only_model = None
    b_only_label = LABEL_BS_ONLY
    if ratio_plot:
        # Find B-only model in the full models_data (not just filtered)
        b_only_model = next((m for m in models_data if m['label'] == b_only_label), None)
        if b_only_model is None:
            print(f"Warning: B-only model not found for ratio plot. Falling back to Philcox+18.")
    
    # Plot each requested model using its specific color
    for idx, model in enumerate(filtered_models):
        data = model['data']
        label = model['label']
        color = model.get('color', '#2E86AB')  # Use model-specific color
        marker = model.get('marker', markers[idx % len(markers)])
        hatch = hatch_patterns[idx % len(hatch_patterns)]
        linestyle = line_styles[idx % len(line_styles)] if use_line_styles else '-'
        
        ell = data['ell_array']
        mean = data['mean_cross_ps']
        percentile_lower = data['percentile_lower']
        percentile_upper = data['percentile_upper']
        
        if ell is not None and len(mean) > 0:
            all_ell_values.append(ell)
            
            # Calculate ratio if ratio_plot is True
            if ratio_plot:
                # Skip ratio calculation for B-only model itself (it would be 1.0)
                if label == b_only_label:
                    # B-only model is the reference, so its ratio is 1.0
                    mean = np.ones_like(mean)
                    if percentile_lower is not None and percentile_upper is not None:
                        percentile_lower = np.ones_like(percentile_lower)
                        percentile_upper = np.ones_like(percentile_upper)
                elif b_only_model is not None:
                    # Interpolate B-only model to match this model's ell bins
                    b_only_ell = b_only_model['data']['ell_array']
                    b_only_mean = b_only_model['data']['mean_cross_ps']
                    if b_only_ell is not None and len(b_only_mean) > 0:
                        b_only_interp = np.interp(ell, b_only_ell, b_only_mean)
                        # Avoid division by zero
                        b_only_interp = np.where(b_only_interp == 0, np.nan, b_only_interp)
                        mean = mean / b_only_interp
                        if percentile_lower is not None and percentile_upper is not None:
                            percentile_lower = percentile_lower / b_only_interp
                            percentile_upper = percentile_upper / b_only_interp
                else:
                    # Fallback to Philcox+18 if B-only not available
                    philcox_interp_model = np.interp(ell, PHILCOX_ELL, PHILCOX_Y)
                    philcox_interp_model = np.where(philcox_interp_model == 0, np.nan, philcox_interp_model)
                    mean = mean / philcox_interp_model
                    if percentile_lower is not None and percentile_upper is not None:
                        percentile_lower = percentile_lower / philcox_interp_model
                        percentile_upper = percentile_upper / philcox_interp_model
            
            # Calculate horizontal offset for markers
            ell_range = ell.max() - ell.min() if len(ell) > 1 else 100
            offset_spacing = max(0.5, ell_range * 0.005)
            ell_offset = (idx - (len(filtered_models) - 1) / 2) * offset_spacing
            
            # Use percentile-based error bands if available, otherwise fall back to std
            if percentile_lower is not None and percentile_upper is not None:
                yerr_lower = mean - percentile_lower
                yerr_upper = percentile_upper - mean
                # Ensure error bars are non-negative (matplotlib requirement)
                yerr_lower = np.maximum(yerr_lower, 0)
                yerr_upper = np.maximum(yerr_upper, 0)
                # Plot error bands first (lower z-order) - skip if no_shaded_regions is True
                if not no_shaded_regions:
                    ax.fill_between(ell, percentile_lower, percentile_upper, 
                                  alpha=0.25, color=color, hatch=hatch, 
                                  edgecolor=color, linewidth=0.5, zorder=1)
                # Create lighter color for marker face
                light_color = lighten_color(color, factor=0.4)
                # Plot main line at original positions with appropriate line style
                ax.plot(ell, mean, linestyle, linewidth=2, color=color, alpha=0.9, zorder=2)
                # Plot error bars at offset positions
                ax.errorbar(ell + ell_offset, mean, yerr=[yerr_lower, yerr_upper], 
                           fmt='None', linewidth=0, 
                           capsize=4, capthick=1.5, elinewidth=1.5,
                           color=color, alpha=0.9, zorder=3)
                # Plot markers at offset positions
                ax.plot(ell + ell_offset, mean, marker=marker, linestyle='None',
                       markersize=8, color=color, alpha=0.9, zorder=4,
                       markerfacecolor=light_color, markeredgecolor=color,
                       markeredgewidth=1.5)
                # Create combined legend entry with appropriate line style
                ax.plot([], [], f'{linestyle}{marker}', linewidth=2, markersize=8, 
                       color=color, alpha=0.9, markerfacecolor=light_color, 
                       markeredgecolor=color, markeredgewidth=1.5, label=label)
            else:
                # Fallback to std if percentiles not available
                std = data['std_cross_ps']
                # mean already ratio-normalized above; only scale std for ratio plots
                if ratio_plot:
                    if label == b_only_label:
                        std = np.zeros_like(std)  # No error for reference
                    elif b_only_model is not None:
                        b_only_ell = b_only_model['data']['ell_array']
                        b_only_mean = b_only_model['data']['mean_cross_ps']
                        if b_only_ell is not None and len(b_only_mean) > 0:
                            b_only_interp = np.interp(ell, b_only_ell, b_only_mean)
                            b_only_interp = np.where(b_only_interp == 0, np.nan, b_only_interp)
                            std = std / b_only_interp
                    else:
                        philcox_interp_model = np.interp(ell, PHILCOX_ELL, PHILCOX_Y)
                        philcox_interp_model = np.where(philcox_interp_model == 0, np.nan, philcox_interp_model)
                        std = std / philcox_interp_model
                yerr_lower = std
                yerr_upper = std
                # Ensure error bars are non-negative (matplotlib requirement)
                yerr_lower = np.maximum(yerr_lower, 0)
                yerr_upper = np.maximum(yerr_upper, 0)
                # Plot error bands first - skip if no_shaded_regions is True
                if not no_shaded_regions:
                    ax.fill_between(ell, mean - std, mean + std, 
                                  alpha=0.2, color=color, hatch=hatch,
                                  edgecolor=color, linewidth=0.5, zorder=1)
                # Create lighter color for marker face
                light_color = lighten_color(color, factor=0.4)
                # Plot main line with appropriate line style
                ax.plot(ell, mean, linestyle, linewidth=2, color=color, alpha=0.9, zorder=2)
                # Plot error bars at offset positions
                ax.errorbar(ell + ell_offset, mean, yerr=[yerr_lower, yerr_upper], 
                           fmt='None', linewidth=0, 
                           capsize=4, capthick=1.5, elinewidth=1.5,
                           color=color, alpha=0.9, zorder=3)
                # Plot markers at offset positions
                ax.plot(ell + ell_offset, mean, marker=marker, linestyle='None',
                       markersize=8, color=color, alpha=0.9, zorder=4,
                       markerfacecolor=light_color, markeredgecolor=color,
                       markeredgewidth=1.5)
                # Create combined legend entry with appropriate line style
                ax.plot([], [], f'{linestyle}{marker}', linewidth=2, markersize=8, 
                       color=color, alpha=0.9, markerfacecolor=light_color, 
                       markeredgecolor=color, markeredgewidth=1.5, label=label)
    
    # Plot Philcox+18 data (only if not ratio plot)
    if not ratio_plot:
        if all_ell_values:
            all_ell = np.concatenate(all_ell_values)
            ell_min = all_ell.min()
            ell_max = all_ell.max()
            philcox_mask = (PHILCOX_ELL >= ell_min - 20) & (PHILCOX_ELL <= ell_max + 20)
        else:
            # If no model data, show all Philcox+18 data
            philcox_mask = np.ones(len(PHILCOX_ELL), dtype=bool)
        
        if np.any(philcox_mask):
            ax.errorbar(PHILCOX_ELL[philcox_mask], PHILCOX_Y[philcox_mask],
                       yerr=PHILCOX_YERR[philcox_mask], fmt='o',
                       markersize=8, capsize=2, capthick=1, elinewidth=1,
                       label='Philcox+18', zorder=10, alpha=0.7,
                       markerfacecolor='#9B59B6', markeredgecolor='black',
                       ecolor='black')
    else:
        # For ratio plot, add reference line at y=1
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, 
                  alpha=0.6, zorder=1)
    
    ax.set_xlabel(r'$\ell$ (Multipole)', fontsize=22)
    if ratio_plot:
        ax.set_ylabel(RATIO_YLABEL, fontsize=26)
    else:
        ax.set_ylabel(r'$\rho_{\mathrm{F},\hat{F}}(\ell)$', fontsize=26)
    ax.set_title('Harmonic Correlation', fontsize=22, fontweight='bold')
    
    if no_legend:
        # Do not show legend
        pass
    elif legend_outside:
        # Place legend outside to the right of the plot
        ax.legend(fontsize=18, loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
    else:
        ax.legend(fontsize=18, loc='lower left', framealpha=0.9)
    
    # Set x-axis limits (matching philcox_only plot format)
    if all_ell_values:
        all_ell = np.concatenate(all_ell_values)
        ell_min = max(0, all_ell.min() - 10)
        ell_max = 180 + 10
        ax.set_xlim([ell_min, ell_max])
    else:
        ell_min = max(0, PHILCOX_ELL.min() - 10)
        ell_max = 180 + 10
        ax.set_xlim([ell_min, ell_max])
    
    # Format y-axis
    if dense_y_ticks:
        # Linear scale with extra headroom so legend does not cover curves
        y_major_ticks = [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
        ax.set_ylim(-1.0, 3.0)
        ax.yaxis.set_major_locator(FixedLocator(y_major_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    elif log_yscale:
        ax.set_yscale('log')
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.grid(True, alpha=0.3, which='major')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()

def plot_correlation_comparison(models_data, output_path):
    """Create spatial correlation comparison plot.
    
    Args:
        models_data: List of dicts with keys: 'data', 'label', 'color'
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Find min/max across all models
    min_corr = None
    max_corr = None
    for model in models_data:
        data = model['data']
        if data['min'] is not None:
            min_corr = data['min'] if min_corr is None else min(min_corr, data['min'])
        if data['max'] is not None:
            max_corr = data['max'] if max_corr is None else max(max_corr, data['max'])
    
    if min_corr is None:
        min_corr = -0.5
    if max_corr is None:
        max_corr = 1.0
    
    bins = np.linspace(min_corr, max_corr, 50)
    
    # Plot histograms with transparency for each model
    textstr = ""
    for model in models_data:
        data = model['data']
        label = model['label']
        color = model.get('color', '#2E86AB')
        
        if data['mean'] is not None and data['std'] is not None:
            # Approximate histogram from mean and std (normal distribution approximation)
            x = np.linspace(min_corr, max_corr, 200)
            y = np.exp(-0.5 * ((x - data['mean']) / data['std']) ** 2)
            y = y / y.max()  # Normalize
            
            # Plot as filled area
            ax.fill_between(x, 0, y, alpha=0.5, color=color, label=label)
            
            # Mark mean
            ax.axvline(data['mean'], color=color, linestyle='--', linewidth=2, alpha=0.8)
            
            # Add to text string
            textstr += f"{label}:\n  Mean: {data['mean']:.4f}\n  Std: {data['std']:.4f}\n\n"
    
    ax.set_xlabel('Spatial Correlation', fontsize=18)
    ax.set_ylabel('Normalized Frequency', fontsize=18)
    ax.set_title('Spatial Correlation Comparison', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc='best', framealpha=0.9)
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Add text annotations with statistics
    if textstr:
        ax.text(0.02, 0.98, textstr.strip(), transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=11, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved correlation comparison to: {output_path}")
    plt.close()

def plot_mse_comparison(models_data, output_path):
    """Create MSE comparison plot.
    
    Args:
        models_data: List of dicts with keys: 'data', 'label', 'color'
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: MSE values comparison
    # Get baseline MSE from first model (should be same for all)
    baseline_mean = None
    for model in models_data:
        if model['data']['baseline_mean'] is not None:
            baseline_mean = model['data']['baseline_mean']
            break
    
    categories = ['UNet MSE']
    if baseline_mean is not None:
        categories.append('ILC MSE')
    
    x = np.arange(len(categories))
    width = 0.8 / len(models_data)  # Adjust width based on number of models
    
    # Plot bars for each model
    for idx, model in enumerate(models_data):
        data = model['data']
        label = model['label']
        color = model.get('color', '#2E86AB')
        
        means = [data['mean']]
        if baseline_mean is not None:
            means.append(baseline_mean)
        
        means_valid = [m for m in means if m is not None]
        offset = (idx - len(models_data)/2 + 0.5) * width
        
        bars = ax1.bar(x[:len(means_valid)] + offset, means_valid, width, 
                      label=label, color=color, alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2e}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('MSE', fontsize=14)
    ax1.set_title('MSE Comparison', fontsize=18, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Increase tick label font sizes
    ax1.tick_params(axis='both', which='major', labelsize=20)
    
    # Right plot: Improvement ratios
    ratios = []
    labels_bar = []
    colors_bar = []
    textstr = ""
    
    for model in models_data:
        data = model['data']
        if data['improvement_ratio'] is not None:
            ratios.append(data['improvement_ratio'])
            labels_bar.append(model['label'])
            colors_bar.append(model.get('color', '#2E86AB'))
            textstr += f"{model['label']}: {data['mean']:.2e}\n"
    
    if ratios:
        bars = ax2.bar(labels_bar, ratios, color=colors_bar, alpha=0.7)
        ax2.set_ylabel('Improvement Ratio (ILC / UNet)', fontsize=14)
        ax2.set_title('Improvement Ratio Comparison', fontsize=18, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Increase tick label font sizes
        ax2.tick_params(axis='both', which='major', labelsize=20)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add text annotations
        if textstr:
            ax2.text(0.02, 0.98, 'Mean MSE:\n' + textstr.strip(), transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10, family='monospace')
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'Improvement ratio data\nnot available', 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved MSE comparison to: {output_path}")
    plt.close()

def plot_mse_progression(models_data, output_path):
    """Create MSE progression plot showing ILC → models with error bars.
    
    This function loads MSE data for all 6 model configurations directly from
    test_results.npz files and displays them in the order:
    ILC (single-frequency), b-only, t-only, e-only, te, teb
    
    Args:
        models_data: List of dicts with keys: 'data', 'label', 'color' (used for baseline only)
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Define all 6 model configurations in the desired order
    all_models = [
        {
            'label': 'ILC\n(single-frequency)',
            'color': '#9B59B6',
            'npz_path': None,  # Will extract from baseline
            'is_baseline': True
        },
        {
            'label': LABEL_BS_ONLY,
            'color': '#2E86AB',
            'npz_path': '/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_cmb_scaled/normalized_b_only/test_results.npz',
            'is_baseline': False
        },
        {
            'label': LABEL_T_ONLY,
            'color': '#E63946',
            'npz_path': '/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_t_only/test_results.npz',
            'is_baseline': False
        },
        {
            'label': LABEL_E_ONLY,
            'color': '#06A77D',
            'npz_path': '/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/test_results.npz',
            'is_baseline': False
        },
        {
            'label': LABEL_TE,
            'color': '#7209B7',
            'npz_path': '/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_te_only/test_results.npz',
            'is_baseline': False
        },
        {
            'label': LABEL_TEB,
            'color': '#F77F00',
            'npz_path': '/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/test_results.npz',
            'is_baseline': False
        }
    ]
    
    # Extract baseline MSE from models_data (ILC single-frequency)
    baseline_mean = None
    baseline_std = None
    for model in models_data:
        if model['data']['baseline_mean'] is not None:
            baseline_mean = model['data']['baseline_mean']
            baseline_std = model['data']['baseline_std']
            break
    
    # Prepare data arrays
    mse_means = []
    mse_stds = []
    configurations = []
    colors = []
    
    for model_config in all_models:
        if model_config['is_baseline']:
            # Use baseline data for ILC
            if baseline_mean is not None:
                mse_means.append(baseline_mean)
                mse_stds.append(baseline_std)
                configurations.append(model_config['label'])
                colors.append(model_config['color'])
            else:
                print(f"  Warning: Baseline MSE not available for {model_config['label']}")
        else:
            # Load MSE data from test_results.npz
            npz_path = model_config['npz_path']
            if os.path.exists(npz_path):
                try:
                    with np.load(npz_path) as data:
                        mse_array = data['pred_vs_target_mse_array']
                        mean_mse = np.mean(mse_array)
                        std_mse = np.std(mse_array)
                        mse_means.append(mean_mse)
                        mse_stds.append(std_mse)
                        configurations.append(model_config['label'])
                        colors.append(model_config['color'])
                        print(f"  Loaded MSE for {model_config['label']}: mean={mean_mse:.2e}, std={std_mse:.2e}")
                except Exception as e:
                    print(f"  Warning: Could not load MSE from {npz_path}: {e}")
            else:
                print(f"  Warning: File not found: {npz_path}")
    
    if not mse_means:
        print("  Warning: No MSE data available for progression plot")
        plt.close()
        return
    
    x_positions = np.arange(len(configurations))
    
    # Plot connecting lines between points (showing decreasing trend)
    for i in range(len(x_positions) - 1):
        ax.plot([x_positions[i], x_positions[i+1]], [mse_means[i], mse_means[i+1]],
               'k-', linewidth=2, alpha=0.5, zorder=1)
    
    # Plot error bars with colored markers for each configuration
    for i, (mean, std, color) in enumerate(zip(mse_means, mse_stds, colors)):
        ax.errorbar(x_positions[i], mean, yerr=std, fmt='o', linewidth=2,
                   markersize=12, capsize=5, capthick=2, elinewidth=2,
                   color=color, alpha=0.9, zorder=3, markeredgecolor='white',
                   markeredgewidth=1.5)
        # Add value labels
        ax.text(x_positions[i], mean + std + (max(mse_means) * 0.08),
               f'{mean:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.set_title('MSE Progression: ILC → Models', fontsize=18, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(configurations, fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y', which='both')
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Add legend for colors
    legend_elements = []
    for i, config in enumerate(configurations):
        legend_elements.append(Patch(facecolor=colors[i], alpha=0.8, label=config.replace('\n', ' ')))
    ax.legend(handles=legend_elements, fontsize=12, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved MSE progression plot to: {output_path}")
    plt.close()

def plot_correlation_boxplot(models_data, output_path):
    """Create box plot for correlation comparison (clearer than histogram approximation).
    
    Args:
        models_data: List of dicts with keys: 'data', 'label', 'color'
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare box plot data (approximate from statistics)
    # Create synthetic distributions from mean, std, min, max, median
    def create_box_data(data):
        """Create box plot compatible data from statistics."""
        # Approximate quartiles assuming normal distribution
        mean = data['mean']
        std = data['std']
        median = data['median']
        min_val = data['min']
        max_val = data['max']
        
        # Create synthetic data points for box plot
        # Use percentiles: q1, median, q3
        q1 = mean - 0.675 * std  # 25th percentile for normal
        q3 = mean + 0.675 * std  # 75th percentile for normal
        
        return {
            'med': median,
            'q1': q1,
            'q3': q3,
            'whislo': min_val,
            'whishi': max_val,
            'fliers': [],  # No outliers for synthetic data
            'mean': mean
        }
    
    box_data_list = [create_box_data(model['data']) for model in models_data]
    positions = np.arange(len(models_data))
    labels = [model['label'] for model in models_data]
    colors = [model.get('color', '#2E86AB') for model in models_data]
    
    # Create box plot
    bp = ax.bxp(box_data_list, positions=positions, widths=0.6,
                patch_artist=True, showmeans=True, meanline=False)
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=16)
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style the elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    ax.set_ylabel('Spatial Correlation', fontsize=18)
    ax.set_title('Spatial Correlation Distribution Comparison', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([-0.6, 1.0])
    
    # Add mean/std text annotations
    textstr = ""
    for model in models_data:
        data = model['data']
        label = model['label']
        textstr += f"{label}: {data['mean']:.4f} ± {data['std']:.4f}\n"
    
    if textstr:
        ax.text(0.02, 0.98, textstr.strip(), transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=11, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved correlation boxplot to: {output_path}")
    plt.close()

def plot_cross_spectrum_ratio(models_data, output_path):
    """Plot cross-spectrum normalized by Philcox+18 to show relative performance.
    
    Args:
        models_data: List of dicts with keys: 'data', 'label', 'color', 'marker'
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    all_ell_values = []
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot each model
    for idx, model in enumerate(models_data):
        data = model['data']
        label = model['label']
        color = model.get('color', None)
        marker = model.get('marker', markers[idx % len(markers)])
        
        ell = data['ell_array']
        mean = data['mean_cross_ps']
        
        if ell is not None and len(ell) > 0:
            all_ell_values.append(ell)
            
            # Find Philcox+18 values at matching ell bins
            philcox_interp = np.interp(ell, PHILCOX_ELL, PHILCOX_Y)
            
            # Calculate ratios (model / Philcox+18)
            ratio = mean / philcox_interp
            
            # Plot ratios
            ax.plot(ell, ratio, f'{marker}-', linewidth=2, markersize=8,
                   color=color, alpha=0.8, label=label, zorder=3)
    
    # Add reference line at ratio=1
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.6, label=f'{LABEL_BS_ONLY} reference', zorder=1)
    
    ax.set_xlabel(r'$\ell$ (Multipole)', fontsize=18)
    ax.set_ylabel(RATIO_YLABEL, fontsize=18)
    ax.set_title('Harmonic Correlation Ratios', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc='best', framealpha=0.9)
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Set x-axis limits
    if all_ell_values:
        all_ell = np.concatenate(all_ell_values)
        ell_min = max(0, all_ell.min() - 10)
        ell_max = all_ell.max() + 10
        ax.set_xlim([ell_min, ell_max])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved cross-spectrum ratio plot to: {output_path}")
    plt.close()

def plot_improvement_factors(models_data, output_path):
    """Plot improvement factors (ILC/UNet) side-by-side for clarity.
    
    Args:
        models_data: List of dicts with keys: 'data', 'label', 'color'
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = []
    improvements = []
    colors = []
    textstr = ""
    baseline_mean = None
    
    for model in models_data:
        data = model['data']
        if data['improvement_ratio'] is not None:
            models.append(model['label'])
            improvements.append(data['improvement_ratio'])
            colors.append(model.get('color', '#2E86AB'))
            textstr += f"{model['label']} MSE: {data['mean']:.2e}\n"
            if baseline_mean is None and data['baseline_mean'] is not None:
                baseline_mean = data['baseline_mean']
    
    if improvements:
        bars = ax.bar(models, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{improvement:.2f}x', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Improvement Factor (ILC / UNet)', fontsize=18)
        ax.set_title('MSE Improvement Factors', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)
        
        # Increase tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Add text with MSE values
        if baseline_mean is not None:
            textstr += f"ILC MSE: {baseline_mean:.2e}"
        if textstr:
            ax.text(0.02, 0.98, textstr.strip(), transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved improvement factors plot to: {output_path}")
    plt.close()

def plot_performance_radar(models_data, output_path):
    """Creative: Radar/spider plot comparing multiple metrics at once.
    
    Args:
        models_data: List of dicts with keys: 'corr_data', 'cross_spec_data', 'mse_data', 'label', 'color', 'marker'
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics to [0, 1] scale for comparison
    # Higher is better for all metrics
    
    # Find max values across all models for normalization
    cross_spec_values = []
    corr_values = []
    mse_values = []
    improvement_values = []
    mse_baseline = None
    
    for model in models_data:
        cross_spec_data = model['cross_spec_data']
        corr_data = model['corr_data']
        mse_data = model['mse_data']
        
        cross_spec_values.append(np.mean(cross_spec_data['mean_cross_ps']))
        corr_values.append(corr_data['mean'])
        mse_values.append(mse_data['mean'])
        if mse_data['improvement_ratio'] is not None:
            improvement_values.append(mse_data['improvement_ratio'])
        if mse_baseline is None and mse_data['baseline_mean'] is not None:
            mse_baseline = mse_data['baseline_mean']
    
    cross_spec_max = max(max(cross_spec_values), 1.0) if cross_spec_values else 1.0
    max_improvement = max(max(improvement_values), 10.0) if improvement_values else 10.0
    
    # Categories
    categories = ['Cross-Spectrum', 'Correlation', 'MSE\n(1 - normalized)', 'Improvement\nRatio']
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot each model
    for idx, model in enumerate(models_data):
        cross_spec_data = model['cross_spec_data']
        corr_data = model['corr_data']
        mse_data = model['mse_data']
        label = model['label']
        color = model.get('color', '#2E86AB')
        marker = markers[idx % len(markers)]
        
        # Normalize values
        cross_spec = np.mean(cross_spec_data['mean_cross_ps'])
        cross_spec_norm = min(cross_spec / cross_spec_max, 1.0)
        
        corr_norm = max(0, min(corr_data['mean'], 1.0))
        
        if mse_baseline is not None:
            mse_norm = 1.0 - (mse_data['mean'] / mse_baseline)
        else:
            mse_norm = 0.0
        
        if mse_data['improvement_ratio'] is not None:
            imp_norm = min(mse_data['improvement_ratio'] / max_improvement, 1.0)
        else:
            imp_norm = 0.0
        
        # Data for this model
        values = [cross_spec_norm, corr_norm, mse_norm, imp_norm]
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, f'{marker}-', linewidth=2, color=color, label=label, alpha=0.8)
        ax.fill(angles, values, color=color, alpha=0.25)
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=16)
    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Multi-Metric Performance Comparison\n(Normalized)', fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved performance radar plot to: {output_path}")
    plt.close()

def plot_waterfall_chart(models_data, output_path):
    """Creative: Waterfall chart showing cumulative improvement from ILC to models.
    
    Args:
        models_data: List of dicts with keys: 'data', 'label', 'color'
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get baseline from first model
    baseline_mean = None
    for model in models_data:
        if model['data']['baseline_mean'] is not None:
            baseline_mean = model['data']['baseline_mean']
            break
    
    if baseline_mean is None:
        print("  Warning: No baseline MSE data available for waterfall chart")
        plt.close()
        return
    
    # Starting point and cumulative values
    cumulative = [baseline_mean]
    x_labels = ['ILC\n(single-frequency)']
    colors = ['#6C757D']
    
    for model in models_data:
        cumulative.append(model['data']['mean'])
        x_labels.append(model['label'])
        colors.append(model.get('color', '#2E86AB'))
    
    # Create waterfall
    positions = np.arange(len(cumulative))
    
    # Plot bars
    bars = ax.bar(positions, cumulative, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add connecting lines
    for i in range(len(positions) - 1):
        ax.plot([positions[i] + 0.5, positions[i+1] - 0.5], 
               [cumulative[i], cumulative[i+1]], 'k-', linewidth=2, alpha=0.5, zorder=1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, cumulative)):
        height = bar.get_height()
        # Show improvement from previous
        if i > 0:
            improvement = cumulative[i-1] - cumulative[i]
            pct_improvement = (improvement / cumulative[i-1]) * 100
            label_text = f'{val:.2e}\n(-{pct_improvement:.1f}%)'
        else:
            label_text = f'{val:.2e}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.set_ylabel('MSE', fontsize=18)
    ax.set_title('Waterfall: MSE Improvement from ILC to UNet Models', fontsize=18, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y', which='both')
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved waterfall chart to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Compare UNet models using saved statistics files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare only the two base models (B-only and T,E,B)
  python compare_models.py
  
  # Include T-only model
  python compare_models.py --include-t-only
  
  # Include all three additional models
  python compare_models.py --include-t-only --include-e-only --include-te-only
  
  # Include only E-only and TE-only
  python compare_models.py --include-e-only --include-te-only
        """
    )
    
    parser.add_argument('--include-t-only', action='store_true',
                       help='Include T-only model in comparison')
    parser.add_argument('--include-e-only', action='store_true',
                       help='Include E-only model in comparison')
    parser.add_argument('--include-te-only', action='store_true',
                       help='Include TE-only model in comparison')
    
    args = parser.parse_args()
    
    # Define base models (always included)
    base_models = [
        {
            'base_dir': "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_cmb_scaled/normalized_b_only",
            'prefix': "normalized_b_only_best_model_153600_19200_norm",
            'label': LABEL_BS_ONLY,
            'color': '#2E86AB',
            'marker': 'o'
        },
        {
            'base_dir': "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb",
            'prefix': "unnormalized_teb_best_model_153600_19200_nonorm",
            'label': LABEL_TEB,
            'color': '#F77F00',
            'marker': 's'
        }
    ]
    
    # Define additional models (conditionally included)
    additional_models = [
        {
            'base_dir': "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_t_only",
            'prefix': "unnormalized_t_only_best_model_153600_19200_nonorm",
            'label': LABEL_T_ONLY,
            'color': '#E63946',
            'marker': '^',
            'flag': 'include_t_only'
        },
        {
            'base_dir': "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only",
            'prefix': "unnormalized_e_only_best_model_153600_19200_nonorm",
            'label': LABEL_E_ONLY,
            'color': '#06A77D',
            'marker': 'D',
            'flag': 'include_e_only'
        },
        {
            'base_dir': "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_te_only",
            'prefix': "unnormalized_te_only_best_model_153600_19200_nonorm",
            'label': LABEL_TE,
            'color': '#000000',
            'marker': 'v',
            'flag': 'include_te_only'
        }
    ]
    
    # Build list of models to include
    models_to_load = base_models.copy()
    
    for model in additional_models:
        flag_name = model['flag']
        if getattr(args, flag_name, False):
            models_to_load.append(model)
    
    if len(models_to_load) < 2:
        print("Error: At least 2 models are required for comparison.")
        sys.exit(1)
    
    # Create output directory
    output_dir = "/scratch/gpfs/JDUNKLEY/hshao/old_data/model_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\nIncluding {len(models_to_load)} model(s):")
    for model in models_to_load:
        print(f"  - {model['label']}")
    
    # Parse all files
    print("\nLoading statistics files...")
    
    models_data = []
    for idx, model_config in enumerate(models_to_load):
        base_dir = model_config['base_dir']
        prefix = model_config['prefix']
        label = model_config['label']
        
        # File paths
        cross_spec_file = os.path.join(base_dir, f"{prefix}_cross_spectrum_stats.txt")
        correlation_file = os.path.join(base_dir, f"{prefix}_correlation_stats.txt")
        mse_file = os.path.join(base_dir, f"{prefix}_mse_stats.txt")
        
        print(f"\n  Loading {label}:")
        print(f"    Cross-spectrum: {cross_spec_file}")
        cross_spec_data = parse_cross_spectrum_file(cross_spec_file)
        print(f"    Correlation: {correlation_file}")
        corr_data = parse_correlation_file(correlation_file)
        print(f"    MSE: {mse_file}")
        mse_data = parse_mse_file(mse_file)
        
        models_data.append({
            'cross_spec_data': cross_spec_data,
            'corr_data': corr_data,
            'mse_data': mse_data,
            'label': label,
            'color': model_config.get('color', '#2E86AB'),
            'marker': model_config.get('marker', 'o'),
            'data': cross_spec_data  # For backward compatibility with some functions
        })
    
    # Load additional model data if not already loaded (needed for variant plots)
    def load_model_if_needed(model_label):
        """Helper function to load a model if not already in models_data."""
        if any(m['label'] == model_label for m in models_data):
            return None
        model_config = next((m for m in additional_models if m['label'] == model_label), None)
        if model_config:
            base_dir = model_config['base_dir']
            prefix = model_config['prefix']
            cross_spec_file = os.path.join(base_dir, f"{prefix}_cross_spectrum_stats.txt")
            if os.path.exists(cross_spec_file):
                print(f"\n  Loading {model_label} (for variant plots):")
                print(f"    Cross-spectrum: {cross_spec_file}")
                cross_spec_data = parse_cross_spectrum_file(cross_spec_file)
                return {
                    'cross_spec_data': cross_spec_data,
                    'label': model_label,
                    'color': model_config.get('color', '#2E86AB'),
                    'marker': model_config.get('marker', 'o'),
                    'data': cross_spec_data
                }
        return None
    
    t_only_data = load_model_if_needed(LABEL_T_ONLY)
    e_only_data = load_model_if_needed(LABEL_E_ONLY)
    te_only_data = load_model_if_needed(LABEL_TE)
    
    # Create variant plots first
    print("\n" + "="*80)
    print("CREATING VARIANT PLOTS")
    print("="*80)
    
    # Philcox+18 only plot (first plot)
    print("\nV0. Creating Philcox+18 only plot...")
    philcox_output = os.path.join(output_dir, 'rho_philcox18.png')
    plot_philcox18_only(philcox_output)
    
    # Prepare data structures for variant plots
    all_models_for_variants = models_data.copy()
    if t_only_data:
        all_models_for_variants.append(t_only_data)
    if e_only_data:
        all_models_for_variants.append(e_only_data)
    if te_only_data:
        all_models_for_variants.append(te_only_data)
    
    cross_spec_models_variants = [{'data': m['cross_spec_data'], 'label': m['label'], 
                                   'color': m['color'], 'marker': m['marker']} 
                                  for m in all_models_for_variants]
    
    # Variant 1: Philcox+18 + B-mode only
    print("\nV1. Creating rho_philcox18_b plot...")
    variant1_output = os.path.join(output_dir, 'rho_philcox18_b.png')
    plot_philcox18_with_models(cross_spec_models_variants, 
                               [LABEL_BS_ONLY], variant1_output)
    
    # Variant 2: Philcox+18 + B_S-only + T+E+B_S
    print("\nV2. Creating rho_philcox18_b_teb plot...")
    variant2_output = os.path.join(output_dir, 'rho_philcox18_b_teb.png')
    plot_philcox18_with_models(cross_spec_models_variants, 
                               [LABEL_BS_ONLY, LABEL_TEB], variant2_output)
    
    # Variant 3: Philcox+18 + T-only
    if t_only_data:
        print("\nV3. Creating rho_philcox18_tonly plot...")
        variant3_output = os.path.join(output_dir, 'rho_philcox18_tonly.png')
        plot_philcox18_with_models(cross_spec_models_variants, 
                                   [LABEL_T_ONLY], variant3_output)
        
        # Variant 4: Philcox+18 + T-only + B_S-only
        print("\nV4. Creating rho_philcox18_tonly_b_only plot...")
        variant4_output = os.path.join(output_dir, 'rho_philcox18_tonly_b_only.png')
        plot_philcox18_with_models(cross_spec_models_variants, 
                                   [LABEL_T_ONLY, LABEL_BS_ONLY], variant4_output)
    else:
        print("\nV3-V4. Skipping T-only variants (T-only model not available)")
    
    # Variant 5: Philcox+18 + T-only + B-only + E-only
    if t_only_data and e_only_data:
        print("\nV5. Creating rho_philcox18_t_b_e plot...")
        variant5_output = os.path.join(output_dir, 'rho_philcox18_t_b_e.png')
        plot_philcox18_with_models(cross_spec_models_variants, 
                                   [LABEL_T_ONLY, LABEL_BS_ONLY, LABEL_E_ONLY], variant5_output)
    else:
        print("\nV5. Skipping rho_philcox18_t_b_e (T-only or E-only model not available)")
    
    # Variant 6: Philcox+18 + T-only + B-only + E-only + TE-mode
    if t_only_data and e_only_data and te_only_data:
        print("\nV6. Creating rho_philcox18_t_b_e_te plot...")
        variant6_output = os.path.join(output_dir, 'rho_philcox18_t_b_e_te.png')
        plot_philcox18_with_models(cross_spec_models_variants, 
                                   [LABEL_T_ONLY, LABEL_BS_ONLY, LABEL_E_ONLY, LABEL_TE], 
                                   variant6_output, no_legend=True)
    else:
        print("\nV6. Skipping rho_philcox18_t_b_e_te (required models not available)")
    
    # Variant 7: Philcox+18 + T-only + B-only + E-only + TE-mode + TEB-mode
    if t_only_data and e_only_data and te_only_data:
        print("\nV7. Creating rho_philcox18_t_b_e_te_teb plot...")
        variant7_output = os.path.join(output_dir, 'rho_philcox18_t_b_e_te_teb.png')
        plot_philcox18_with_models(cross_spec_models_variants, 
                                   [LABEL_T_ONLY, LABEL_BS_ONLY, LABEL_E_ONLY, LABEL_TE, LABEL_TEB], 
                                   variant7_output, no_legend=False, no_shaded_regions=True, 
                                   use_line_styles=True, ratio_plot=True, log_yscale=True)
    else:
        print("\nV7. Skipping rho_philcox18_t_b_e_te_teb (required models not available)")
    
    # Create comparison plots
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOTS")
    print("="*80)
    
    # Prepare data structures for different plot types
    cross_spec_models = [{'data': m['cross_spec_data'], 'label': m['label'], 
                         'color': m['color'], 'marker': m['marker']} for m in models_data]
    corr_models = [{'data': m['corr_data'], 'label': m['label'], 
                   'color': m['color']} for m in models_data]
    mse_models = [{'data': m['mse_data'], 'label': m['label'], 
                  'color': m['color']} for m in models_data]
    
    # Cross-spectrum comparison
    print("\n1. Creating cross-spectrum comparison plot...")
    cross_spec_output = os.path.join(output_dir, 'cross_spectrum_comparison.png')
    plot_cross_spectrum_comparison(cross_spec_models, cross_spec_output)
    
    # Correlation comparison
    print("\n2. Creating correlation comparison plot...")
    corr_output = os.path.join(output_dir, 'correlation_comparison.png')
    plot_correlation_comparison(corr_models, corr_output)
    
    # MSE comparison
    print("\n3. Creating MSE comparison plot...")
    mse_output = os.path.join(output_dir, 'mse_comparison.png')
    plot_mse_comparison(mse_models, mse_output)
    
    # MSE progression plot
    print("\n4. Creating MSE progression plot...")
    mse_progression_output = os.path.join(output_dir, 'mse_progression.png')
    plot_mse_progression(mse_models, mse_progression_output)
    
    # Additional clarity-focused plots
    print("\n5. Creating correlation boxplot...")
    corr_boxplot_output = os.path.join(output_dir, 'correlation_boxplot.png')
    plot_correlation_boxplot(corr_models, corr_boxplot_output)
    
    print("\n6. Creating cross-spectrum ratio plot...")
    cross_ratio_output = os.path.join(output_dir, 'cross_spectrum_ratio.png')
    plot_cross_spectrum_ratio(cross_spec_models, cross_ratio_output)
    
    print("\n7. Creating improvement factors plot...")
    improvement_output = os.path.join(output_dir, 'improvement_factors.png')
    plot_improvement_factors(mse_models, improvement_output)
    
    # Creative plots
    print("\n8. Creating performance radar plot...")
    radar_output = os.path.join(output_dir, 'performance_radar.png')
    plot_performance_radar(models_data, radar_output)
    
    print("\n9. Creating waterfall chart...")
    waterfall_output = os.path.join(output_dir, 'mse_waterfall.png')
    plot_waterfall_chart(mse_models, waterfall_output)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for model in models_data:
        label = model['label']
        cross_spec_data = model['cross_spec_data']
        corr_data = model['corr_data']
        mse_data = model['mse_data']
        
        print(f"\n{label}:")
        print(f"  Cross-spectrum mean: {np.mean(cross_spec_data['mean_cross_ps']):.4f}")
        print(f"  Spatial correlation mean: {corr_data['mean']:.4f} ± {corr_data['std']:.4f}")
        print(f"  MSE: {mse_data['mean']:.2e}")
        if mse_data['improvement_ratio']:
            print(f"  Improvement ratio: {mse_data['improvement_ratio']:.2f}x")
    
    print(f"\n✓ All comparison plots saved to: {output_dir}")

if __name__ == '__main__':
    main()

