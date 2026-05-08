import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
import argparse
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import plot_params
plot_params.setup_latex_path()
plot_params.patch_texmanager()
plot_params.setup_latex_preamble()
matplotlib.rcParams.update(plot_params.params)

def compute_mse(patch1, patch2):
    return np.mean((patch1 - patch2) ** 2)

def visualize_unet_residuals(npz_file, sample_indices=None, output_dir=None):
    print(f'\nLoading test results from: {npz_file}')
    data = np.load(npz_file)
    unet_targets = data['unet_targets']
    unet_predictions = data['unet_predictions']
    print(f'  Loaded arrays:')
    print(f'    unet_targets shape: {unet_targets.shape}')
    print(f'    unet_predictions shape: {unet_predictions.shape}')
    if sample_indices is None:
        sample_indices = [0]
    else:
        max_idx = len(unet_targets) - 1
        sample_indices = [idx for idx in sample_indices if 0 <= idx <= max_idx]
        if len(sample_indices) == 0:
            print(f'Warning: No valid sample indices. Using first sample.')
            sample_indices = [0]
    print(f'\nVisualizing samples: {sample_indices}')
    if output_dir is None:
        output_dir = os.path.dirname(npz_file)
    os.makedirs(output_dir, exist_ok=True)
    for sample_idx in sample_indices:
        print(f'\nProcessing sample {sample_idx}...')
        unet_target = unet_targets[sample_idx]
        unet_pred = unet_predictions[sample_idx]
        unet_residual = unet_target - unet_pred
        zeros_patch = np.zeros_like(unet_target)
        mse_ilc_vs_cmb = compute_mse(unet_target, zeros_patch)
        mse_target_vs_pred = compute_mse(unet_target, unet_pred)
        print(f'  MSE(ILC vs CMB): {mse_ilc_vs_cmb:.6e}')
        print(f'  MSE(Target vs Prediction): {mse_target_vs_pred:.6e}')
        (fig, axes) = plt.subplots(1, 3, figsize=(18, 5))
        vmin_shared = min(unet_target.min(), unet_pred.min())
        vmax_shared = max(unet_target.max(), unet_pred.max())
        ax1 = axes[0]
        im1 = ax1.imshow(unet_target, cmap='RdBu_r', vmin=vmin_shared, vmax=vmax_shared, origin='lower')
        ax1.set_title(f'ILC Residual (MSE = {mse_ilc_vs_cmb:.2e})', fontsize=14, fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.set_label('$\\mu$K', fontsize=14)
        cbar1.locator = MaxNLocator(nbins=6)
        cbar1.ax.tick_params(labelsize=12)
        cbar1.update_ticks()
        ax2 = axes[1]
        im2 = ax2.imshow(unet_pred, cmap='RdBu_r', vmin=vmin_shared, vmax=vmax_shared, origin='lower')
        ax2.set_title('UNet Prediction', fontsize=14, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('$\\mu$K', fontsize=14)
        cbar2.locator = MaxNLocator(nbins=6)
        cbar2.ax.tick_params(labelsize=12)
        cbar2.update_ticks()
        ax3 = axes[2]
        vmin_residual = unet_residual.min()
        vmax_residual = unet_residual.max()
        im3 = ax3.imshow(unet_residual, cmap='RdBu_r', vmin=vmin_residual, vmax=vmax_residual, origin='lower')
        ax3.set_title(f'UNet Residual (MSE = {mse_target_vs_pred:.2e})', fontsize=14, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes('right', size='5%', pad=0.05)
        cbar3 = plt.colorbar(im3, cax=cax3)
        cbar3.set_label('$\\mu$K', fontsize=14)
        cbar3.locator = MaxNLocator(nbins=6)
        cbar3.ax.tick_params(labelsize=12)
        cbar3.update_ticks()
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'unet_residuals_sample_{sample_idx}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'  Saved to: {output_path}')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize UNet predictions from test_results.npz')
    parser.add_argument('--npz-file', type=str, required=True, help='Path to test_results.npz file')
    parser.add_argument('--sample-indices', type=str, default=None, help='Comma-separated list of sample indices to visualize (e.g., "0,1,2"). Default: first sample')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots (default: same directory as npz_file)')
    args = parser.parse_args()
    sample_indices = None
    if args.sample_indices:
        sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
    visualize_unet_residuals(npz_file=args.npz_file, sample_indices=sample_indices, output_dir=args.output_dir)
    print('\n[ok] Visualization complete!')
if __name__ == '__main__':
    main()
