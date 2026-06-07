import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
import argparse
from tqdm import tqdm
import threading
import time
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import plot_params
plot_params.setup_latex_path()
plot_params.patch_texmanager()
plot_params.setup_latex_preamble()
matplotlib.rcParams.update(plot_params.params)
import architecture
from cmb_spectrum_2d import calculate_2d_spectrum
from paths_config import B_SMALL_DIR, E_T_DIR, ILC_DIR, OUTPUT_DIR_INTERSCALE as OUTPUT_DIR, STATS_FILE_16CH, STATS_FILE_8CH
STATS_FILE = STATS_FILE_8CH
FREQUENCIES = [95, 145, 220, 270]
import healpy as hp
NSIDE = 1024
PIX_SIZE = int(hp.nside2resol(NSIDE, arcmin=True))
DELTA_ELL = 50
ELL_CUTOFF = 2000
LMAX = 2000
PATCH_SIZE = 128
CONFIG = {'batch_size': 64, 'in_channels': 8, 'out_channels': 1, 'feature_dims': [32, 64, 128, 256, 512, 1024], 'negative_slope': 0.01, 'toy_model': 'InterscaleMultifreq'}

def gpu_warmup_task(device, stop_event, verbose=False):
    if device.type != 'cuda':
        return
    if verbose:
        print(f'\n[GPU Warmup] Starting background GPU warmup task on {device}')
    matrix_size = 512
    x = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    y = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    z = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float32)
    iteration = 0
    batch_count = 0
    while not stop_event.is_set():
        for _ in range(10):
            torch.matmul(x, y, out=z)
            z.add_(0.001)
            x.mul_(0.9999)
            x.add_(0.0001)
            iteration += 1
        if batch_count % 100 == 0:
            torch.cuda.synchronize(device)
        batch_count += 1
        if verbose and iteration % 10000000 == 0:
            print(f'[GPU Warmup] Running... ({iteration} iterations)')
        time.sleep(0.0001)
    if verbose:
        print(f'[GPU Warmup] Stopped after {iteration} iterations')

def start_gpu_warmup(device, verbose=False):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return (None, None)
    stop_event = threading.Event()
    warmup_thread = threading.Thread(target=gpu_warmup_task, args=(device, stop_event, verbose), daemon=True)
    warmup_thread.start()
    if verbose:
        print(f'[GPU Warmup] Background thread started')
    return (stop_event, warmup_thread)

def stop_gpu_warmup(stop_event, warmup_thread, verbose=False):
    if stop_event is None or warmup_thread is None:
        return
    if verbose:
        print(f'[GPU Warmup] Stopping background task...')
    stop_event.set()
    warmup_thread.join(timeout=5.0)
    if verbose:
        if warmup_thread.is_alive():
            print(f'[GPU Warmup] Thread still alive after timeout (may still stop later)')
        else:
            print(f'[GPU Warmup] Thread stopped successfully')

def load_test_data(normalize=True, use_b_small=True, use_ilc=True, use_e_t=False, n_subsample=None):
    print('\n' + '=' * 80)
    print('LOADING TEST DATA')
    print('=' * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (stop_event, warmup_thread) = start_gpu_warmup(device, verbose=True)
    if not use_ilc and (not use_b_small):
        raise ValueError('At least one of use_ilc or use_b_small must be True')
    stats_file = STATS_FILE_16CH if use_e_t else STATS_FILE_8CH
    ilc_residuals_file = f'{ILC_DIR}/ilc_residuals_b.npy'
    ilc_cmb_file = f'{ILC_DIR}/ILC_cmb_b.npy'
    cmb_draw_file = f'{B_SMALL_DIR}/sim1-150_freq220_cmb_draw_b_all_scales.npy'
    load_tasks = [('stats', stats_file)]
    if use_ilc:
        for freq in FREQUENCIES:
            load_tasks.append((f'ILC_fg_{freq}', f'{ILC_DIR}/ILC_foregrounds_b_{freq}.npy'))
    load_tasks.append(('ilc_residuals', ilc_residuals_file))
    load_tasks.append(('ilc_cmb', ilc_cmb_file))
    load_tasks.append(('cmb_draw', cmb_draw_file))
    if use_b_small:
        for freq in FREQUENCIES:
            load_tasks.append((f'B_small_{freq}', f'{B_SMALL_DIR}/sim1-150_freq{freq}_B_patches_small.npy'))
    if use_e_t:
        for freq in FREQUENCIES:
            load_tasks.append((f'E_{freq}', f'{E_T_DIR}/sim1-150_freq{freq}_E_patches.npy'))
        for freq in FREQUENCIES:
            load_tasks.append((f'T_{freq}', f'{E_T_DIR}/sim1-150_freq{freq}_T_patches.npy'))
    print(f'\nStats file: {stats_file}')
    print(f'  WARNING: IMPORTANT: Statistics are from TRAIN SET ONLY (no data leakage)')
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Normalization statistics file not found: {stats_file}")
    ILC_foregrounds = {}
    B_small = {}
    E_modes = {}
    T_modes = {}
    stats = None
    ilc_residuals = None
    ilc_cmb = None
    cmb_draw_b_all_scales = None
    with tqdm(load_tasks, desc='Loading test arrays', unit='file', dynamic_ncols=True) as pbar:
        for (label, path) in pbar:
            if not os.path.exists(path):
                raise FileNotFoundError(f'{label} file not found: {path}')
            t0 = time.perf_counter()
            data = np.load(path)
            dt = time.perf_counter() - t0
            pbar.set_postfix_str(f'{label} {dt:.1f}s', refresh=True)
            if label == 'stats':
                stats = data
            elif label.startswith('ILC_fg_'):
                freq = int(label.split('_')[-1])
                ILC_foregrounds[freq] = data
            elif label == 'ilc_residuals':
                ilc_residuals = data
            elif label == 'ilc_cmb':
                ilc_cmb = data
            elif label == 'cmb_draw':
                cmb_draw_b_all_scales = data
            elif label.startswith('B_small_'):
                freq = int(label.split('_')[-1])
                B_small[freq] = data
            elif label.startswith('E_'):
                freq = int(label.split('_')[-1])
                E_modes[freq] = data
            elif label.startswith('T_'):
                freq = int(label.split('_')[-1])
                T_modes[freq] = data
    if stats is None or ilc_residuals is None or ilc_cmb is None or cmb_draw_b_all_scales is None:
        raise RuntimeError('Incomplete load (stats or core ILC/CMB arrays missing)')
    test_indices = stats['test_indices']
    print(f'\n  Test indices: {len(test_indices)} samples')
    print('\nVerifying data consistency...')
    n_patches = ilc_residuals.shape[0]
    print(f'  Total patches: {n_patches}')
    ref_spatial_shape = ilc_residuals.shape[1:]
    assert ilc_cmb.shape[0] == n_patches, f'Mismatch: ILC_cmb has {ilc_cmb.shape[0]} patches, expected {n_patches}'
    assert ilc_cmb.shape[1:] == ref_spatial_shape, f'Mismatch: ILC_cmb has spatial shape {ilc_cmb.shape[1:]}, expected {ref_spatial_shape}'
    assert cmb_draw_b_all_scales.shape[0] == n_patches, f'Mismatch: cmb_draw_b_all_scales has {cmb_draw_b_all_scales.shape[0]} patches, expected {n_patches}'
    assert cmb_draw_b_all_scales.shape[1:] == ref_spatial_shape, f'Mismatch: cmb_draw_b_all_scales has spatial shape {cmb_draw_b_all_scales.shape[1:]}, expected {ref_spatial_shape}'
    if use_e_t:
        for freq in FREQUENCIES:
            assert E_modes[freq].shape[0] == n_patches, f'Mismatch: E mode at {freq} GHz has {E_modes[freq].shape[0]} patches, expected {n_patches}'
            assert E_modes[freq].shape[1:] == ref_spatial_shape, f'Mismatch: E mode at {freq} GHz has spatial shape {E_modes[freq].shape[1:]}, expected {ref_spatial_shape}'
            assert T_modes[freq].shape[0] == n_patches, f'Mismatch: T mode at {freq} GHz has {T_modes[freq].shape[0]} patches, expected {n_patches}'
            assert T_modes[freq].shape[1:] == ref_spatial_shape, f'Mismatch: T mode at {freq} GHz has spatial shape {T_modes[freq].shape[1:]}, expected {ref_spatial_shape}'
    print('  [ok] All datasets have matching patch counts and spatial dimensions')
    print('\nStacking input channels...')
    input_channels = []
    if use_ilc:
        input_channels.extend([ILC_foregrounds[95], ILC_foregrounds[145], ILC_foregrounds[220], ILC_foregrounds[270]])
    if use_b_small:
        input_channels.extend([B_small[95], B_small[145], B_small[220], B_small[270]])
    if use_e_t:
        input_channels.extend([E_modes[95], E_modes[145], E_modes[220], E_modes[270]])
        input_channels.extend([T_modes[95], T_modes[145], T_modes[220], T_modes[270]])
    UNet_input = np.stack(input_channels, axis=1)
    n_channels = len(input_channels)
    channel_desc_parts = []
    if use_ilc:
        channel_desc_parts.append('4 ILC foregrounds')
    if use_b_small:
        channel_desc_parts.append('4 B small-scale')
    if use_e_t:
        channel_desc_parts.append('4 E modes')
        channel_desc_parts.append('4 T modes')
    channel_desc = ' + '.join(channel_desc_parts)
    print(f'  Using {n_channels}-channel input ({channel_desc})')
    UNet_target = ilc_residuals[:, np.newaxis, :, :]
    print(f'\nExtracting test set using test_indices (ensuring alignment)...')
    test_X = UNet_input[test_indices]
    test_y = UNet_target[test_indices]
    test_ilc_cmb = ilc_cmb[test_indices]
    test_cmb_draw = cmb_draw_b_all_scales[test_indices]
    test_indices_final = test_indices.copy()
    if n_subsample is not None:
        n_test = len(test_X)
        n_subsample_actual = min(n_subsample, n_test)
        print(f'\nSubsampling first {n_subsample_actual} patches from {n_test} total test patches...')
        subsample_indices = np.arange(n_subsample_actual)
        test_X = test_X[subsample_indices]
        test_y = test_y[subsample_indices]
        test_ilc_cmb = test_ilc_cmb[subsample_indices]
        test_cmb_draw = test_cmb_draw[subsample_indices]
        test_indices_final = test_indices_final[subsample_indices]
        print(f'  After subsampling: {n_subsample_actual} patches')
    print(f'\nTest set shapes (all aligned using same test_indices):')
    print(f'  UNet input: {test_X.shape}')
    print(f'  UNet target: {test_y.shape}')
    print(f'  ILC CMB: {test_ilc_cmb.shape}')
    print(f'  Pure CMB (cmb_draw): {test_cmb_draw.shape}')
    assert test_X.shape[0] == test_y.shape[0] == test_ilc_cmb.shape[0] == test_cmb_draw.shape[0], 'Mismatch in test set sizes - alignment issue!'
    print('  [ok] All test datasets have matching sample counts (properly aligned)')
    if normalize:
        print('\nNormalizing test data using train statistics...')
        full_input_mean = stats['input_mean']
        full_input_std = stats['input_std']
        target_mean = stats['target_mean']
        target_std = stats['target_std']
        n_stats_channels = len(full_input_mean)
        expected_channels = 4 * (int(use_ilc) + int(use_b_small) + int(use_e_t) * 2)
        if use_e_t and n_stats_channels < 16:
            raise ValueError(f'Normalization stats file has only {n_stats_channels} channels, but {expected_channels} channels are requested with --add-e-t flag. Please regenerate the stats file with 16 channels.')
        elif n_stats_channels < expected_channels:
            raise ValueError(f'Normalization stats file has only {n_stats_channels} channels, but {expected_channels} channels are requested. Please regenerate the stats file with the correct number of channels.')
        input_mean = []
        input_std = []
        if use_ilc:
            input_mean.extend(full_input_mean[0:4])
            input_std.extend(full_input_std[0:4])
        if use_b_small:
            input_mean.extend(full_input_mean[4:8])
            input_std.extend(full_input_std[4:8])
        if use_e_t:
            input_mean.extend(full_input_mean[8:12])
            input_std.extend(full_input_std[8:12])
            input_mean.extend(full_input_mean[12:16])
            input_std.extend(full_input_std[12:16])
        input_mean = np.array(input_mean)
        input_std = np.array(input_std)
        n_channels = test_X.shape[1]
        for i in range(n_channels):
            test_X[:, i] = (test_X[:, i] - input_mean[i]) / (input_std[i] + 1e-08)
        test_y = (test_y - target_mean) / (target_std + 1e-08)
        print('  Normalization complete')
        stop_gpu_warmup(stop_event, warmup_thread, verbose=True)
        return (test_X, test_y, test_ilc_cmb, test_cmb_draw, test_indices_final, {'input_mean': input_mean, 'input_std': input_std, 'target_mean': target_mean, 'target_std': target_std})
    else:
        print('\nUsing unnormalized data')
        stop_gpu_warmup(stop_event, warmup_thread, verbose=True)
        return (test_X, test_y, test_ilc_cmb, test_cmb_draw, test_indices_final, None)

def load_model(model_path, normalize=True, in_channels=8):
    print(f'\nLoading model from: {model_path}')
    mean_init_path = None
    if not normalize:
        model_dir = os.path.dirname(model_path)
        mean_init_path = os.path.join(model_dir, 'mean_init_stats.pkl')
        if not os.path.exists(mean_init_path):
            stats = np.load(STATS_FILE)
            input_mean = stats['input_mean']
            input_std = stats['input_std']
            target_mean = stats['target_mean']
            target_std = stats['target_std']
            train_x_mean = float(np.mean(input_mean))
            train_x_std = float(np.mean(input_std))
            mean_init_dict = {'train_x_mean': train_x_mean, 'train_x_std': train_x_std, 'train_y_mean': float(target_mean), 'train_y_std': float(target_std)}
            import pickle
            os.makedirs(model_dir, exist_ok=True)
            with open(mean_init_path, 'wb') as f:
                pickle.dump(mean_init_dict, f)
            print(f'  Created mean_init_stats.pkl in model directory')
        else:
            print(f'  Using existing mean_init_stats.pkl from model directory')
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        print(f'  Using config from checkpoint:')
        print(f"    in_channels: {checkpoint_config.get('in_channels', in_channels)}")
        print(f"    feature_dims: {checkpoint_config.get('feature_dims', CONFIG['feature_dims'])}")
        actual_in_channels = checkpoint_config.get('in_channels', in_channels)
        actual_feature_dims = checkpoint_config.get('feature_dims', CONFIG['feature_dims'])
        actual_negative_slope = checkpoint_config.get('negative_slope', CONFIG['negative_slope'])
    else:
        print(f'  WARNING: Warning: No config in checkpoint, using CONFIG from script')
        actual_in_channels = in_channels
        actual_feature_dims = CONFIG['feature_dims']
        actual_negative_slope = CONFIG['negative_slope']
    model = architecture.UNET(in_channels=actual_in_channels, out_channels=CONFIG['out_channels'], feature_dims=actual_feature_dims, negative_slope=actual_negative_slope, mean_init=mean_init_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f'  Model loaded on device: {device}')
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.3e}")
    if 'val_correlation' in checkpoint:
        print(f"  Validation correlation: {checkpoint['val_correlation']:.3f}")
    return (model, device)

def generate_predictions(model, test_X, test_y, device, batch_size=32):
    print('\nGenerating predictions...')
    test_dataset = TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_y))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    targets = []
    with torch.no_grad():
        for (x, y) in test_loader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    print(f'  Predictions shape: {predictions.shape}')
    print(f'  Targets shape: {targets.shape}')
    return (predictions, targets)

def visualize_cmb_reconstructions(cmb_draw, ilc_cmb, unet_cmb_reconstructed, sample_indices, output_dir=None, ell_array=None, unet_cross_spectra_array=None, ilc_cross_spectra_array=None, mean_unet_cross_ps=None, percentile_lower_unet=None, percentile_upper_unet=None, mean_ilc_cross_ps=None, percentile_lower_ilc=None, percentile_upper_ilc=None, null_cross_spectra_array=None, mean_null_cross_ps_overall=None, unet_spatial_correlations=None, null_correlations_array=None, mean_null_correlation_overall=None, ilc_cmb_mse_array=None, unet_recon_mse_array=None, index_mapping=None):
    print(f'\nCreating 1x6 plots for {len(sample_indices)} test samples (saving each separately)...')
    n_samples = len(sample_indices)
    n_cols = 6
    for (row_idx, sample_idx) in enumerate(sample_indices):
        if index_mapping is not None:
            if sample_idx not in index_mapping:
                print(f'  Skipping sample {sample_idx} (not in subsampled data)')
                continue
            array_idx = index_mapping[sample_idx]
        else:
            array_idx = sample_idx
        (fig, axes) = plt.subplots(1, n_cols, figsize=(18, 3))
        axes = axes.reshape(1, -1) if n_cols > 1 else np.array([[axes]])
        pure_cmb = cmb_draw[array_idx]
        ilc_cmb_patch = ilc_cmb[array_idx]
        unet_cmb_patch = unet_cmb_reconstructed[array_idx]
        vmin = min(pure_cmb.min(), ilc_cmb_patch.min(), unet_cmb_patch.min())
        vmax = max(pure_cmb.max(), ilc_cmb_patch.max(), unet_cmb_patch.max())
        im1 = axes[0, 0].imshow(pure_cmb, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 0].set_title(f'Sample {sample_idx}\nPure Primordial CMB', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        im2 = axes[0, 1].imshow(ilc_cmb_patch, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 1].set_title(f'Sample {sample_idx}\nILC CMB Reconstruction', fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        im3 = axes[0, 2].imshow(unet_cmb_patch, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 2].set_title(f'Sample {sample_idx}\nUNet CMB Reconstructed', fontsize=10, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        pure_flat = pure_cmb.flatten()
        ilc_flat = ilc_cmb_patch.flatten()
        unet_flat = unet_cmb_patch.flatten()
        corr_ilc = np.corrcoef(pure_flat, ilc_flat)[0, 1]
        corr_unet = np.corrcoef(pure_flat, unet_flat)[0, 1]
        axes[0, 2].text(0.02, 0.98, f'ILC corr: {corr_ilc:.3f}\nUNet corr: {corr_unet:.3f}', transform=axes[0, 2].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
        ax4 = axes[0, 3]
        if ell_array is not None and unet_cross_spectra_array is not None and (ilc_cross_spectra_array is not None):
            if array_idx < len(unet_cross_spectra_array):
                unet_cross_ps_sample = unet_cross_spectra_array[array_idx]
                ilc_cross_ps_sample = ilc_cross_spectra_array[array_idx]
                if mean_unet_cross_ps is not None and percentile_lower_unet is not None and (percentile_upper_unet is not None):
                    ax4.fill_between(ell_array, percentile_lower_unet, percentile_upper_unet, color='blue', alpha=0.2, label='UNet 16-84th percentile')
                    ax4.plot(ell_array, mean_unet_cross_ps, 'b-', linewidth=2, alpha=0.6, label='UNet mean')
                if mean_ilc_cross_ps is not None and percentile_lower_ilc is not None and (percentile_upper_ilc is not None):
                    ax4.fill_between(ell_array, percentile_lower_ilc, percentile_upper_ilc, color='orange', alpha=0.2, label='ILC 16-84th percentile')
                    ax4.plot(ell_array, mean_ilc_cross_ps, 'orange', linewidth=2, alpha=0.6, label='ILC mean')
                ax4.plot(ell_array, unet_cross_ps_sample, 'b-', linewidth=1.5, alpha=0.8, label='UNet (this sample)')
                ax4.plot(ell_array, ilc_cross_ps_sample, 'orange', linewidth=1.5, alpha=0.8, linestyle='--', label='ILC (this sample)')
                if mean_null_cross_ps_overall is not None:
                    ax4.plot(ell_array, mean_null_cross_ps_overall, 'gray', linewidth=1.5, linestyle=':', alpha=0.7, label='Null (mean)')
                ax4.set_xlabel('$\\ell$ (Multipole)', fontsize=9)
                ax4.set_ylabel('Normalized Cross Power', fontsize=9)
                ax4.set_title(f'Sample {sample_idx}\nCross-Power Spectra', fontsize=10, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.legend(fontsize=7, loc='best')
            else:
                ax4.axis('off')
                ax4.text(0.5, 0.5, f'Sample {sample_idx}\nNo cross-spectrum data', ha='center', va='center', fontsize=10)
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, f'Column 4\n(No cross-spectrum data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax5 = axes[0, 4]
        if null_correlations_array is not None and unet_spatial_correlations is not None:
            if array_idx < len(unet_spatial_correlations):
                actual_corr = unet_spatial_correlations[array_idx]
                bins = np.linspace(-0.2, 1.0, 30)
                ax5.hist(null_correlations_array, bins=bins, alpha=0.7, color='lightblue', edgecolor='black', label=f'Null correlations (N={len(null_correlations_array)})')
                if mean_null_correlation_overall is not None:
                    ax5.axvline(mean_null_correlation_overall, color='gray', linestyle='-', linewidth=3, label=f'Mean null: {mean_null_correlation_overall:.4f}')
                ax5.axvline(actual_corr, color='green', linestyle='--', linewidth=3, label=f'Actual: {actual_corr:.4f}')
                ax5.set_xlabel('Spatial Correlation', fontsize=9)
                ax5.set_ylabel('Frequency', fontsize=9)
                ax5.set_title(f'Sample {sample_idx}\nNull Correlation Test', fontsize=10, fontweight='bold')
                ax5.legend(fontsize=7, loc='best')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.axis('off')
                ax5.text(0.5, 0.5, f'Sample {sample_idx}\nNo correlation data', ha='center', va='center', fontsize=10)
        else:
            ax5.axis('off')
            ax5.text(0.5, 0.5, f'Column 5\n(No correlation data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax6 = axes[0, 5]
        if ilc_cmb_mse_array is not None and unet_recon_mse_array is not None:
            hb = ax6.hexbin(ilc_cmb_mse_array, unet_recon_mse_array, gridsize=50, cmap='Blues', mincnt=1, norm=plt.matplotlib.colors.LogNorm(), edgecolors='none', alpha=0.8, zorder=2)
            cbar = plt.colorbar(hb, ax=ax6, fraction=0.046, pad=0.04)
            cbar.set_label('Count', fontsize=8)
            mse_min = min(ilc_cmb_mse_array.min(), unet_recon_mse_array.min())
            mse_max = max(ilc_cmb_mse_array.max(), unet_recon_mse_array.max())
            ax6.plot([mse_min, mse_max], [mse_min, mse_max], 'r--', linewidth=2, label='y=x', zorder=3, alpha=0.8)
            if array_idx < len(ilc_cmb_mse_array):
                ax6.scatter(ilc_cmb_mse_array[array_idx], unet_recon_mse_array[array_idx], s=150, c='green', marker='*', edgecolors='black', linewidths=1.5, zorder=4, label=f'Sample {sample_idx}')
            ax6.set_xlabel('ILC CMB MSE', fontsize=9)
            ax6.set_ylabel('UNet CMB MSE', fontsize=9)
            ax6.set_title('MSE Comparison\n(All Patches)', fontsize=10, fontweight='bold')
            ax6.set_aspect('equal', adjustable='box')
            ax6.grid(True, alpha=0.3, zorder=1)
            ax6.legend(loc='lower right', fontsize=7)
        else:
            ax6.axis('off')
            ax6.text(0.5, 0.5, f'Column 6\n(No MSE data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        fig.suptitle(f'Sample {sample_idx}', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        if output_dir:
            cmb_viz_subdir = os.path.join(output_dir, 'cmb_reconstructions')
            os.makedirs(cmb_viz_subdir, exist_ok=True)
            sample_output_path = os.path.join(cmb_viz_subdir, f'cmb_reconstruction_sample_{sample_idx}.png')
            plt.savefig(sample_output_path, dpi=150, bbox_inches='tight')
            print(f'  Saved sample {sample_idx} to: {sample_output_path}')
        else:
            print(f'  Displaying sample {sample_idx}...')
            plt.show()
        plt.close()

def visualize_fg_reconstructions(b_all_scales_220, ILC_fg_reconstruction, unet_fg_220_reconstruction, sample_indices, output_dir=None, ell_array=None, unet_fg_cross_spectra_array=None, mean_unet_fg_cross_ps=None, percentile_lower_unet_fg=None, percentile_upper_unet_fg=None, null_cross_spectra_array=None, mean_null_cross_ps_overall=None, unet_fg_spatial_correlations=None, null_correlations_array=None, mean_null_correlation_overall=None, ILC_fg_mse_array=None, unet_fg_mse_array=None):
    print(f'\nCreating foreground reconstruction plots (2x3) for {len(sample_indices)} test samples (saving each separately)...')
    n_samples = len(sample_indices)
    for (row_idx, sample_idx) in enumerate(sample_indices):
        (fig, axes) = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'wspace': 0.27, 'hspace': 0.2})
        ilc_fg_patch = ILC_fg_reconstruction[sample_idx]
        true_fg_patch = b_all_scales_220[sample_idx]
        unet_fg_patch = unet_fg_220_reconstruction[sample_idx]
        (vmin1, vmax1) = (true_fg_patch.min(), true_fg_patch.max())
        (vmin2, vmax2) = (ilc_fg_patch.min(), ilc_fg_patch.max())
        (vmin3, vmax3) = (unet_fg_patch.min(), unet_fg_patch.max())
        ax1 = axes[0, 0]
        im1 = ax1.imshow(true_fg_patch, cmap='RdYlBu_r', vmin=vmin1, vmax=vmax1, origin='lower')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.set_label('$\\mu$K', fontsize=16)
        cbar1.locator = MaxNLocator(nbins=6)
        cbar1.ax.tick_params(labelsize=16)
        cbar1.update_ticks()
        ax1.set_title('True Foreground\\\\B-modes (220 GHz)', fontsize=16, fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2 = axes[0, 1]
        im2 = ax2.imshow(ilc_fg_patch, cmap='RdYlBu_r', vmin=vmin2, vmax=vmax2, origin='lower')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('$\\mu$K', fontsize=16)
        cbar2.locator = MaxNLocator(nbins=6)
        cbar2.ax.tick_params(labelsize=14)
        cbar2.update_ticks()
        ax2.set_title('ILC Foreground\\\\Reconstruction (All Scales)', fontsize=16, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3 = axes[0, 2]
        im3 = ax3.imshow(unet_fg_patch, cmap='RdYlBu_r', vmin=vmin3, vmax=vmax3, origin='lower')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes('right', size='5%', pad=0.05)
        cbar3 = plt.colorbar(im3, cax=cax3)
        cbar3.set_label('$\\mu$K', fontsize=16)
        cbar3.locator = MaxNLocator(nbins=6)
        cbar3.ax.tick_params(labelsize=14)
        cbar3.update_ticks()
        ax3.set_title('UNet Foreground\\\\Reconstruction (220 GHz)', fontsize=16, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
        true_fg_flat = true_fg_patch.flatten()
        ilc_fg_flat = ilc_fg_patch.flatten()
        unet_fg_flat = unet_fg_patch.flatten()
        corr_ilc = np.corrcoef(true_fg_flat, ilc_fg_flat)[0, 1]
        corr_unet = np.corrcoef(true_fg_flat, unet_fg_flat)[0, 1]
        ax3.text(0.02, 0.98, f'ILC corr: {corr_ilc:.3f}\nUNet corr: {corr_unet:.3f}', transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
        ax4 = axes[1, 0]
        if ell_array is not None and unet_fg_cross_spectra_array is not None:
            if sample_idx < len(unet_fg_cross_spectra_array):
                unet_fg_cross_ps_sample = unet_fg_cross_spectra_array[sample_idx]
                if mean_unet_fg_cross_ps is not None and percentile_lower_unet_fg is not None and (percentile_upper_unet_fg is not None):
                    ax4.fill_between(ell_array, percentile_lower_unet_fg, percentile_upper_unet_fg, color='blue', alpha=0.2, label='16-84th percentile')
                    ax4.plot(ell_array, mean_unet_fg_cross_ps, 'b-', linewidth=2, alpha=0.6, label='Mean', linestyle='--')
                ax4.plot(ell_array, unet_fg_cross_ps_sample, 'green', linewidth=1.5, alpha=0.8, label='This sample')
                if mean_null_cross_ps_overall is not None:
                    ax4.plot(ell_array, mean_null_cross_ps_overall, 'gray', linewidth=1.5, linestyle=':', alpha=0.7, label='Null (mean)')
                ax4.set_xlabel('$\\ell$ (Multipole)', fontsize=16)
                ax4.set_ylabel('$\\tilde{C}_\\ell^{True,UNet}$', fontsize=16)
                ax4.set_title('Normalized Cross-Power Spectra', fontsize=16, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.legend(fontsize=12, loc='best', ncol=2)
                ax4.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax4.yaxis.set_major_locator(MaxNLocator(nbins=6))
            else:
                ax4.axis('off')
                ax4.text(0.5, 0.5, f'Sample {sample_idx}\nNo cross-spectrum data', ha='center', va='center', fontsize=10)
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, f'Column 4\n(No cross-spectrum data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax5 = axes[1, 1]
        if null_correlations_array is not None and unet_fg_spatial_correlations is not None:
            if sample_idx < len(unet_fg_spatial_correlations):
                actual_corr = unet_fg_spatial_correlations[sample_idx]
                bins = np.linspace(-0.2, 1.0, 30)
                ax5.hist(null_correlations_array, bins=bins, alpha=0.7, color='lightblue', edgecolor='#143d80')
                if mean_null_correlation_overall is not None:
                    ax5.axvline(mean_null_correlation_overall, color='#143d80', linestyle='--', linewidth=3, label='Null mean')
                ax5.axvline(actual_corr, color='green', linestyle='-', linewidth=3, label='This sample')
                ax5.set_xlabel('Spatial Correlation', fontsize=16)
                ax5.set_ylabel('Frequency', fontsize=16)
                ax5.set_title('Null Correlation Test', fontsize=16, fontweight='bold')
                if unet_fg_spatial_correlations is not None and len(unet_fg_spatial_correlations) > 0:
                    mean_corr = np.mean(unet_fg_spatial_correlations)
                    ax5.text(0.4, 0.2, f'This sample: {actual_corr:.3f}\nMean correlation: {mean_corr:.3f}', transform=ax5.transAxes, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=16)
                ax5.legend(fontsize=16, loc='best')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.axis('off')
                ax5.text(0.5, 0.5, f'Sample {sample_idx}\nNo correlation data', ha='center', va='center', fontsize=10)
        else:
            ax5.axis('off')
            ax5.text(0.5, 0.5, f'Column 5\n(No correlation data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax6 = axes[1, 2]
        if ILC_fg_mse_array is not None and unet_fg_mse_array is not None:
            hb = ax6.hexbin(ILC_fg_mse_array, unet_fg_mse_array, gridsize=50, cmap='Blues', mincnt=1, norm=plt.matplotlib.colors.LogNorm(), edgecolors='none', alpha=0.8, zorder=2)
            cbar = plt.colorbar(hb, ax=ax6, fraction=0.046, pad=0.04)
            cbar.set_label('Count', fontsize=12)
            mse_min = min(ILC_fg_mse_array.min(), unet_fg_mse_array.min())
            mse_max = max(ILC_fg_mse_array.max(), unet_fg_mse_array.max())
            ax6.plot([mse_min, mse_max], [mse_min, mse_max], 'r--', linewidth=2, label='y=x', zorder=3, alpha=0.8)
            if sample_idx < len(ILC_fg_mse_array):
                ax6.scatter(ILC_fg_mse_array[sample_idx], unet_fg_mse_array[sample_idx], s=150, c='green', marker='*', edgecolors='black', linewidths=1.5, zorder=4, label=f'Sample {sample_idx}')
            ax6.set_xlabel('ILC Foreground MSE', fontsize=16)
            ax6.set_ylabel('UNet Foreground MSE', fontsize=16)
            ax6.set_title('MSE Comparison\n(All Patches)', fontsize=16, fontweight='bold')
            ax6.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
            ax6.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
            ax6.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax6.grid(True, alpha=0.3, zorder=1)
            ax6.legend(loc='lower right', fontsize=12)
        else:
            ax6.axis('off')
            ax6.text(0.5, 0.5, f'Column 6\n(No MSE data)', ha='center', va='center', fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        fig.suptitle(f'Sample {sample_idx}', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(w_pad=0.05)
        if output_dir:
            fg_viz_subdir = os.path.join(output_dir, 'fg_reconstructions')
            os.makedirs(fg_viz_subdir, exist_ok=True)
            sample_output_path = os.path.join(fg_viz_subdir, f'fg_reconstruction_sample_{sample_idx}.png')
            plt.savefig(sample_output_path, dpi=600, bbox_inches='tight')
            print(f'  Saved sample {sample_idx} to: {sample_output_path}')
        else:
            print(f'  Displaying sample {sample_idx}...')
            plt.show()
        plt.close()

def filter_patch(image, ell_cutoff=200, filter_dir='high_ell', pix_size=PIX_SIZE):
    N = image.shape[0]
    ell_scale_factor = 2.0 * np.pi
    image_fft = np.fft.fft2(image)
    image_fft_shifted = np.fft.fftshift(image_fft)
    ones = np.ones(N)
    inds = (np.arange(N) + 0.5 - N / 2.0) / (N - 1.0)
    kX = np.outer(ones, inds) / (pix_size / 60.0 * np.pi / 180.0)
    kY = np.transpose(kX)
    K = np.sqrt(kX ** 2.0 + kY ** 2.0)
    ell_grid = K * ell_scale_factor
    if filter_dir in ['high_ell', 'small_scale']:
        filter_mask = ell_grid > ell_cutoff
    elif filter_dir in ['low_ell', 'large_scale']:
        filter_mask = ell_grid < ell_cutoff
    else:
        raise ValueError(f"filter_dir must be 'high_ell'/'small_scale' or 'low_ell'/'large_scale', got {filter_dir}")
    filtered_fft = image_fft_shifted * filter_mask
    filtered_fft = np.fft.ifftshift(filtered_fft)
    filtered_image = np.fft.ifft2(filtered_fft).real
    return np.real(filtered_image)

def visualize_fg_reconstructions_large_scale(b_all_scales_220, ILC_fg_reconstruction, unet_fg_220_reconstruction, sample_indices, output_dir=None, ell_array=None, unet_fg_cross_spectra_array=None, mean_unet_fg_cross_ps=None, percentile_lower_unet_fg=None, percentile_upper_unet_fg=None, null_cross_spectra_array=None, mean_null_cross_ps_overall=None, unet_fg_spatial_correlations=None, null_correlations_array=None, mean_null_correlation_overall=None, ILC_fg_mse_array=None, unet_fg_mse_array=None, ell_cutoff=200):
    print(f'\nCreating LARGE-SCALE foreground reconstruction plots (2x3) for {len(sample_indices)} test samples (saving each separately)...')
    print(f'  Filtering to large scales: ell < {ell_cutoff}')
    n_samples = len(sample_indices)
    for (row_idx, sample_idx) in enumerate(sample_indices):
        (fig, axes) = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'wspace': 0.27, 'hspace': 0.2})
        ilc_fg_patch_all = ILC_fg_reconstruction[sample_idx]
        true_fg_patch_all = b_all_scales_220[sample_idx]
        unet_fg_patch_all = unet_fg_220_reconstruction[sample_idx]
        print(f'  Filtering sample {sample_idx} to large scales (ell < {ell_cutoff})...')
        true_fg_patch = filter_patch(true_fg_patch_all, ell_cutoff=ell_cutoff, filter_dir='large_scale', pix_size=PIX_SIZE)
        ilc_fg_patch = filter_patch(ilc_fg_patch_all, ell_cutoff=ell_cutoff, filter_dir='large_scale', pix_size=PIX_SIZE)
        unet_fg_patch = filter_patch(unet_fg_patch_all, ell_cutoff=ell_cutoff, filter_dir='large_scale', pix_size=PIX_SIZE)
        (vmin1, vmax1) = (true_fg_patch.min(), true_fg_patch.max())
        (vmin2, vmax2) = (ilc_fg_patch.min(), ilc_fg_patch.max())
        (vmin3, vmax3) = (unet_fg_patch.min(), unet_fg_patch.max())
        ax1 = axes[0, 0]
        im1 = ax1.imshow(true_fg_patch, cmap='RdYlBu_r', vmin=vmin1, vmax=vmax1, origin='lower')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.set_label('$\\mu$K', fontsize=16)
        cbar1.locator = MaxNLocator(nbins=6)
        cbar1.ax.tick_params(labelsize=16)
        cbar1.update_ticks()
        ax1.set_title('True Foreground\\\\B-modes ($\\ell < 200$)', fontsize=16, fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2 = axes[0, 1]
        im2 = ax2.imshow(ilc_fg_patch, cmap='RdYlBu_r', vmin=vmin2, vmax=vmax2, origin='lower')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('$\\mu$K', fontsize=16)
        cbar2.locator = MaxNLocator(nbins=6)
        cbar2.ax.tick_params(labelsize=14)
        cbar2.update_ticks()
        ax2.set_title('ILC Foreground\\\\Reconstruction ($\\ell < 200$)', fontsize=16, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3 = axes[0, 2]
        im3 = ax3.imshow(unet_fg_patch, cmap='RdYlBu_r', vmin=vmin3, vmax=vmax3, origin='lower')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes('right', size='5%', pad=0.05)
        cbar3 = plt.colorbar(im3, cax=cax3)
        cbar3.set_label('$\\mu$K', fontsize=16)
        cbar3.locator = MaxNLocator(nbins=6)
        cbar3.ax.tick_params(labelsize=14)
        cbar3.update_ticks()
        ax3.set_title('UNet Foreground\\\\Reconstruction ($\\ell < 200$)', fontsize=16, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
        true_fg_flat = true_fg_patch.flatten()
        ilc_fg_flat = ilc_fg_patch.flatten()
        unet_fg_flat = unet_fg_patch.flatten()
        corr_ilc = np.corrcoef(true_fg_flat, ilc_fg_flat)[0, 1]
        corr_unet = np.corrcoef(true_fg_flat, unet_fg_flat)[0, 1]
        ax3.text(0.02, 0.98, f'ILC corr: {corr_ilc:.3f}\nUNet corr: {corr_unet:.3f}', transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
        ax4 = axes[1, 0]
        if ell_array is not None and unet_fg_cross_spectra_array is not None:
            if sample_idx < len(unet_fg_cross_spectra_array):
                unet_fg_cross_ps_sample = unet_fg_cross_spectra_array[sample_idx]
                if mean_unet_fg_cross_ps is not None and percentile_lower_unet_fg is not None and (percentile_upper_unet_fg is not None):
                    ax4.fill_between(ell_array, percentile_lower_unet_fg, percentile_upper_unet_fg, color='blue', alpha=0.2, label='16-84th percentile')
                    ax4.plot(ell_array, mean_unet_fg_cross_ps, 'b-', linewidth=2, alpha=0.6, label='Mean', linestyle='--')
                ax4.plot(ell_array, unet_fg_cross_ps_sample, 'green', linewidth=1.5, alpha=0.8, label='This sample')
                if mean_null_cross_ps_overall is not None:
                    ax4.plot(ell_array, mean_null_cross_ps_overall, 'gray', linewidth=1.5, linestyle=':', alpha=0.7, label='Null (mean)')
                ax4.set_xlabel('$\\ell$ (Multipole)', fontsize=16)
                ax4.set_ylabel('$\\tilde{C}_\\ell^{True,UNet}$', fontsize=16)
                ax4.set_title(f'Normalized Cross-Power Spectra', fontsize=16, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.legend(fontsize=12, loc='best', ncol=2)
                ax4.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax4.yaxis.set_major_locator(MaxNLocator(nbins=6))
            else:
                ax4.axis('off')
                ax4.text(0.5, 0.5, f'Sample {sample_idx}\nNo cross-spectrum data', ha='center', va='center', fontsize=10)
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, f'Column 4\n(No cross-spectrum data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax5 = axes[1, 1]
        if null_correlations_array is not None and unet_fg_spatial_correlations is not None:
            if sample_idx < len(unet_fg_spatial_correlations):
                actual_corr = unet_fg_spatial_correlations[sample_idx]
                bins = np.linspace(-0.2, 1.0, 30)
                ax5.hist(null_correlations_array, bins=bins, alpha=0.7, color='lightblue', edgecolor='#143d80')
                if mean_null_correlation_overall is not None:
                    ax5.axvline(mean_null_correlation_overall, color='#143d80', linestyle='--', linewidth=3, label='Null mean')
                ax5.axvline(actual_corr, color='green', linestyle='-', linewidth=3, label='This sample')
                ax5.set_xlabel('Spatial Correlation', fontsize=16)
                ax5.set_ylabel('Frequency', fontsize=16)
                ax5.set_title('Null Correlation Test', fontsize=16, fontweight='bold')
                if unet_fg_spatial_correlations is not None and len(unet_fg_spatial_correlations) > 0:
                    mean_corr = np.mean(unet_fg_spatial_correlations)
                    ax5.text(0.4, 0.2, f'This sample: {actual_corr:.3f}\nMean correlation: {mean_corr:.3f}', transform=ax5.transAxes, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=16)
                ax5.legend(fontsize=16, loc='best')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.axis('off')
                ax5.text(0.5, 0.5, f'Sample {sample_idx}\nNo correlation data', ha='center', va='center', fontsize=10)
        else:
            ax5.axis('off')
            ax5.text(0.5, 0.5, f'Column 5\n(No correlation data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax6 = axes[1, 2]
        if ILC_fg_mse_array is not None and unet_fg_mse_array is not None:
            hb = ax6.hexbin(ILC_fg_mse_array, unet_fg_mse_array, gridsize=50, cmap='Blues', mincnt=1, norm=plt.matplotlib.colors.LogNorm(), edgecolors='none', alpha=0.8, zorder=2)
            cbar = plt.colorbar(hb, ax=ax6, fraction=0.046, pad=0.04)
            cbar.set_label('Count', fontsize=12)
            mse_min = 0.0009
            mse_max = 0.002
            ax6.plot([mse_min, mse_max], [mse_min, mse_max], 'r--', linewidth=2, label='y=x', zorder=3, alpha=0.8)
            if sample_idx < len(ILC_fg_mse_array):
                ax6.scatter(ILC_fg_mse_array[sample_idx], unet_fg_mse_array[sample_idx], s=150, c='green', marker='*', edgecolors='black', linewidths=1.5, zorder=4, label=f'Sample {sample_idx}')
            ax6.set_xlabel('ILC Foreground MSE', fontsize=16)
            ax6.set_ylabel('UNet Foreground MSE', fontsize=16)
            ax6.set_title('MSE Comparison\n(All Patches)', fontsize=16, fontweight='bold')
            ax6.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
            ax6.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
            ax6.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax6.grid(True, alpha=0.3, zorder=1)
            ax6.legend(loc='lower right', fontsize=12)
        else:
            ax6.axis('off')
            ax6.text(0.5, 0.5, f'Column 6\n(No MSE data)', ha='center', va='center', fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        fig.suptitle(f'Sample {sample_idx} - Large-Scale ($\\ell < {ell_cutoff}$)', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(w_pad=0.05)
        if output_dir:
            fg_viz_subdir = os.path.join(output_dir, 'fg_reconstructions_large_scale')
            os.makedirs(fg_viz_subdir, exist_ok=True)
            sample_output_path = os.path.join(fg_viz_subdir, f'fg_reconstruction_large_scale_sample_{sample_idx}.png')
            plt.savefig(sample_output_path, dpi=600, bbox_inches='tight')
            print(f'  Saved sample {sample_idx} to: {sample_output_path}')
        else:
            print(f'  Displaying sample {sample_idx}...')
            plt.show()
        plt.close()

def compute_normalized_cross_spectrum(map1, map2, delta_ell=DELTA_ELL, ell_max=ELL_CUTOFF, pix_size=PIX_SIZE, N=PATCH_SIZE):
    if calculate_2d_spectrum is None:
        raise ValueError('calculate_2d_spectrum not available - cannot compute cross-spectrum')
    (ell_array, cl_cross) = calculate_2d_spectrum(map1, map2, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N, lib_mode='NumPy')
    (_, cl_map1_map1) = calculate_2d_spectrum(map1, map1, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N, lib_mode='NumPy')
    (_, cl_map2_map2) = calculate_2d_spectrum(map2, map2, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N, lib_mode='NumPy')
    normalized_cross_ps = cl_cross / np.sqrt(cl_map1_map1 * cl_map2_map2)
    return (ell_array, normalized_cross_ps)

def visualize_compare_all_fg_reconstructions(b_all_scales, fg_reconstructions, config_names, sample_indices, output_dir=None, ell_arrays=None, cross_spectra_arrays=None, mean_cross_spectra=None, percentile_lower=None, percentile_upper=None, original_sample_indices=None, cmb_var_arrays=None):
    print(f'\nCreating comparison plots (1x4) for {len(sample_indices)} test samples...')
    n_cols = 4
    for (idx, array_pos) in enumerate(sample_indices):
        if original_sample_indices is not None and idx < len(original_sample_indices):
            display_idx = original_sample_indices[idx]
        else:
            display_idx = array_pos
        (fig, axes) = plt.subplots(1, n_cols, figsize=(20, 4))
        true_fg_patch = b_all_scales[array_pos]
        for col_idx in range(3):
            ax = axes[col_idx]
            if col_idx >= len(fg_reconstructions) or fg_reconstructions[col_idx] is None:
                ax.axis('off')
                ax.text(0.5, 0.5, f'{config_names[col_idx]}\n\nModel Not Found\n(Placeholder)', ha='center', va='center', fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                continue
            fg_recon = fg_reconstructions[col_idx][array_pos]
            (vmin, vmax) = (fg_recon.min(), fg_recon.max())
            im = ax.imshow(fg_recon, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, origin='lower')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('$\\mu$K', fontsize=14)
            cbar.locator = MaxNLocator(nbins=6)
            cbar.ax.tick_params(labelsize=12)
            cbar.update_ticks()
            true_fg_flat = true_fg_patch.flatten()
            fg_recon_flat = fg_recon.flatten()
            corr = np.corrcoef(true_fg_flat, fg_recon_flat)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            mse = np.mean((true_fg_patch - fg_recon) ** 2)
            mse_str = f', MSE: {mse:.3e}'
            title_text = f'{config_names[col_idx]} ($\\ell < 200$), Corr: {corr:.3f}{mse_str}'
            ax.set_title(title_text, fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        ax4 = axes[3]
        if ell_arrays is not None and cross_spectra_arrays is not None:
            if isinstance(ell_arrays, list):
                ell_array = ell_arrays[0]
            else:
                ell_array = ell_arrays
            colors = ['blue', 'green', 'orange']
            for col_idx in range(3):
                if cross_spectra_arrays is None or col_idx >= len(cross_spectra_arrays) or cross_spectra_arrays[col_idx] is None:
                    continue
                if array_pos < len(cross_spectra_arrays[col_idx]):
                    cross_ps_sample = cross_spectra_arrays[col_idx][array_pos]
                    if mean_cross_spectra is not None and col_idx < len(mean_cross_spectra) and (mean_cross_spectra[col_idx] is not None) and (percentile_lower is not None) and (col_idx < len(percentile_lower)) and (percentile_lower[col_idx] is not None) and (percentile_upper is not None) and (col_idx < len(percentile_upper)) and (percentile_upper[col_idx] is not None):
                        mean_ps = mean_cross_spectra[col_idx]
                        lower_ps = percentile_lower[col_idx]
                        upper_ps = percentile_upper[col_idx]
                        ax4.fill_between(ell_array, lower_ps, upper_ps, color=colors[col_idx], alpha=0.2, label=f'{config_names[col_idx]} 16-84th percentile')
                        ax4.plot(ell_array, mean_ps, color=colors[col_idx], linewidth=2, alpha=0.6, linestyle='--', label=f'{config_names[col_idx]} mean')
                    ax4.plot(ell_array, cross_ps_sample, color=colors[col_idx], linewidth=1.5, alpha=0.8, label=f'{config_names[col_idx]} (this sample)')
            ax4.set_xlabel('$\\ell$ (Multipole)', fontsize=14)
            ax4.set_ylabel('$\\tilde{C}_\\ell^{True,Recon}$', fontsize=14)
            ax4.set_title('Normalized Cross-Power Spectra ($\\ell < 200$)', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=10, loc='best', ncol=1)
            ax4.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax4.yaxis.set_major_locator(MaxNLocator(nbins=6))
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, 'No cross-spectrum data', ha='center', va='center', fontsize=12)
        fig.suptitle(f'Sample {display_idx} - Large-Scale ($\\ell < 200$)', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            sample_output_path = os.path.join(output_dir, f'compare_all_fg_large_scale_sample_{display_idx}.png')
            plt.savefig(sample_output_path, dpi=600, bbox_inches='tight')
            print(f'  Saved sample {display_idx} to: {sample_output_path}')
        else:
            plt.show()
        plt.close()

def find_test_results_npz(norm_suffix, channel_suffix):
    model_dir = os.path.join(OUTPUT_DIR, f'{norm_suffix}_{channel_suffix}')
    npz_path = os.path.join(model_dir, 'test_results.npz')
    if os.path.exists(npz_path):
        return npz_path
    else:
        return None

def load_saved_test_results(npz_path, test_indices):
    try:
        data = np.load(npz_path)
        saved_test_indices = data['test_indices']
        if len(saved_test_indices) != len(test_indices) or not np.array_equal(saved_test_indices, test_indices):
            print(f"    Warning: Saved test_indices don't match current test_indices")
            print(f'      Saved: {len(saved_test_indices)} samples')
            print(f'      Current: {len(test_indices)} samples')
            saved_set = set(saved_test_indices)
            current_set = set(test_indices)
            matching_indices = sorted(list(saved_set & current_set))
            if len(matching_indices) == 0:
                print(f'    Error: No matching test indices found')
                return None
            if len(matching_indices) < len(test_indices):
                print(f'    Warning: Only {len(matching_indices)}/{len(test_indices)} indices match')
                print(f'    Using only matching indices')
            saved_idx_to_pos = {idx: pos for (pos, idx) in enumerate(saved_test_indices)}
            matching_positions = [saved_idx_to_pos[idx] for idx in matching_indices if idx in saved_idx_to_pos]
            predictions = data['unet_predictions'][matching_positions]
            unet_cmb_reconstructed = data['unet_cmb_reconstructed'][matching_positions]
            unet_fg_220_reconstruction = data['unet_fg_220_reconstruction'][matching_positions]
            return {'predictions': predictions, 'unet_cmb_reconstructed': unet_cmb_reconstructed, 'unet_fg_220_reconstruction': unet_fg_220_reconstruction, 'matching_indices': np.array(matching_indices)}
        else:
            return {'predictions': data['unet_predictions'], 'unet_cmb_reconstructed': data['unet_cmb_reconstructed'], 'unet_fg_220_reconstruction': data['unet_fg_220_reconstruction'], 'matching_indices': test_indices}
    except Exception as e:
        print(f'    Error loading test_results.npz: {e}')
        return None

def find_model_path(norm_suffix, channel_suffix, normalize, stats_file=STATS_FILE):
    model_dir = os.path.join(OUTPUT_DIR, f'{norm_suffix}_{channel_suffix}')
    stats = np.load(stats_file)
    train_size = len(stats['train_indices'])
    valid_size = len(stats['valid_indices'])
    file_id = f"{train_size}_{valid_size}_{('norm' if normalize else 'nonorm')}"
    best_model_path = os.path.join(model_dir, f'best_model_{file_id}.pt')
    if os.path.exists(best_model_path):
        return best_model_path
    elif os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if model_files:
                model_path = os.path.join(model_dir, model_files[0])
            print(f'  Warning: Using {model_files[0]} instead of best_model')
                return model_path
            else:
            raise FileNotFoundError(f'No model found in {model_dir}')
        else:
        raise FileNotFoundError(f'Model directory not found: {model_dir}')

def _save_loaded_test_tensors_multifreq(path, test_X, test_y, test_ilc_cmb, test_cmb_draw, test_indices_final, norm_stats, normalized, use_ilc, use_b_small, use_e_t, in_channels, channel_suffix):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    b220_path = os.path.join(B_SMALL_DIR, 'sim1-150_freq220_B_patches.npy')
    mm = np.load(b220_path, mmap_mode='r')
    test_b_all_220 = np.asarray(mm[test_indices_final])
    save_kw = {'test_X': test_X, 'test_y': test_y, 'test_ilc_cmb': test_ilc_cmb, 'test_cmb_draw': test_cmb_draw, 'test_b_all_scales_220': test_b_all_220, 'test_indices': test_indices_final, 'normalized': np.bool_(normalized), 'use_ilc': np.bool_(use_ilc), 'use_b_small': np.bool_(use_b_small), 'use_e_t': np.bool_(use_e_t), 'in_channels': np.int32(in_channels), 'channel_suffix': np.array(channel_suffix)}
    ilc_res_path = os.path.join(ILC_DIR, 'ilc_residuals_b.npy')
    if os.path.isfile(ilc_res_path):
        mmr = np.load(ilc_res_path, mmap_mode='r')
        save_kw['test_ilc_residuals'] = np.asarray(mmr[test_indices_final])
    if norm_stats is not None:
        for (k, v) in norm_stats.items():
            save_kw[f'norm_{k}'] = np.asarray(v)
    np.savez_compressed(path, **save_kw)
    print(f'\n[ok] Saved tensors for loaded test split (before model inference) to:\n  {path}')
    print(f'    Keys: {", ".join(sorted(save_kw.keys()))}')

def load_eval_from_npz_multifreq(npz_path, use_ilc, use_b_small, use_e_t, n_subsample, normalize_run):
    z = np.load(npz_path, allow_pickle=True)
    required = ('test_X', 'test_y', 'test_ilc_cmb', 'test_cmb_draw', 'test_indices', 'test_b_all_scales_220')
    for k in required:
        if k not in z.files:
            raise KeyError(f'{npz_path}: missing required array {k!r}')
    full_X = np.asarray(z['test_X'])
    if full_X.ndim != 4:
        raise ValueError(f'test_X must have shape (N,C,H,W); got {full_X.shape}')
    if use_e_t and full_X.shape[1] != 16:
        raise ValueError(f'With --add-e-t, --test-data-npz must be a 16-channel export; got C={full_X.shape[1]}')
    if use_ilc and use_b_small and use_e_t:
        need_max = 16
        sl = slice(0, 16)
    elif use_ilc and use_b_small:
        need_max = 8
        sl = slice(0, 8)
    elif use_ilc:
        need_max = 4
        sl = slice(0, 4)
    else:
        need_max = 8
        sl = slice(4, 8)
    if full_X.shape[1] < need_max:
        raise ValueError(f'npz test_X has {full_X.shape[1]} channels; this configuration needs at least {need_max}')
    stored_normalized = bool(np.asarray(z['normalized']).item()) if 'normalized' in z.files else False
    if normalize_run != stored_normalized:
        print(f"  WARNING: npz normalized={stored_normalized} but --normalize is {normalize_run}; expect mismatched scaling.")
    test_y = np.asarray(z['test_y'])
    test_ilc_cmb = np.asarray(z['test_ilc_cmb'])
    test_cmb_draw = np.asarray(z['test_cmb_draw'])
    test_indices_final = np.asarray(z['test_indices'])
    test_b_all = np.asarray(z['test_b_all_scales_220'])
    n_cap = None
    if n_subsample is not None:
        n_cap = min(int(n_subsample), full_X.shape[0])
        full_X = full_X[:n_cap]
        test_y = test_y[:n_cap]
        test_ilc_cmb = test_ilc_cmb[:n_cap]
        test_cmb_draw = test_cmb_draw[:n_cap]
        test_indices_final = test_indices_final[:n_cap]
        test_b_all = test_b_all[:n_cap]
        print(f'  Subsampled npz data to first {n_cap} patches')
    test_X = full_X[:, sl, :, :]
    norm_stats = None
    if normalize_run:
        raw = {k[5:]: np.asarray(z[k]) for k in z.files if k.startswith('norm_')}
        if not raw or 'target_mean' not in raw:
            raise ValueError('npz has no norm_* fields; use an export saved with --normalize, or run without --normalize')
        full_im = raw['input_mean']
        full_is = raw['input_std']
        if use_e_t and len(full_im) < 16:
            raise ValueError('npz norm_* input_mean must have 16 entries when using --add-e-t')
        im = []
        ist = []
        if use_ilc:
            im.extend(full_im[0:4])
            ist.extend(full_is[0:4])
        if use_b_small:
            im.extend(full_im[4:8])
            ist.extend(full_is[4:8])
        if use_e_t:
            im.extend(full_im[8:12])
            ist.extend(full_is[8:12])
            im.extend(full_im[12:16])
            ist.extend(full_is[12:16])
        input_mean = np.array(im)
        input_std = np.array(ist)
        norm_stats = {'input_mean': input_mean, 'input_std': input_std, 'target_mean': raw['target_mean'], 'target_std': raw['target_std']}
    bundle = {'test_b_all_scales_220': test_b_all}
    if 'test_ilc_residuals' in z.files:
        tir = np.asarray(z['test_ilc_residuals'])
        if n_cap is not None:
            tir = tir[:n_cap]
        bundle['test_ilc_residuals'] = tir
    print(f'\n[ok] Loaded test data from npz (channels fed to model: {test_X.shape[1]})')
    return (test_X, test_y, test_ilc_cmb, test_cmb_draw, test_indices_final, norm_stats, bundle)

def main():
    parser = argparse.ArgumentParser(description='Evaluate UNet predictions on test data')
    parser.add_argument('--normalize', action='store_true', help='Use normalized model (default: unnormalized)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results (default: same as model dir)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model checkpoint (default: auto-detect from output dir)')
    parser.add_argument('--ilc-only', action='store_true', help='Use ILC-only model (4 channels), skip B small-scale channels')
    parser.add_argument('--b-only', action='store_true', help='Use B-only model (4 channels), skip ILC foreground channels')
    parser.add_argument('--add-e-t', action='store_true', help='Add E and T mode channels (8 additional channels: channels 8-15)')
    parser.add_argument('--compare-all', action='store_true', help='Compare all three model configurations (ILC-only, ILC+B, ILC+B+E+T) side-by-side')
    parser.add_argument('--sample-indices', type=str, default=None, help='Comma-separated list of sample indices to visualize (e.g., "0,5,10,15"). Default: first 50 samples')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization (only save data)')
    parser.add_argument('--n-subsample', type=int, default=0, help='Limit test patches (default: 0 = use full test set from stats).')
    parser.add_argument('--save-loaded-test-data', type=str, default=None, metavar='PATH', help='Save test tensors right after loading (before inference) as a compressed .npz')
    parser.add_argument('--export-loaded-test-only', action='store_true', help='Load data, write --save-loaded-test-data, then exit (no checkpoint, no inference)')
    parser.add_argument('--test-data-npz', type=str, default=None, metavar='PATH', help='Load test tensors from a compressed .npz ')
    args = parser.parse_args()
    if args.export_loaded_test_only and (not args.save_loaded_test_data):
        parser.error('--export-loaded-test-only requires --save-loaded-test-data PATH')
    if args.test_data_npz and args.export_loaded_test_only:
        parser.error('--export-loaded-test-only cannot be used with --test-data-npz')
    if args.test_data_npz and args.save_loaded_test_data:
        parser.error('--save-loaded-test-data cannot be used with --test-data-npz')
    if args.compare_all:
        if args.test_data_npz:
            parser.error('--test-data-npz is not supported with --compare-all')
        if args.export_loaded_test_only:
            parser.error('--export-loaded-test-only is not supported with --compare-all')
        if args.save_loaded_test_data:
            parser.error('--save-loaded-test-data is not supported with --compare-all')
        if args.ilc_only or args.b_only or args.add_e_t:
            parser.error('--compare-all cannot be used with --ilc-only, --b-only, or --add-e-t')
    use_e_t = args.add_e_t
    if args.ilc_only:
        use_ilc = True
        use_b_small = False
        base_channels = 4
        channel_desc = 'ILC only (4 channels)'
    elif args.b_only:
        use_ilc = False
        use_b_small = True
        base_channels = 4
        channel_desc = 'B small only (4 channels)'
    else:
        use_ilc = True
        use_b_small = True
        base_channels = 8
        channel_desc = 'Full (8 channels)'
    if use_e_t:
        additional_channels = 8
        total_channels = base_channels + additional_channels
        channel_desc += f' + 8 (E + T) = {total_channels} total'
    else:
        total_channels = base_channels
    norm_suffix = 'normalized' if args.normalize else 'unnormalized'
    if use_ilc and use_b_small and use_e_t:
        channel_suffix = 'full_et'
    elif use_ilc and use_b_small:
        channel_suffix = 'full'
    elif use_ilc:
        channel_suffix = 'ilc_only'
    elif use_b_small:
        channel_suffix = 'b_only'
    in_channels = total_channels
    print(f'\nConfiguration:')
    print(f'  Normalization: {norm_suffix}')
    print(f'  Input channels: {channel_desc}')
    print(f'  Number of channels: {in_channels}')
    if args.n_subsample > 0:
        print(f'  Test subsample: first {args.n_subsample} patches')
    else:
        print(f'  Test subsample: full test set (from stats)')
    if args.compare_all:
        print('\n' + '=' * 80)
        print('COMPARE-ALL MODE: Evaluating all three model configurations')
        print('=' * 80)
        norm_suffix = 'normalized' if args.normalize else 'unnormalized'
        configs = [{'name': 'ILC-only', 'use_ilc': True, 'use_b_small': False, 'use_e_t': False, 'in_channels': 4, 'channel_suffix': 'ilc_only'}, {'name': 'ILC+B', 'use_ilc': True, 'use_b_small': True, 'use_e_t': False, 'in_channels': 8, 'channel_suffix': 'full'}, {'name': 'ILC+B+E+T', 'use_ilc': True, 'use_b_small': True, 'use_e_t': True, 'in_channels': 16, 'channel_suffix': 'full_et'}]
        print('\nLoading test indices (same for all configurations)...')
        stats_file = STATS_FILE_16CH
        if not os.path.exists(stats_file):
            stats_file = STATS_FILE_8CH
        stats = np.load(stats_file)
        test_indices = stats['test_indices']
        print(f'  Test indices: {len(test_indices)} samples')
        n_subsample = 500
        if len(test_indices) > n_subsample:
            test_indices = test_indices[:n_subsample]
            print(f'  Subsampled to first {n_subsample} patches')
        print('\nLoading shared test data (ILC CMB, pure CMB, foregrounds)...')
        ilc_cmb_file = f'{ILC_DIR}/ILC_cmb_b.npy'
        ilc_cmb_full = np.load(ilc_cmb_file)
        test_ilc_cmb = ilc_cmb_full[test_indices]
        print(f'  ILC CMB: {test_ilc_cmb.shape}')
        cmb_draw_file = f'{B_SMALL_DIR}/sim1-150_freq220_cmb_draw_b_all_scales.npy'
        cmb_draw_full = np.load(cmb_draw_file)
        test_cmb_draw = cmb_draw_full[test_indices]
        print(f'  Pure CMB: {test_cmb_draw.shape}')
        b_all_scales_file = f'{B_SMALL_DIR}/sim1-150_freq220_B_patches.npy'
        b_all_scales_full = np.load(b_all_scales_file)
        b_all_scales = b_all_scales_full[test_indices]
        print(f'  True foreground (b_all_scales): {b_all_scales.shape}')
        observed_cmb_220 = test_cmb_draw + b_all_scales
        print(f'  Observed CMB 220 GHz: {observed_cmb_220.shape}')
        config_results = []
        for config in configs:
            print('\n' + '-' * 80)
            print(f"Processing Configuration: {config['name']}")
            print('-' * 80)
            try:
                print(f"\nLoading test data inputs for {config['name']}...")
                (test_X, test_y, _, _, test_indices_from_load, norm_stats) = load_test_data(normalize=args.normalize, use_b_small=config['use_b_small'], use_ilc=config['use_ilc'], use_e_t=config['use_e_t'], n_subsample=len(test_indices))
                assert test_X.shape[0] == len(test_indices), f'Mismatch: test_X has {test_X.shape[0]} samples, expected {len(test_indices)}'
                print(f'  Loaded {test_X.shape[0]} samples with {test_X.shape[1]} input channels')
                print(f"\nChecking for saved test results (test_results.npz) for {config['name']}...")
                npz_path = find_test_results_npz(norm_suffix, config['channel_suffix'])
                if npz_path is not None:
                    print(f'  Found test_results.npz: {npz_path}')
                    print(f'  Loading saved predictions...')
                    saved_results = load_saved_test_results(npz_path, test_indices)
                    if saved_results is not None:
                        print(f'  [ok] Successfully loaded saved results from test_results.npz')
                        predictions = saved_results['predictions']
                        unet_cmb_reconstructed = saved_results['unet_cmb_reconstructed']
                        unet_fg_reconstruction_all = saved_results['unet_fg_220_reconstruction']
                        targets = None
                        print(f'  Filtering foreground reconstruction to large scales (ell < 200)...')
                        unet_fg_reconstruction = np.array([filter_patch(unet_fg_reconstruction_all[i], ell_cutoff=ELL_CUTOFF, filter_dir='large_scale', pix_size=PIX_SIZE) for i in tqdm(range(len(unet_fg_reconstruction_all)), desc=f"    Filtering {config['name']}", unit='patch')])
                        config_results.append({'name': config['name'], 'predictions': predictions, 'targets': targets, 'unet_cmb_reconstructed': unet_cmb_reconstructed, 'unet_fg_reconstruction': unet_fg_reconstruction, 'norm_stats': norm_stats, 'model': None, 'device': None, 'missing': False, 'loaded_from_npz': True})
                        print(f"  [ok] Completed {config['name']} (loaded from saved results)")
                        if 'matching_indices' in saved_results:
                            matching_indices = saved_results['matching_indices']
                            if len(matching_indices) < len(test_indices):
                                print(f'    Note: Using {len(matching_indices)} matching samples from saved results')
                        else:
                            matching_indices = test_indices
                        continue
                    else:
                        print(f'  WARNING: Could not load from test_results.npz (indices mismatch or error)')
                        print(f'  Falling back to model evaluation...')
                        saved_results = None
                else:
                    print(f'  No test_results.npz found, will try to load model...')
                    saved_results = None
                if saved_results is None:
                    print(f"\nFinding model for {config['name']}...")
                    try:
                        model_path = find_model_path(norm_suffix, config['channel_suffix'], args.normalize, stats_file)
                        print(f'  Model path: {model_path}')
                    except FileNotFoundError as e:
                        print(f"  WARNING: WARNING: Model not found for {config['name']}")
                        print(f'    Error: {e}')
                        print(f'    Skipping this configuration and using placeholder in visualizations')
                        n_samples = len(test_indices)
                        spatial_shape = b_all_scales.shape[1:]
                        config_results.append({'name': config['name'], 'predictions': None, 'targets': None, 'unet_cmb_reconstructed': None, 'unet_fg_reconstruction': None, 'norm_stats': None, 'model': None, 'device': None, 'missing': True})
                        continue
                    (model, device) = load_model(model_path, normalize=args.normalize, in_channels=config['in_channels'])
                    print(f"\nGenerating predictions for {config['name']}...")
                    (predictions, targets) = generate_predictions(model, test_X, test_y, device, batch_size=CONFIG['batch_size'])
                    if norm_stats is not None:
                        predictions = predictions * norm_stats['target_std'] + norm_stats['target_mean']
                        targets = targets * norm_stats['target_std'] + norm_stats['target_mean']
                    if predictions.ndim == 4 and predictions.shape[1] == 1:
                        predictions = predictions[:, 0, :, :]
                    if targets is not None and targets.ndim == 4 and (targets.shape[1] == 1):
                        targets = targets[:, 0, :, :]
                    unet_cmb_reconstructed = test_ilc_cmb - predictions
                    unet_fg_reconstruction_all = observed_cmb_220 - unet_cmb_reconstructed
                    print(f'  Filtering foreground reconstruction to large scales (ell < 200)...')
                    unet_fg_reconstruction = np.array([filter_patch(unet_fg_reconstruction_all[i], ell_cutoff=ELL_CUTOFF, filter_dir='large_scale', pix_size=PIX_SIZE) for i in tqdm(range(len(unet_fg_reconstruction_all)), desc=f"    Filtering {config['name']}", unit='patch')])
                    config_results.append({'name': config['name'], 'predictions': predictions, 'targets': targets, 'unet_cmb_reconstructed': unet_cmb_reconstructed, 'unet_fg_reconstruction': unet_fg_reconstruction, 'norm_stats': norm_stats, 'model': model, 'device': device, 'missing': False, 'loaded_from_npz': False})
                    print(f"  [ok] Completed {config['name']}")
            except Exception as e:
                print(f"  WARNING: ERROR: Failed to process {config['name']}: {e}")
                print(f'    Skipping this configuration and using placeholder in visualizations')
                n_samples = len(test_indices)
                spatial_shape = b_all_scales.shape[1:]
                config_results.append({'name': config['name'], 'predictions': None, 'targets': None, 'unet_cmb_reconstructed': None, 'unet_fg_reconstruction': None, 'norm_stats': None, 'model': None, 'device': None, 'missing': True})
                continue
        print('\n' + '=' * 80)
        print('All three configurations loaded and evaluated')
        print('=' * 80)
        print(f'  Test samples: {len(test_indices)}')
        print(f'  Spatial dimensions: {b_all_scales.shape[1:]}')
        print('\nFiltering true foreground (b_all_scales) to large scales (ell < 200)...')
        b_all_scales_large = np.array([filter_patch(b_all_scales[i], ell_cutoff=ELL_CUTOFF, filter_dir='large_scale', pix_size=PIX_SIZE) for i in tqdm(range(len(b_all_scales)), desc='  Filtering true FG', unit='patch')])
        print(f'  Filtered true foreground shape: {b_all_scales_large.shape}')
        compare_all_data = {'test_indices': test_indices, 'test_ilc_cmb': test_ilc_cmb, 'test_cmb_draw': test_cmb_draw, 'b_all_scales': b_all_scales_large, 'observed_cmb_220': observed_cmb_220, 'config_results': config_results, 'norm_suffix': norm_suffix}
        print('\n' + '=' * 80)
        print('PHASE 3: COMPUTING METRICS AND CREATING COMPARISON VISUALIZATIONS')
        print('=' * 80)
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(OUTPUT_DIR, f'comparison_all_configs_{norm_suffix}')
        os.makedirs(output_dir, exist_ok=True)
        print(f'\nOutput directory: {output_dir}')
        n_samples = len(test_indices)
        print('\n' + '-' * 80)
        print('Computing metrics for all three configurations (large-scale, ell < 200)...')
        print('-' * 80)
        all_mse_arrays = []
        all_corr_arrays = []
        all_cmb_var_arrays = []
        all_cross_spectra_arrays = []
        all_mean_cross_spectra = []
        all_percentile_lower = []
        all_percentile_upper = []
        all_ell_arrays = []
        print('\nComputing true CMB variance (baseline) for all test samples...')
        true_cmb_var_list = []
        for patch_idx in range(n_samples):
            true_cmb_patch = test_cmb_draw[patch_idx]
            true_cmb_var = np.var(true_cmb_patch)
            true_cmb_var_list.append(true_cmb_var)
        true_cmb_var_array = np.array(true_cmb_var_list)
        print(f'  True CMB Variance - Mean: {np.mean(true_cmb_var_array):.6e}, Std: {np.std(true_cmb_var_array):.6e}')
        for (config_idx, config) in enumerate(configs):
            config_name = config['name']
            print(f'\nProcessing metrics for {config_name}...')
            if config_idx >= len(config_results) or config_results[config_idx].get('missing', False):
                print(f'  WARNING: Skipping metrics for {config_name} (model not found)')
                all_mse_arrays.append(None)
                all_corr_arrays.append(None)
                all_cmb_var_arrays.append(None)
                all_cross_spectra_arrays.append(None)
                all_mean_cross_spectra.append(None)
                all_percentile_lower.append(None)
                all_percentile_upper.append(None)
                all_ell_arrays.append(None)
                continue
            fg_reconstruction = config_results[config_idx]['unet_fg_reconstruction']
            cmb_reconstruction = config_results[config_idx]['unet_cmb_reconstructed']
            mse_list = []
            corr_list = []
            cmb_var_list = []
            for patch_idx in range(n_samples):
                true_fg_patch = b_all_scales[patch_idx]
                fg_recon_patch = fg_reconstruction[patch_idx]
                mse = np.mean((true_fg_patch - fg_recon_patch) ** 2)
                mse_list.append(mse)
                true_flat = true_fg_patch.flatten()
                recon_flat = fg_recon_patch.flatten()
                corr = np.corrcoef(true_flat, recon_flat)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                corr_list.append(corr)
                cmb_patch = cmb_reconstruction[patch_idx]
                cmb_var = np.var(cmb_patch)
                cmb_var_list.append(cmb_var)
            mse_array = np.array(mse_list)
            corr_array = np.array(corr_list)
            cmb_var_array = np.array(cmb_var_list)
            all_mse_arrays.append(mse_array)
            all_corr_arrays.append(corr_array)
            all_cmb_var_arrays.append(cmb_var_array)
            print(f'  MSE - Mean: {np.mean(mse_array):.6e}, Std: {np.std(mse_array):.6e}')
            print(f'  Correlation - Mean: {np.mean(corr_array):.4f}, Std: {np.std(corr_array):.4f}')
            print(f'  CMB Variance - Mean: {np.mean(cmb_var_array):.6e}, Std: {np.std(cmb_var_array):.6e}')
            if calculate_2d_spectrum is not None:
                print(f'  Computing cross-power spectra for {config_name}...')
                cross_spectra_list = []
                ell_array_ref = None
                for patch_idx in tqdm(range(n_samples), desc=f'  {config_name} cross-spectra', unit='patch'):
                    try:
                        true_fg_patch = b_all_scales[patch_idx]
                        fg_recon_patch = fg_reconstruction[patch_idx]
                        (ell_array, cross_ps) = compute_normalized_cross_spectrum(true_fg_patch, fg_recon_patch, delta_ell=DELTA_ELL, ell_max=ELL_CUTOFF, pix_size=PIX_SIZE, N=PATCH_SIZE)
                        cross_spectra_list.append(cross_ps)
                        if ell_array_ref is None:
                            ell_array_ref = ell_array.copy()
                    except Exception as e:
                        print(f'    Warning: Error computing cross-spectrum for patch {patch_idx}: {e}')
                        if ell_array_ref is not None:
                            nan_array = np.full_like(ell_array_ref, np.nan)
                            cross_spectra_list.append(nan_array)
                        else:
                            continue
                cross_spectra_array = np.array(cross_spectra_list)
                mean_cross_ps = np.nanmean(cross_spectra_array, axis=0)
                percentile_lower = np.nanpercentile(cross_spectra_array, 16, axis=0)
                percentile_upper = np.nanpercentile(cross_spectra_array, 84, axis=0)
                all_cross_spectra_arrays.append(cross_spectra_array)
                all_mean_cross_spectra.append(mean_cross_ps)
                all_percentile_lower.append(percentile_lower)
                all_percentile_upper.append(percentile_upper)
                all_ell_arrays.append(ell_array_ref)
                mean_cross_corr = np.nanmean(mean_cross_ps)
                print(f'  Mean cross-spectrum correlation: {mean_cross_corr:.4f}')
            else:
                print(f'  Skipping cross-spectra (calculate_2d_spectrum not available)')
                all_cross_spectra_arrays.append(None)
                all_mean_cross_spectra.append(None)
                all_percentile_lower.append(None)
                all_percentile_upper.append(None)
                all_ell_arrays.append(None)
        print('\n' + '-' * 80)
        print('Saving comparison statistics...')
        print('-' * 80)
        stats_file = os.path.join(output_dir, 'comparison_stats_large_scale.txt')
        with open(stats_file, 'w') as f:
            f.write('# Comparison Statistics for All Three Configurations (Large-Scale, ell < 200)\n')
            f.write('# Format: Configuration, MSE_mean, MSE_std, Correlation_mean, Correlation_std, Mean_Cross_Spectrum_Corr, CMB_Variance_mean\n')
            f.write(f'# Number of test samples: {n_samples}\n')
            f.write(f'# Analysis: Large-scale foregrounds only (ell < 200)\n')
            f.write('#\n')
            true_cmb_var_mean = np.mean(true_cmb_var_array)
            f.write(f'# Baseline: True CMB Variance - Mean: {true_cmb_var_mean:.6e}\n')
            f.write('#\n')
            for (config_idx, config) in enumerate(configs):
                config_name = config['name']
                mse_array = all_mse_arrays[config_idx]
                corr_array = all_corr_arrays[config_idx]
                cmb_var_array = all_cmb_var_arrays[config_idx]
                if mse_array is None or corr_array is None or cmb_var_array is None:
                    f.write(f'{config_name}  MISSING_MODEL  (model file not found)\n')
                    continue
                mse_mean = np.mean(mse_array)
                mse_std = np.std(mse_array)
                corr_mean = np.mean(corr_array)
                corr_std = np.std(corr_array)
                cmb_var_mean = np.mean(cmb_var_array)
                if all_mean_cross_spectra[config_idx] is not None:
                    mean_cross_corr = np.nanmean(all_mean_cross_spectra[config_idx])
                else:
                    mean_cross_corr = np.nan
                f.write(f'{config_name}  {mse_mean:.6e}  {mse_std:.6e}  {corr_mean:.6f}  {corr_std:.6f}  {mean_cross_corr:.6f}  {cmb_var_mean:.6e}\n')
        print(f'  Saved statistics to: {stats_file}')
        if not args.no_viz:
            print('\n' + '-' * 80)
            print('Creating comparison visualizations...')
            print('-' * 80)
            if args.sample_indices:
                sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
                max_idx = n_samples - 1
                sample_indices = [idx for idx in sample_indices if 0 <= idx <= max_idx]
                if len(sample_indices) == 0:
                    print(f'  Warning: No valid sample indices. Using first 10 samples.')
                    sample_indices = list(range(min(10, n_samples)))
            else:
                sample_indices = list(range(min(10, n_samples)))
            original_sample_indices = [test_indices[idx] for idx in sample_indices]
            print(f'  Visualizing samples (array positions): {sample_indices}')
            print(f'  Original dataset indices: {original_sample_indices}')
            fg_reconstructions = []
            for i in range(3):
                if i < len(config_results) and (not config_results[i].get('missing', False)):
                    fg_reconstructions.append(config_results[i]['unet_fg_reconstruction'])
                else:
                    fg_reconstructions.append(None)
            config_names = [configs[i]['name'] for i in range(3)]
            ell_array_viz = all_ell_arrays[0] if all_ell_arrays[0] is not None else None
            visualize_compare_all_fg_reconstructions(b_all_scales, fg_reconstructions, config_names, sample_indices, output_dir=output_dir, ell_arrays=ell_array_viz, cross_spectra_arrays=all_cross_spectra_arrays, mean_cross_spectra=all_mean_cross_spectra, percentile_lower=all_percentile_lower, percentile_upper=all_percentile_upper, original_sample_indices=original_sample_indices, cmb_var_arrays=all_cmb_var_arrays)
        print('\n' + '=' * 80)
        print('COMPARISON SUMMARY')
        print('=' * 80)
        print(f'\nBaseline (True CMB):')
        print(f'  Mean CMB Variance: {np.mean(true_cmb_var_array):.6e}')
        for (config_idx, config) in enumerate(configs):
            config_name = config['name']
            if all_mse_arrays[config_idx] is None or all_corr_arrays[config_idx] is None or all_cmb_var_arrays[config_idx] is None:
                print(f'\n{config_name}:')
                print(f'  WARNING: Model not found - skipped')
                continue
            mse_mean = np.mean(all_mse_arrays[config_idx])
            corr_mean = np.mean(all_corr_arrays[config_idx])
            cmb_var_mean = np.mean(all_cmb_var_arrays[config_idx])
            if all_mean_cross_spectra[config_idx] is not None:
                mean_cross_corr = np.nanmean(all_mean_cross_spectra[config_idx])
            else:
                mean_cross_corr = np.nan
            print(f'\n{config_name}:')
            print(f'  MSE: {mse_mean:.6e}')
            print(f'  Spatial Correlation: {corr_mean:.4f}')
            print(f'  Mean Cross-Spectrum Correlation: {mean_cross_corr:.4f}')
            print(f'  Mean CMB Variance: {cmb_var_mean:.6e}')
        print(f'\n[ok] All comparison results saved to: {output_dir}')
        return
    n_subsample_arg = args.n_subsample if args.n_subsample > 0 else None
    hybrid_npz_bundle = None
    if args.test_data_npz:
        (test_X, test_y, test_ilc_cmb, test_cmb_draw, test_indices_final, norm_stats, hybrid_npz_bundle) = load_eval_from_npz_multifreq(args.test_data_npz, use_ilc=use_ilc, use_b_small=use_b_small, use_e_t=use_e_t, n_subsample=n_subsample_arg, normalize_run=args.normalize)
    else:
        (test_X, test_y, test_ilc_cmb, test_cmb_draw, test_indices_final, norm_stats) = load_test_data(normalize=args.normalize, use_b_small=use_b_small, use_ilc=use_ilc, use_e_t=use_e_t, n_subsample=n_subsample_arg)
    if args.save_loaded_test_data:
        _save_loaded_test_tensors_multifreq(args.save_loaded_test_data, test_X, test_y, test_ilc_cmb, test_cmb_draw, test_indices_final, norm_stats, args.normalize, use_ilc, use_b_small, use_e_t, in_channels, channel_suffix)
    if args.export_loaded_test_only:
        print('\n[ok] --export-loaded-test-only: skipping model load and evaluation.')
        return
    if args.model_path:
        model_path = args.model_path
    else:
        model_dir = os.path.join(OUTPUT_DIR, f'{norm_suffix}_{channel_suffix}')
        stats = np.load(STATS_FILE_16CH if use_e_t else STATS_FILE_8CH)
        train_size = len(stats['train_indices'])
        valid_size = len(stats['valid_indices'])
        file_id = f"{train_size}_{valid_size}_{('norm' if args.normalize else 'nonorm')}"
        best_model_path = os.path.join(model_dir, f'best_model_{file_id}.pt')
        if os.path.exists(best_model_path):
            model_path = best_model_path
        elif os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])
                print(f'\nWarning: Using {model_files[0]} instead of best_model')
                else:
                raise FileNotFoundError(f'No model found in {model_dir}')
            else:
            raise FileNotFoundError(f'Model directory not found: {model_dir}')
    (model, device) = load_model(model_path, normalize=args.normalize, in_channels=in_channels)
    (predictions, targets) = generate_predictions(model, test_X, test_y, device, batch_size=CONFIG['batch_size'])
    if norm_stats is not None:
        print('\nDenormalizing predictions and targets...')
        predictions = predictions * norm_stats['target_std'] + norm_stats['target_mean']
        targets = targets * norm_stats['target_std'] + norm_stats['target_mean']
    if predictions.ndim == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0, :, :]
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0, :, :]
    ilc_residuals_file = f'{ILC_DIR}/ilc_residuals_b.npy'
    print(f'\n  DIAGNOSTIC: Loading true ILC residuals...')
    if hybrid_npz_bundle is not None and 'test_ilc_residuals' in hybrid_npz_bundle:
        test_ilc_residuals = hybrid_npz_bundle['test_ilc_residuals']
        print('    (from --test-data-npz)')
        print(f'    Verifying: ilc_residuals = ILC_cmb - cmb_draw...')
        computed_residuals = test_ilc_cmb - test_cmb_draw
        residuals_diff = np.abs(test_ilc_residuals - computed_residuals).max()
        pred_flat = predictions.flatten()
        true_residuals_flat = test_ilc_residuals.flatten()
        pred_vs_residuals_corr = np.corrcoef(pred_flat, true_residuals_flat)[0, 1]
        
        pred_vs_residuals_mse = np.mean((pred_flat - true_residuals_flat) ** 2)
        pred_vs_neg_residuals_mse = np.mean((pred_flat + true_residuals_flat) ** 2)
        print(f'    MSE(predictions, ilc_residuals): {pred_vs_residuals_mse:.6e}')
        print(f'    MSE(predictions, -ilc_residuals): {pred_vs_neg_residuals_mse:.6e}')
        
            
    elif os.path.exists(ilc_residuals_file):
        print(f'    Loading from {ilc_residuals_file}...')
        ilc_residuals_full = np.load(ilc_residuals_file)
        test_ilc_residuals = ilc_residuals_full[test_indices_final]
        print(f'    Verifying: ilc_residuals = ILC_cmb - cmb_draw...')
        computed_residuals = test_ilc_cmb - test_cmb_draw
        residuals_diff = np.abs(test_ilc_residuals - computed_residuals).max()
        
        pred_flat = predictions.flatten()
        true_residuals_flat = test_ilc_residuals.flatten()
        pred_vs_residuals_corr = np.corrcoef(pred_flat, true_residuals_flat)[0, 1]
        
        pred_vs_residuals_mse = np.mean((pred_flat - true_residuals_flat) ** 2)
        pred_vs_neg_residuals_mse = np.mean((pred_flat + true_residuals_flat) ** 2)

    unet_cmb_reconstructed = test_ilc_cmb - predictions
    print(f'  unet_cmb_reconstructed shape: {unet_cmb_reconstructed.shape}')
    if test_ilc_residuals is not None:
        print(f'\n  DIAGNOSTIC: Verifying CMB reconstruction formula...')
        unet_cmb_flat = unet_cmb_reconstructed.flatten()
        cmb_draw_flat = test_cmb_draw.flatten()
        unet_vs_cmb_corr = np.corrcoef(unet_cmb_flat, cmb_draw_flat)[0, 1]
        print(f'    unet_cmb_reconstructed vs cmb_draw correlation: {unet_vs_cmb_corr:.6f}')
        unet_cmb_reconstructed_alt = test_ilc_cmb + predictions
        unet_cmb_alt_flat = unet_cmb_reconstructed_alt.flatten()
        unet_vs_cmb_corr_alt = np.corrcoef(unet_cmb_alt_flat, cmb_draw_flat)[0, 1]
        print(f'    (ILC_cmb + UNet_pred) vs cmb_draw correlation: {unet_vs_cmb_corr_alt:.6f}')
        
        unet_cmb_from_true_residuals = test_ilc_cmb - test_ilc_residuals
        unet_cmb_from_true_flat = unet_cmb_from_true_residuals.flatten()
        true_residuals_corr = np.corrcoef(unet_cmb_from_true_flat, cmb_draw_flat)[0, 1]
    print('\n' + '=' * 80)
    print('COMPUTING FOREGROUND RECONSTRUCTIONS AT 220 GHz')
    print('=' * 80)
    b_all_scales_file = f'{B_SMALL_DIR}/sim1-150_freq220_B_patches.npy'
    print(f'\nLoading b_all_scales foreground at 220 GHz: {b_all_scales_file}')
    if hybrid_npz_bundle is not None:
        b_all_scales = hybrid_npz_bundle['test_b_all_scales_220']
        print(f'  Using patches from --test-data-npz, shape: {b_all_scales.shape}')
        else:
    if not os.path.exists(b_all_scales_file):
            raise FileNotFoundError(f'B all-scales file not found: {b_all_scales_file}')
    b_all_scales_full = np.load(b_all_scales_file)
        print(f'  Full dataset shape: {b_all_scales_full.shape}')
        b_all_scales = b_all_scales_full[test_indices_final]
    print(f'  Test set shape (using test_indices): {b_all_scales.shape}')
    assert b_all_scales.shape[0] == len(test_indices_final), f'Mismatch: b_all_scales has {b_all_scales.shape[0]} samples, expected {len(test_indices_final)}'
    assert b_all_scales.shape[1:] == test_cmb_draw.shape[1:], f'Mismatch: b_all_scales has spatial shape {b_all_scales.shape[1:]}, expected {test_cmb_draw.shape[1:]}'
    observed_cmb_220 = test_cmb_draw + b_all_scales
    unet_fg_220_reconstruction = observed_cmb_220 - unet_cmb_reconstructed
    ILC_fg_reconstruction = observed_cmb_220 - test_ilc_cmb
    ilc_residuals_verify = test_cmb_draw - test_ilc_cmb
    predictions_2d = predictions.squeeze() if predictions.ndim > 2 else predictions
    if predictions_2d.ndim == 3:
        pass
    else:
        raise ValueError(f'Unexpected predictions shape: {predictions.shape}')
    unet_fg_expected = b_all_scales + predictions_2d - ilc_residuals_verify
    diff = np.abs(unet_fg_220_reconstruction - unet_fg_expected).max()
    print('\n' + '=' * 80)
    print('COMPUTING CROSS-POWER SPECTRA')
    print('=' * 80)
    n_subsample = len(test_cmb_draw)
    print(f'\nComputing cross-power spectra for {n_subsample} subsampled patches...')
    print(f'  Parameters: delta_ell={DELTA_ELL}, ell_max={ELL_CUTOFF}, pix_size={PIX_SIZE}, N={PATCH_SIZE}')
    unet_cross_spectra_list = []
    ilc_cross_spectra_list = []
    ell_array_ref = None
    for patch_idx in tqdm(range(n_subsample), desc='Cross-spectra', unit='patch'):
        try:
            pure_cmb_patch = test_cmb_draw[patch_idx]
            ilc_cmb_patch = test_ilc_cmb[patch_idx]
            unet_cmb_patch = unet_cmb_reconstructed[patch_idx]
            (ell_array, unet_cross_ps) = compute_normalized_cross_spectrum(pure_cmb_patch, unet_cmb_patch)
            unet_cross_spectra_list.append(unet_cross_ps)
            (_, ilc_cross_ps) = compute_normalized_cross_spectrum(pure_cmb_patch, ilc_cmb_patch)
            ilc_cross_spectra_list.append(ilc_cross_ps)
            if ell_array_ref is None:
                ell_array_ref = ell_array.copy()
        except Exception as e:
            print(f'    Warning: Error computing cross-spectrum for patch {patch_idx}: {e}')
            continue
    unet_cross_spectra_array = np.array(unet_cross_spectra_list)
    ilc_cross_spectra_array = np.array(ilc_cross_spectra_list)
    print('\n' + '=' * 80)
    print('COMPUTING STATISTICS AND SAVING TO FILES')
    print('=' * 80)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(OUTPUT_DIR, f'{norm_suffix}_{channel_suffix}')
    os.makedirs(output_dir, exist_ok=True)
    print('\nComputing mean and percentiles across patches...')
    mean_unet_cross_ps = np.nanmean(unet_cross_spectra_array, axis=0)
    percentile_lower_unet = np.nanpercentile(unet_cross_spectra_array, 16, axis=0)
    percentile_upper_unet = np.nanpercentile(unet_cross_spectra_array, 84, axis=0)
    mean_ilc_cross_ps = np.nanmean(ilc_cross_spectra_array, axis=0)
    percentile_lower_ilc = np.nanpercentile(ilc_cross_spectra_array, 16, axis=0)
    percentile_upper_ilc = np.nanpercentile(ilc_cross_spectra_array, 84, axis=0)
    print(f'  UNet cross-spectrum - Mean shape: {mean_unet_cross_ps.shape}, Percentile shape: {percentile_lower_unet.shape}')
    print(f'  ILC cross-spectrum - Mean shape: {mean_ilc_cross_ps.shape}, Percentile shape: {percentile_lower_ilc.shape}')
    unet_stats_file = os.path.join(output_dir, 'unet_cross_spectrum_stats.txt')
    print(f'\nSaving UNet cross-spectrum statistics to: {unet_stats_file}')
    with open(unet_stats_file, 'w') as f:
        f.write('# UNet Cross-Power Spectrum Statistics\n')
        f.write('# Format: ell, mean, percentile_16, percentile_84\n')
        f.write('# Computed from normalized cross-power spectra: primordial_b_modes  x  unet_cmb_reconstruction\n')
        f.write(f'# Number of patches: {len(unet_cross_spectra_list)}\n')
        f.write(f'# Parameters: delta_ell={DELTA_ELL}, ell_max={ELL_CUTOFF}, pix_size={PIX_SIZE}, N={PATCH_SIZE}\n')
        f.write('#\n')
        for i in range(len(ell_array_ref)):
            f.write(f'{ell_array_ref[i]:.2f}  {mean_unet_cross_ps[i]:.6e}  {percentile_lower_unet[i]:.6e}  {percentile_upper_unet[i]:.6e}\n')
    print('  [ok] Saved!')
    ilc_stats_file = os.path.join(output_dir, 'ilc_cross_spectrum_stats.txt')
    print(f'\nSaving ILC cross-spectrum statistics to: {ilc_stats_file}')
    with open(ilc_stats_file, 'w') as f:
        f.write('# ILC Cross-Power Spectrum Statistics\n')
        f.write('# Format: ell, mean, percentile_16, percentile_84\n')
        f.write('# Computed from normalized cross-power spectra: primordial_b_modes  x  ilc_cmb\n')
        f.write(f'# Number of patches: {len(ilc_cross_spectra_list)}\n')
        f.write(f'# Parameters: delta_ell={DELTA_ELL}, ell_max={ELL_CUTOFF}, pix_size={PIX_SIZE}, N={PATCH_SIZE}\n')
        f.write('#\n')
        for i in range(len(ell_array_ref)):
            f.write(f'{ell_array_ref[i]:.2f}  {mean_ilc_cross_ps[i]:.6e}  {percentile_lower_ilc[i]:.6e}  {percentile_upper_ilc[i]:.6e}\n')
    print('  [ok] Saved!')
    print('\n' + '=' * 80)
    print('COMPUTING NULL HYPOTHESIS TEST (CROSS-PATCH CORRELATIONS)')
    print('=' * 80)
    null_cross_spectra_list = []
    print(f'\nComputing null cross-spectra for {n_subsample} patches...')
    print(f'  For each patch: computing cross-spectra with all OTHER primordial CMB patches')
    for patch_idx in tqdm(range(n_subsample), desc='Null cross-spectra', unit='patch'):
        try:
            current_unet_cmb = unet_cmb_reconstructed[patch_idx]
            patch_null_cross_spectra = []
            for other_idx in range(n_subsample):
                if other_idx == patch_idx:
                    continue
                try:
                    other_pure_cmb = test_cmb_draw[other_idx]
                    (_, null_cross_ps) = compute_normalized_cross_spectrum(current_unet_cmb, other_pure_cmb)
                    patch_null_cross_spectra.append(null_cross_ps)
                except Exception as e:
                    continue
            if len(patch_null_cross_spectra) > 0:
                patch_null_cross_spectra_array = np.array(patch_null_cross_spectra)
                mean_null_cross_ps = np.nanmean(patch_null_cross_spectra_array, axis=0)
                null_cross_spectra_list.append(mean_null_cross_ps)
            else:
                null_cross_spectra_list.append(np.zeros_like(ell_array_ref))
        except Exception as e:
            print(f'    Warning: Error computing null cross-spectrum for patch {patch_idx}: {e}')
            null_cross_spectra_list.append(np.zeros_like(ell_array_ref))
            continue
    null_cross_spectra_array = np.array(null_cross_spectra_list)
    mean_null_cross_ps_overall = np.nanmean(null_cross_spectra_array, axis=0)
    print(f'\n  [ok] Computed null cross-spectra for {len(null_cross_spectra_list)} patches')
    print(f'    Null cross-spectra shape: {null_cross_spectra_array.shape}')
    print(f'    Mean null cross-spectrum shape: {mean_null_cross_ps_overall.shape}')
    print(f'    Mean null cross-spectrum (overall): {np.mean(mean_null_cross_ps_overall):.6e} (should be close to 0)')
    print('\n' + '=' * 80)
    print('COMPUTING SPATIAL PEARSON CORRELATIONS')
    print('=' * 80)
    unet_spatial_correlations = []
    ilc_spatial_correlations = []
    print(f'\nComputing spatial correlations for {n_subsample} patches...')
    for patch_idx in tqdm(range(n_subsample), desc='Spatial correlations', unit='patch'):
        try:
            pure_cmb_patch = test_cmb_draw[patch_idx]
            ilc_cmb_patch = test_ilc_cmb[patch_idx]
            unet_cmb_patch = unet_cmb_reconstructed[patch_idx]
            pure_flat = pure_cmb_patch.flatten()
            ilc_flat = ilc_cmb_patch.flatten()
            unet_flat = unet_cmb_patch.flatten()
            corr_matrix_unet = np.corrcoef(pure_flat, unet_flat)
            unet_corr = corr_matrix_unet[0, 1]
            if np.isnan(unet_corr):
                unet_corr = 0.0
            unet_spatial_correlations.append(unet_corr)
            corr_matrix_ilc = np.corrcoef(pure_flat, ilc_flat)
            ilc_corr = corr_matrix_ilc[0, 1]
            if np.isnan(ilc_corr):
                ilc_corr = 0.0
            ilc_spatial_correlations.append(ilc_corr)
        except Exception as e:
            print(f'    Warning: Error computing spatial correlation for patch {patch_idx}: {e}')
            unet_spatial_correlations.append(0.0)
            ilc_spatial_correlations.append(0.0)
            continue
    unet_spatial_correlations = np.array(unet_spatial_correlations)
    ilc_spatial_correlations = np.array(ilc_spatial_correlations)
    print(f'    UNet spatial correlations - Mean: {np.mean(unet_spatial_correlations):.4f}, Std: {np.std(unet_spatial_correlations):.4f}')
    print(f'    ILC spatial correlations - Mean: {np.mean(ilc_spatial_correlations):.4f}, Std: {np.std(ilc_spatial_correlations):.4f}')
    print('\n' + '=' * 80)
    print('COMPUTING NULL CORRELATION COEFFICIENTS')
    print('=' * 80)
    null_correlations_list = []
    print(f'\nComputing null correlations for {n_subsample} patches...')
    print(f'  For each patch: correlating with {min(100, n_subsample - 1)} random primordial CMB patches (excluding its own)')
    for patch_idx in tqdm(range(n_subsample), desc='Null correlations', unit='patch'):
        try:
            current_unet_cmb = unet_cmb_reconstructed[patch_idx]
            current_unet_flat = current_unet_cmb.flatten()
            available_indices = [j for j in range(n_subsample) if j != patch_idx]
            n_samples = min(100, len(available_indices))
            random_indices = np.random.choice(available_indices, size=n_samples, replace=False)
            for j in random_indices:
                try:
                    other_pure_cmb = test_cmb_draw[j]
                    other_pure_flat = other_pure_cmb.flatten()
                    corr_matrix = np.corrcoef(current_unet_flat, other_pure_flat)
                    null_corr = corr_matrix[0, 1]
                    if not np.isnan(null_corr):
                        null_correlations_list.append(null_corr)
                except Exception as e:
                    continue
        except Exception as e:
            print(f'    Warning: Error computing null correlation for patch {patch_idx}: {e}')
            continue
    null_correlations_array = np.array(null_correlations_list)
    mean_null_correlation_overall = np.mean(null_correlations_array) if len(null_correlations_array) > 0 else None
    print(f'    Total null correlations: {len(null_correlations_array)}')
    if mean_null_correlation_overall is not None:
        print(f'    Mean null correlation (overall): {mean_null_correlation_overall:.6f} (should be close to 0)')
        print(f'    Std of null correlations: {np.std(null_correlations_array):.6f}')
    else:
        print(f'    Warning: No valid null correlations computed')
    print('\n' + '=' * 80)
    print('COMPUTING MSE FOR EACH PATCH')
    print('=' * 80)

    def compute_patch_mse(patch1, patch2):
        mse_val = np.mean((patch1 - patch2) ** 2)
        if np.isnan(mse_val) or np.isinf(mse_val):
            return None
        return mse_val
    ilc_cmb_mse_list = []
    unet_recon_mse_list = []
    unet_cmb_var_list = []
    true_cmb_var_list = []
    print(f'\nComputing MSEs for {n_subsample} patches...')
    for patch_idx in tqdm(range(n_subsample), desc='MSE computation', unit='patch'):
        try:
            pure_cmb_patch = test_cmb_draw[patch_idx]
            ilc_cmb_patch = test_ilc_cmb[patch_idx]
            unet_cmb_patch = unet_cmb_reconstructed[patch_idx]
            ilc_mse = compute_patch_mse(pure_cmb_patch, ilc_cmb_patch)
            if ilc_mse is not None:
                ilc_cmb_mse_list.append(ilc_mse)
            else:
                ilc_cmb_mse_list.append(0.0)
            unet_mse = compute_patch_mse(pure_cmb_patch, unet_cmb_patch)
            if unet_mse is not None:
                unet_recon_mse_list.append(unet_mse)
            else:
                unet_recon_mse_list.append(0.0)
            unet_cmb_var = np.var(unet_cmb_patch)
            unet_cmb_var_list.append(unet_cmb_var)
            true_cmb_var = np.var(pure_cmb_patch)
            true_cmb_var_list.append(true_cmb_var)
        except Exception as e:
            print(f'    Warning: Error computing MSE for patch {patch_idx}: {e}')
            ilc_cmb_mse_list.append(0.0)
            unet_recon_mse_list.append(0.0)
            unet_cmb_var_list.append(0.0)
            true_cmb_var_list.append(0.0)
            continue
    ilc_cmb_mse_array = np.array(ilc_cmb_mse_list)
    unet_recon_mse_array = np.array(unet_recon_mse_list)
    unet_cmb_var_array = np.array(unet_cmb_var_list)
    true_cmb_var_array = np.array(true_cmb_var_list)
    print(f'    ILC CMB MSE - Mean: {np.mean(ilc_cmb_mse_array):.6e}, Std: {np.std(ilc_cmb_mse_array):.6e}')
    print(f'    UNet recon MSE - Mean: {np.mean(unet_recon_mse_array):.6e}, Std: {np.std(unet_recon_mse_array):.6e}')
    print(f'    Improvement ratio (ILC/UNet): {np.mean(ilc_cmb_mse_array) / np.mean(unet_recon_mse_array):.4f}x')
    print(f'    True CMB Variance (baseline) - Mean: {np.mean(true_cmb_var_array):.6e}, Std: {np.std(true_cmb_var_array):.6e}')
    print(f'    UNet CMB Variance - Mean: {np.mean(unet_cmb_var_array):.6e}, Std: {np.std(unet_cmb_var_array):.6e}')
    mse_stats_file = os.path.join(output_dir, 'mse_stats.txt')
    print(f'\nSaving MSE statistics to: {mse_stats_file}')
    with open(mse_stats_file, 'w') as f:
        f.write('# MSE Statistics for Test Patches\n')
        f.write('# Format: patch_idx, ilc_cmb_mse, unet_recon_mse, true_cmb_variance, unet_cmb_variance\n')
        f.write(f'# Number of patches: {len(ilc_cmb_mse_list)}\n')
        f.write('#\n')
        for i in range(len(ilc_cmb_mse_list)):
            f.write(f'{i}  {ilc_cmb_mse_array[i]:.6e}  {unet_recon_mse_array[i]:.6e}  {true_cmb_var_array[i]:.6e}  {unet_cmb_var_array[i]:.6e}\n')
    print('  [ok] Saved!')
    print('\n' + '=' * 80)
    print('COMPUTING FOREGROUND RECONSTRUCTION ANALYSIS')
    print('=' * 80)
    unet_fg_cross_spectra_list = []
    ell_array_fg_ref = None
    if calculate_2d_spectrum is not None:
        print(f'\nComputing foreground cross-power spectra for {n_subsample} patches...')
        print(f'  Parameters: delta_ell={DELTA_ELL}, ell_max={LMAX} (all scales), pix_size={PIX_SIZE}, N={PATCH_SIZE}')
        for patch_idx in tqdm(range(n_subsample), desc='FG cross-spectra', unit='patch'):
            try:
                true_fg_patch = b_all_scales[patch_idx]
                unet_fg_patch = unet_fg_220_reconstruction[patch_idx]
                (ell_array, unet_fg_cross_ps) = compute_normalized_cross_spectrum(true_fg_patch, unet_fg_patch, delta_ell=DELTA_ELL, ell_max=LMAX, pix_size=PIX_SIZE, N=PATCH_SIZE)
                unet_fg_cross_spectra_list.append(unet_fg_cross_ps)
                if ell_array_fg_ref is None:
                    ell_array_fg_ref = ell_array.copy()
            except Exception as e:
                print(f'    Warning: Error computing foreground cross-spectrum for patch {patch_idx}: {e}')
                if ell_array_fg_ref is not None:
                    nan_array = np.full_like(ell_array_fg_ref, np.nan)
                    unet_fg_cross_spectra_list.append(nan_array)
                else:
                    continue
        unet_fg_cross_spectra_array = np.array(unet_fg_cross_spectra_list)
        mean_unet_fg_cross_ps = np.nanmean(unet_fg_cross_spectra_array, axis=0)
        percentile_lower_unet_fg = np.nanpercentile(unet_fg_cross_spectra_array, 16, axis=0)
        percentile_upper_unet_fg = np.nanpercentile(unet_fg_cross_spectra_array, 84, axis=0)
        print(f'  [ok] Computed foreground cross-spectra for {len(unet_fg_cross_spectra_list)} patches')
    else:
        unet_fg_cross_spectra_array = None
        mean_unet_fg_cross_ps = None
        percentile_lower_unet_fg = None
        percentile_upper_unet_fg = None
        print('  Skipping foreground cross-spectra (calculate_2d_spectrum not available)')
    print(f'\nComputing foreground spatial correlations for {n_subsample} patches...')
    unet_fg_spatial_correlations = []
    for patch_idx in tqdm(range(n_subsample), desc='FG correlations', unit='patch'):
        try:
            true_fg_patch = b_all_scales[patch_idx]
            unet_fg_patch = unet_fg_220_reconstruction[patch_idx]
            true_fg_flat = true_fg_patch.flatten()
            unet_fg_flat = unet_fg_patch.flatten()
            corr = np.corrcoef(true_fg_flat, unet_fg_flat)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            unet_fg_spatial_correlations.append(corr)
        except Exception as e:
            print(f'    Warning: Error computing foreground correlation for patch {patch_idx}: {e}')
            unet_fg_spatial_correlations.append(0.0)
    unet_fg_spatial_correlations = np.array(unet_fg_spatial_correlations)
    print(f'  [ok] Computed foreground correlations for {len(unet_fg_spatial_correlations)} patches')
    print(f'    Mean: {np.mean(unet_fg_spatial_correlations):.4f}, Std: {np.std(unet_fg_spatial_correlations):.4f}')
    print(f'\nComputing null correlations for foreground reconstructions...')
    print(f'  For each patch: correlating with {min(100, n_subsample - 1)} random true foreground patches (excluding its own)')
    fg_null_correlations_list = []
    for patch_idx in tqdm(range(n_subsample), desc='FG null correlations', unit='patch'):
        try:
            unet_fg_patch = unet_fg_220_reconstruction[patch_idx]
            unet_fg_flat = unet_fg_patch.flatten()
            available_indices = [j for j in range(n_subsample) if j != patch_idx]
            n_samples = min(100, len(available_indices))
            random_indices = np.random.choice(available_indices, size=n_samples, replace=False)
            for j in random_indices:
                try:
                    true_fg_patch = b_all_scales[j]
                    true_fg_flat = true_fg_patch.flatten()
                    corr = np.corrcoef(unet_fg_flat, true_fg_flat)[0, 1]
                    if not np.isnan(corr):
                        fg_null_correlations_list.append(corr)
                except Exception:
                    continue
        except Exception as e:
            print(f'    Warning: Error computing null correlation for patch {patch_idx}: {e}')
            continue
    fg_null_correlations_array = np.array(fg_null_correlations_list)
    fg_mean_null_correlation_overall = np.mean(fg_null_correlations_array) if len(fg_null_correlations_array) > 0 else None
    print(f'    Total null correlations: {len(fg_null_correlations_array)}')
    if fg_mean_null_correlation_overall is not None:
        print(f'    Mean null correlation: {fg_mean_null_correlation_overall:.6f} (should be close to 0)')
        print(f'    Std of null correlations: {np.std(fg_null_correlations_array):.6f}')
    print(f'\nComputing foreground MSEs for {n_subsample} patches...')
    ILC_fg_mse_list = []
    unet_fg_mse_list = []
    for patch_idx in tqdm(range(n_subsample), desc='FG MSE', unit='patch'):
        try:
            true_fg_patch = b_all_scales[patch_idx]
            ilc_fg_patch = ILC_fg_reconstruction[patch_idx]
            unet_fg_patch = unet_fg_220_reconstruction[patch_idx]
            ilc_fg_mse = compute_patch_mse(true_fg_patch, ilc_fg_patch)
            if ilc_fg_mse is not None:
                ILC_fg_mse_list.append(ilc_fg_mse)
            else:
                ILC_fg_mse_list.append(0.0)
            unet_fg_mse = compute_patch_mse(true_fg_patch, unet_fg_patch)
            if unet_fg_mse is not None:
                unet_fg_mse_list.append(unet_fg_mse)
            else:
                unet_fg_mse_list.append(0.0)
        except Exception as e:
            print(f'    Warning: Error computing foreground MSE for patch {patch_idx}: {e}')
            ILC_fg_mse_list.append(0.0)
            unet_fg_mse_list.append(0.0)
    ILC_fg_mse_array = np.array(ILC_fg_mse_list)
    unet_fg_mse_array = np.array(unet_fg_mse_list)
    print(f'    ILC foreground MSE - Mean: {np.mean(ILC_fg_mse_array):.6e}, Std: {np.std(ILC_fg_mse_array):.6e}')
    print(f'    UNet foreground MSE - Mean: {np.mean(unet_fg_mse_array):.6e}, Std: {np.std(unet_fg_mse_array):.6e}')
    if np.mean(unet_fg_mse_array) > 0:
        print(f'    Improvement ratio (ILC/UNet): {np.mean(ILC_fg_mse_array) / np.mean(unet_fg_mse_array):.4f}x')
    print('\n' + '=' * 80)
    print('SAVING TEST DATA FOR ANALYSIS')
    print('=' * 80)
    output_file = os.path.join(output_dir, 'test_results.npz')
    print(f'\nSaving test results to: {output_file}')
    np.savez(output_file, unet_predictions=predictions, unet_targets=targets, ilc_cmb=test_ilc_cmb, cmb_draw=test_cmb_draw, unet_cmb_reconstructed=unet_cmb_reconstructed, b_all_scales_220=b_all_scales, observed_cmb_220=observed_cmb_220, unet_fg_220_reconstruction=unet_fg_220_reconstruction, ILC_fg_reconstruction=ILC_fg_reconstruction, test_indices=test_indices_final, normalize=args.normalize, in_channels=in_channels, channel_mode=channel_suffix)
    print('  Test results saved!')
    if not args.no_viz:
        if args.sample_indices:
            sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
            max_idx = len(predictions) - 1
            sample_indices = [idx for idx in sample_indices if 0 <= idx <= max_idx]
            if len(sample_indices) == 0:
                print(f'\nWarning: No valid sample indices. Using first 50 samples.')
                sample_indices = list(range(min(50, len(predictions))))
        else:
            sample_indices = list(range(min(50, len(predictions))))
        subsample_index_map = {test_indices_final[i]: i for i in range(len(test_indices_final))}
        if args.sample_indices:
            original_sample_indices = [test_indices_final[idx] if idx < len(test_indices_final) else idx for idx in sample_indices]
        else:
            original_sample_indices = list(test_indices_final[:min(50, len(test_indices_final))])
        print(f'\nVisualizing samples: {original_sample_indices}')
        print(f"  These are original dataset indices. Mapped to array positions: {[subsample_index_map[idx] if idx in subsample_index_map else 'N/A' for idx in original_sample_indices]}")
        visualize_cmb_reconstructions(test_cmb_draw, test_ilc_cmb, unet_cmb_reconstructed, original_sample_indices, output_dir=output_dir, ell_array=ell_array_ref, unet_cross_spectra_array=unet_cross_spectra_array, ilc_cross_spectra_array=ilc_cross_spectra_array, mean_unet_cross_ps=mean_unet_cross_ps, percentile_lower_unet=percentile_lower_unet, percentile_upper_unet=percentile_upper_unet, mean_ilc_cross_ps=mean_ilc_cross_ps, percentile_lower_ilc=percentile_lower_ilc, percentile_upper_ilc=percentile_upper_ilc, null_cross_spectra_array=null_cross_spectra_array, mean_null_cross_ps_overall=mean_null_cross_ps_overall, unet_spatial_correlations=unet_spatial_correlations, null_correlations_array=null_correlations_array, mean_null_correlation_overall=mean_null_correlation_overall, ilc_cmb_mse_array=ilc_cmb_mse_array, unet_recon_mse_array=unet_recon_mse_array, index_mapping=subsample_index_map)
        print(f'\nCreating foreground reconstruction visualizations (all scales) for samples: {sample_indices}')
        visualize_fg_reconstructions(b_all_scales, ILC_fg_reconstruction, unet_fg_220_reconstruction, sample_indices, output_dir=output_dir, ell_array=ell_array_fg_ref, unet_fg_cross_spectra_array=unet_fg_cross_spectra_array, mean_unet_fg_cross_ps=mean_unet_fg_cross_ps, percentile_lower_unet_fg=percentile_lower_unet_fg, percentile_upper_unet_fg=percentile_upper_unet_fg, null_cross_spectra_array=null_cross_spectra_array, mean_null_cross_ps_overall=mean_null_cross_ps_overall, unet_fg_spatial_correlations=unet_fg_spatial_correlations, null_correlations_array=fg_null_correlations_array, mean_null_correlation_overall=fg_mean_null_correlation_overall, ILC_fg_mse_array=ILC_fg_mse_array, unet_fg_mse_array=unet_fg_mse_array)
        print('\n' + '=' * 80)
        print('COMPUTING LARGE-SCALE FOREGROUND RECONSTRUCTION ANALYSIS (ell < 200)')
        print('=' * 80)
        ell_cutoff_large_scale = 200
        print(f'\nFiltering foregrounds to large scales (ell < {ell_cutoff_large_scale})...')
        b_all_scales_large = np.array([filter_patch(b_all_scales[i], ell_cutoff=ell_cutoff_large_scale, filter_dir='large_scale', pix_size=PIX_SIZE) for i in tqdm(range(n_subsample), desc='Filtering true FG', unit='patch')])
        ILC_fg_reconstruction_large = np.array([filter_patch(ILC_fg_reconstruction[i], ell_cutoff=ell_cutoff_large_scale, filter_dir='large_scale', pix_size=PIX_SIZE) for i in tqdm(range(n_subsample), desc='Filtering ILC FG', unit='patch')])
        unet_fg_220_reconstruction_large = np.array([filter_patch(unet_fg_220_reconstruction[i], ell_cutoff=ell_cutoff_large_scale, filter_dir='large_scale', pix_size=PIX_SIZE) for i in tqdm(range(n_subsample), desc='Filtering UNet FG', unit='patch')])
        unet_fg_cross_spectra_list_large = []
        ell_array_fg_large_ref = None
        if calculate_2d_spectrum is not None:
            print(f'\nComputing large-scale foreground cross-power spectra for {n_subsample} patches...')
            print(f'  Parameters: delta_ell={DELTA_ELL}, ell_max={ell_cutoff_large_scale} (large scales), pix_size={PIX_SIZE}, N={PATCH_SIZE}')
            for patch_idx in tqdm(range(n_subsample), desc='FG large-scale cross-spectra', unit='patch'):
                try:
                    true_fg_patch_large = b_all_scales_large[patch_idx]
                    unet_fg_patch_large = unet_fg_220_reconstruction_large[patch_idx]
                    (ell_array, unet_fg_cross_ps) = compute_normalized_cross_spectrum(true_fg_patch_large, unet_fg_patch_large, delta_ell=DELTA_ELL, ell_max=ell_cutoff_large_scale, pix_size=PIX_SIZE, N=PATCH_SIZE)
                    unet_fg_cross_spectra_list_large.append(unet_fg_cross_ps)
                    if ell_array_fg_large_ref is None:
                        ell_array_fg_large_ref = ell_array.copy()
                except Exception as e:
                    print(f'    Warning: Error computing large-scale foreground cross-spectrum for patch {patch_idx}: {e}')
                    if ell_array_fg_large_ref is not None:
                        nan_array = np.full_like(ell_array_fg_large_ref, np.nan)
                        unet_fg_cross_spectra_list_large.append(nan_array)
                    else:
                        continue
            unet_fg_cross_spectra_array_large = np.array(unet_fg_cross_spectra_list_large)
            mean_unet_fg_cross_ps_large = np.nanmean(unet_fg_cross_spectra_array_large, axis=0)
            percentile_lower_unet_fg_large = np.nanpercentile(unet_fg_cross_spectra_array_large, 16, axis=0)
            percentile_upper_unet_fg_large = np.nanpercentile(unet_fg_cross_spectra_array_large, 84, axis=0)        else:
            unet_fg_cross_spectra_array_large = None
            mean_unet_fg_cross_ps_large = None
            percentile_lower_unet_fg_large = None
            percentile_upper_unet_fg_large = None
            ell_array_fg_large_ref = None
            print('  Skipping large-scale foreground cross-spectra (calculate_2d_spectrum not available)')
        print(f'\nComputing large-scale foreground spatial correlations for {n_subsample} patches...')
        unet_fg_spatial_correlations_large = []
        for patch_idx in tqdm(range(n_subsample), desc='FG large-scale correlations', unit='patch'):
            try:
                true_fg_patch_large = b_all_scales_large[patch_idx]
                unet_fg_patch_large = unet_fg_220_reconstruction_large[patch_idx]
                true_fg_flat = true_fg_patch_large.flatten()
                unet_fg_flat = unet_fg_patch_large.flatten()
                corr = np.corrcoef(true_fg_flat, unet_fg_flat)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                unet_fg_spatial_correlations_large.append(corr)
            except Exception as e:
                print(f'    Warning: Error computing large-scale foreground correlation for patch {patch_idx}: {e}')
                unet_fg_spatial_correlations_large.append(0.0)
        unet_fg_spatial_correlations_large = np.array(unet_fg_spatial_correlations_large)
        print(f'    Mean: {np.mean(unet_fg_spatial_correlations_large):.4f}, Std: {np.std(unet_fg_spatial_correlations_large):.4f}')
        print(f'\nComputing null correlations for large-scale foreground reconstructions...')
        print(f'  For each patch: correlating with {min(100, n_subsample - 1)} random large-scale true foreground patches (excluding its own)')
        fg_null_correlations_list_large = []
        for patch_idx in tqdm(range(n_subsample), desc='FG large-scale null correlations', unit='patch'):
            try:
                unet_fg_patch_large = unet_fg_220_reconstruction_large[patch_idx]
                unet_fg_flat = unet_fg_patch_large.flatten()
                available_indices = [j for j in range(n_subsample) if j != patch_idx]
                n_samples = min(100, len(available_indices))
                random_indices = np.random.choice(available_indices, size=n_samples, replace=False)
                for j in random_indices:
                    try:
                        true_fg_patch_large = b_all_scales_large[j]
                        true_fg_flat = true_fg_patch_large.flatten()
                        corr = np.corrcoef(unet_fg_flat, true_fg_flat)[0, 1]
                        if not np.isnan(corr):
                            fg_null_correlations_list_large.append(corr)
                    except Exception:
                        continue
            except Exception as e:
                print(f'    Warning: Error computing null correlation for patch {patch_idx}: {e}')
                continue
        fg_null_correlations_array_large = np.array(fg_null_correlations_list_large)
        fg_mean_null_correlation_overall_large = np.mean(fg_null_correlations_array_large) if len(fg_null_correlations_array_large) > 0 else None
        print(f'    Total null correlations: {len(fg_null_correlations_array_large)}')
        if fg_mean_null_correlation_overall_large is not None:
            print(f'    Mean null correlation: {fg_mean_null_correlation_overall_large:.6f} (should be close to 0)')
            print(f'    Std of null correlations: {np.std(fg_null_correlations_array_large):.6f}')
        print(f'\nComputing large-scale foreground MSEs for {n_subsample} patches...')
        ILC_fg_mse_list_large = []
        unet_fg_mse_list_large = []
        for patch_idx in tqdm(range(n_subsample), desc='FG large-scale MSE', unit='patch'):
            try:
                true_fg_patch_large = b_all_scales_large[patch_idx]
                ilc_fg_patch_large = ILC_fg_reconstruction_large[patch_idx]
                unet_fg_patch_large = unet_fg_220_reconstruction_large[patch_idx]
                ilc_fg_mse = compute_patch_mse(true_fg_patch_large, ilc_fg_patch_large)
                if ilc_fg_mse is not None:
                    ILC_fg_mse_list_large.append(ilc_fg_mse)
                else:
                    ILC_fg_mse_list_large.append(0.0)
                unet_fg_mse = compute_patch_mse(true_fg_patch_large, unet_fg_patch_large)
                if unet_fg_mse is not None:
                    unet_fg_mse_list_large.append(unet_fg_mse)
                else:
                    unet_fg_mse_list_large.append(0.0)
            except Exception as e:
                print(f'    Warning: Error computing large-scale foreground MSE for patch {patch_idx}: {e}')
                ILC_fg_mse_list_large.append(0.0)
                unet_fg_mse_list_large.append(0.0)
        ILC_fg_mse_array_large = np.array(ILC_fg_mse_list_large)
        unet_fg_mse_array_large = np.array(unet_fg_mse_list_large)
        print(f'    ILC foreground MSE - Mean: {np.mean(ILC_fg_mse_array_large):.6e}, Std: {np.std(ILC_fg_mse_array_large):.6e}')
        print(f'    UNet foreground MSE - Mean: {np.mean(unet_fg_mse_array_large):.6e}, Std: {np.std(unet_fg_mse_array_large):.6e}')
        if np.mean(unet_fg_mse_array_large) > 0:
            print(f'    Improvement ratio (ILC/UNet): {np.mean(ILC_fg_mse_array_large) / np.mean(unet_fg_mse_array_large):.4f}x')
        visualize_fg_reconstructions_large_scale(b_all_scales_large, ILC_fg_reconstruction_large, unet_fg_220_reconstruction_large, sample_indices, output_dir=output_dir, ell_array=ell_array_fg_large_ref, unet_fg_cross_spectra_array=unet_fg_cross_spectra_array_large, mean_unet_fg_cross_ps=mean_unet_fg_cross_ps_large, percentile_lower_unet_fg=percentile_lower_unet_fg_large, percentile_upper_unet_fg=percentile_upper_unet_fg_large, null_cross_spectra_array=None, mean_null_cross_ps_overall=None, unet_fg_spatial_correlations=unet_fg_spatial_correlations_large, null_correlations_array=fg_null_correlations_array_large, mean_null_correlation_overall=fg_mean_null_correlation_overall_large, ILC_fg_mse_array=ILC_fg_mse_array_large, unet_fg_mse_array=unet_fg_mse_array_large, ell_cutoff=ell_cutoff_large_scale)
    print('\n' + '=' * 80)
    print('SUMMARY STATISTICS')
    print('=' * 80)
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    mse = np.mean((target_flat - pred_flat) ** 2)
    mae = np.mean(np.abs(target_flat - pred_flat))
    corr = np.corrcoef(target_flat, pred_flat)[0, 1]
    print(f'\nUNet Residual Prediction Metrics (all test samples):')
    print(f'  MSE: {mse:.3e}')
    print(f'  MAE: {mae:.3e}')
    print(f'  Correlation: {corr:.3f}')
    print(f'\nDataset Statistics:')
    print(f'  Number of test samples: {len(predictions)}')
    print(f'  Spatial dimensions: {predictions.shape[1:]} (H, W)')
    print(f'\n  UNet predictions - Mean: {pred_flat.mean():.3e}, Std: {pred_flat.std():.3e}')
    print(f'  UNet targets - Mean: {target_flat.mean():.3e}, Std: {target_flat.std():.3e}')
    print(f'  ILC CMB - Mean: {test_ilc_cmb.flatten().mean():.3e}, Std: {test_ilc_cmb.flatten().std():.3e}')
    print(f'  Pure CMB - Mean: {test_cmb_draw.flatten().mean():.3e}, Std: {test_cmb_draw.flatten().std():.3e}')
    print(f'  UNet CMB reconstructed - Mean: {unet_cmb_reconstructed.flatten().mean():.3e}, Std: {unet_cmb_reconstructed.flatten().std():.3e}')
    print(f'\nCMB Reconstruction Correlations:')
    pure_cmb_flat = test_cmb_draw.flatten()
    ilc_cmb_flat = test_ilc_cmb.flatten()
    unet_cmb_flat = unet_cmb_reconstructed.flatten()
    corr_ilc = np.corrcoef(pure_cmb_flat, ilc_cmb_flat)[0, 1]
    corr_unet = np.corrcoef(pure_cmb_flat, unet_cmb_flat)[0, 1]
    print(f'  ILC CMB vs Pure CMB correlation: {corr_ilc:.4f}')
    print(f'  UNet CMB vs Pure CMB correlation: {corr_unet:.4f}')
    print(f'  Improvement: {corr_unet - corr_ilc:+.4f}')
    print(f'\nCMB Reconstruction Variance:')
    if 'true_cmb_var_array' in locals():
        print(f'  True CMB Variance (baseline) - Mean: {np.mean(true_cmb_var_array):.6e}, Std: {np.std(true_cmb_var_array):.6e}')
    else:
        true_cmb_var_mean = np.var(test_cmb_draw, axis=(1, 2)).mean()
        true_cmb_var_std = np.var(test_cmb_draw, axis=(1, 2)).std()
        print(f'  True CMB Variance (baseline) - Mean: {true_cmb_var_mean:.6e}, Std: {true_cmb_var_std:.6e}')
    if 'unet_cmb_var_array' in locals():
        print(f'  UNet CMB Variance - Mean: {np.mean(unet_cmb_var_array):.6e}, Std: {np.std(unet_cmb_var_array):.6e}')
    else:
        unet_cmb_var_mean = np.var(unet_cmb_reconstructed, axis=(1, 2)).mean()
        unet_cmb_var_std = np.var(unet_cmb_reconstructed, axis=(1, 2)).std()
        print(f'  UNet CMB Variance - Mean: {unet_cmb_var_mean:.6e}, Std: {unet_cmb_var_std:.6e}')
    print(f'\nForeground Reconstruction Statistics (220 GHz):')
    b_all_scales_flat = b_all_scales.flatten()
    unet_fg_flat = unet_fg_220_reconstruction.flatten()
    ilc_fg_flat = ILC_fg_reconstruction.flatten()
    corr_unet_fg = np.corrcoef(b_all_scales_flat, unet_fg_flat)[0, 1]
    corr_ilc_fg = np.corrcoef(b_all_scales_flat, ilc_fg_flat)[0, 1]
    mse_unet_fg = np.mean((b_all_scales_flat - unet_fg_flat) ** 2)
    mse_ilc_fg = np.mean((b_all_scales_flat - ilc_fg_flat) ** 2)
    print(f'  True foreground (b_all_scales) - Mean: {b_all_scales_flat.mean():.3e}, Std: {b_all_scales_flat.std():.3e}')
    print(f'  UNet foreground - Mean: {unet_fg_flat.mean():.3e}, Std: {unet_fg_flat.std():.3e}')
    print(f'  ILC foreground - Mean: {ilc_fg_flat.mean():.3e}, Std: {ilc_fg_flat.std():.3e}')
    print(f'  UNet foreground vs True foreground correlation: {corr_unet_fg:.4f}')
    print(f'  ILC foreground vs True foreground correlation: {corr_ilc_fg:.4f}')
    print(f'  UNet foreground MSE: {mse_unet_fg:.6e}')
    print(f'  ILC foreground MSE: {mse_ilc_fg:.6e}')
    if mse_unet_fg > 0:
        print(f'  Improvement ratio (ILC/UNet MSE): {mse_ilc_fg / mse_unet_fg:.4f}x')
    else:
        print(f'  Improvement ratio: N/A (UNet MSE is zero)')
    print(f'\n Test results saved to: {output_file}')
if __name__ == '__main__':
    main()
