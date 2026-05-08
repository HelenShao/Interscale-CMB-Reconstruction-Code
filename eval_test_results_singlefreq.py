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
import time
from tqdm import tqdm
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
from paths_config import B_SMALL_DIR, OUTPUT_DIR_SINGLEFREQ as OUTPUT_DIR, STATS_FILE_SINGLEFREQ as STATS_FILE, T_E_DIR
FREQUENCY = 220
import healpy as hp
NSIDE = 1024
PIX_SIZE = int(hp.nside2resol(NSIDE, arcmin=True))
DELTA_ELL = 50
ELL_CUTOFF = 200
LMAX = 2000
PATCH_SIZE = 128
PHILCOX_ELL = np.array([30, 60, 75, 90, 100, 120, 135, 150, 165, 180, 200, 215, 235, 250, 270, 285])
PHILCOX_Y = np.array([0.5, 0.2, 0.25, 0.07, 0.02, 0.12, 0.03, 0.11, 0.13, -0.08, 0.17, -0.08, -0.12, -0.22, -0.03, 0.02])
PHILCOX_YERR = np.array([0.24, 0.06, 0.14, 0.08, 0.07, 0.08, 0.08, 0.06, 0.04, 0.08, 0.1, 0.06, 0.06, 0.05, 0.05, 0.05])
CONFIG = {'batch_size': 32, 'in_channels': 3, 'out_channels': 1, 'feature_dims': [32, 64, 128, 256, 512, 1024], 'negative_slope': 0.01, 'toy_model': 'SingleFreqInterscale'}

def load_test_data(normalize=True, input_mode='bte', n_subsample=None):
    print('\n' + '=' * 80)
    print('LOADING TEST DATA')
    print('=' * 80)
    load_tasks = [('stats', STATS_FILE)]
    b_small_path = f'{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_B_patches_small.npy'
    if input_mode in ['bte', 'b_only']:
        load_tasks.append(('B_small', b_small_path))
    load_tasks.append(('B_large', f'{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_B_patches_large.npy'))
    load_tasks.append(('cmb_draw', f'{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_cmb_draw_b_all_scales.npy'))
    if input_mode in ['bte', 'te_only', 't_only']:
        load_tasks.append(('T', f'{T_E_DIR}/sim1-150_freq{FREQUENCY}_T_patches.npy'))
    if input_mode in ['bte', 'te_only', 'e_only']:
        load_tasks.append(('E', f'{T_E_DIR}/sim1-150_freq{FREQUENCY}_E_patches.npy'))
    stats = None
    b_small = None
    b_large = None
    cmb_draw_b_all_scales = None
    t_all = None
    e_all = None
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
            elif label == 'B_small':
                b_small = data
            elif label == 'B_large':
                b_large = data
            elif label == 'cmb_draw':
                cmb_draw_b_all_scales = data
            elif label == 'T':
                t_all = data
            elif label == 'E':
                e_all = data
    if stats is None:
        raise RuntimeError('stats not loaded')
    test_indices = stats['test_indices']
    print(f'\n  Test indices: {len(test_indices)} samples')
    if input_mode not in ['bte', 'te_only', 'e_only']:
        print('\nSkipping E channel (not used for this input mode)')
    print('\nVerifying data consistency...')
    if b_small is not None:
        n_patches = b_small.shape[0]
        ref_spatial_shape = b_small.shape[1:]
    elif t_all is not None:
        n_patches = t_all.shape[0]
        ref_spatial_shape = t_all.shape[1:]
    elif e_all is not None:
        n_patches = e_all.shape[0]
        ref_spatial_shape = e_all.shape[1:]
    else:
        raise ValueError('No input data loaded!')
    print(f'  Total patches: {n_patches}')
    assert b_large.shape[0] == n_patches, f'Mismatch: B large has {b_large.shape[0]} patches, expected {n_patches}'
    assert b_large.shape[1:] == ref_spatial_shape, f'Mismatch: B large has spatial shape {b_large.shape[1:]}, expected {ref_spatial_shape}'
    assert cmb_draw_b_all_scales.shape[0] == n_patches, f'Mismatch: cmb_draw_b_all_scales has {cmb_draw_b_all_scales.shape[0]} patches, expected {n_patches}'
    assert cmb_draw_b_all_scales.shape[1:] == ref_spatial_shape, f'Mismatch: cmb_draw_b_all_scales has spatial shape {cmb_draw_b_all_scales.shape[1:]}, expected {ref_spatial_shape}'
    if b_small is not None:
        assert b_small.shape[0] == n_patches, f'Mismatch: B small has {b_small.shape[0]} patches, expected {n_patches}'
        assert b_small.shape[1:] == ref_spatial_shape, f'Mismatch: B small has spatial shape {b_small.shape[1:]}, expected {ref_spatial_shape}'
    if t_all is not None:
        assert t_all.shape[0] == n_patches, f'Mismatch: T all has {t_all.shape[0]} patches, expected {n_patches}'
        assert t_all.shape[1:] == ref_spatial_shape, f'Mismatch: T all has spatial shape {t_all.shape[1:]}, expected {ref_spatial_shape}'
    if e_all is not None:
        assert e_all.shape[0] == n_patches, f'Mismatch: E all has {e_all.shape[0]} patches, expected {n_patches}'
        assert e_all.shape[1:] == ref_spatial_shape, f'Mismatch: E all has spatial shape {e_all.shape[1:]}, expected {ref_spatial_shape}'
    print('  [ok] All datasets have matching patch counts and spatial dimensions')
    print('\nConstructing observed B all-scales = CMB + B_large + B_small...')
    print('  This represents the total observed B-mode signal (CMB + foregrounds at all scales)')
    if b_small is None:
        if not os.path.exists(b_small_path):
            raise FileNotFoundError(f'B small-scale file not found: {b_small_path}')
        t0 = time.perf_counter()
        b_small = np.load(b_small_path)
        tqdm.write(f'  Loaded B_small for CMB reconstruction in {time.perf_counter() - t0:.2f}s  shape={b_small.shape}')
    b_all_scales = cmb_draw_b_all_scales + b_large + b_small
    print(f'  observed B all-scales shape: {b_all_scales.shape}')
    print('\nStacking input channels...')
    if input_mode == 'b_only':
        input_channels = [b_small]
    elif input_mode == 't_only':
        input_channels = [t_all]
    elif input_mode == 'e_only':
        input_channels = [e_all]
    elif input_mode == 'te_only':
        input_channels = [t_all, e_all]
    else:
        input_channels = [b_small, t_all, e_all]
    UNet_input = np.stack(input_channels, axis=1)
    n_channels = len(input_channels)
    if input_mode == 'b_only':
        print(f'  Using {n_channels}-channel input (B small-scale only)')
    elif input_mode == 't_only':
        print(f'  Using {n_channels}-channel input (T only)')
    elif input_mode == 'e_only':
        print(f'  Using {n_channels}-channel input (E only)')
    elif input_mode == 'te_only':
        print(f'  Using {n_channels}-channel input (T + E only)')
    else:
        print(f'  Using {n_channels}-channel input (B small-scale + T + E)')
    UNet_target = b_large[:, np.newaxis, :, :]
    print(f'\nExtracting test set using test_indices (ensuring alignment)...')
    test_X = UNet_input[test_indices]
    test_y = UNet_target[test_indices]
    test_b_all_scales = b_all_scales[test_indices]
    test_cmb_draw = cmb_draw_b_all_scales[test_indices]
    test_indices_final = test_indices.copy()
    if n_subsample is not None:
        n_test = len(test_X)
        n_subsample_actual = min(n_subsample, n_test)
        print(f'\nSubsampling first {n_subsample_actual} patches from {n_test} total test patches...')
        subsample_indices = np.arange(n_subsample_actual)
        test_X = test_X[subsample_indices]
        test_y = test_y[subsample_indices]
        test_b_all_scales = test_b_all_scales[subsample_indices]
        test_cmb_draw = test_cmb_draw[subsample_indices]
        test_indices_final = test_indices_final[subsample_indices]
        print(f'  After subsampling: {n_subsample_actual} patches')
    print(f'\nTest set shapes (all aligned using same test_indices):')
    print(f'  UNet input: {test_X.shape}')
    print(f'  UNet target: {test_y.shape}')
    print(f'  Observed B all-scales: {test_b_all_scales.shape}')
    print(f'  Pure CMB (cmb_draw): {test_cmb_draw.shape}')
    assert test_X.shape[0] == test_y.shape[0] == test_b_all_scales.shape[0] == test_cmb_draw.shape[0], 'Mismatch in test set sizes - alignment issue!'
    print('  [ok] All test datasets have matching sample counts (properly aligned)')
    if normalize:
        print('\nNormalizing test data using train statistics...')
        full_input_mean = stats['input_mean']
        full_input_std = stats['input_std']
        target_mean = stats['target_mean']
        target_std = stats['target_std']
        if input_mode == 'bte':
            input_mean = full_input_mean
            input_std = full_input_std
        elif input_mode == 'b_only':
            input_mean = full_input_mean[:1] if len(full_input_mean) > 1 else full_input_mean
            input_std = full_input_std[:1] if len(full_input_std) > 1 else full_input_std
        elif input_mode == 't_only':
            if len(full_input_mean) >= 3:
                input_mean = full_input_mean[1:2]
                input_std = full_input_std[1:2]
            else:
                raise ValueError(f'Expected 3-channel stats for t_only mode, got {len(full_input_mean)} channels')
        elif input_mode == 'e_only':
            if len(full_input_mean) >= 3:
                input_mean = full_input_mean[2:3]
                input_std = full_input_std[2:3]
            else:
                raise ValueError(f'Expected 3-channel stats for e_only mode, got {len(full_input_mean)} channels')
        elif len(full_input_mean) >= 3:
            input_mean = full_input_mean[1:3]
            input_std = full_input_std[1:3]
        else:
            raise ValueError(f'Expected 3-channel stats for te_only mode, got {len(full_input_mean)} channels')
        n_channels = test_X.shape[1]
        for i in range(n_channels):
            test_X[:, i] = (test_X[:, i] - input_mean[i]) / (input_std[i] + 1e-08)
        test_y = (test_y - target_mean) / (target_std + 1e-08)
        print('  Normalization complete')
        return (test_X, test_y, test_b_all_scales, test_cmb_draw, test_indices_final, {'input_mean': input_mean, 'input_std': input_std, 'target_mean': target_mean, 'target_std': target_std})
    else:
        print('\nUsing unnormalized data')
        return (test_X, test_y, test_b_all_scales, test_cmb_draw, test_indices_final, None)

def _save_loaded_test_tensors(path, test_X, test_y, test_b_all_scales, test_cmb_draw, test_indices_final, norm_stats, input_mode, normalized, in_channels, b_small_dir, frequency):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    b_small_file = os.path.join(b_small_dir, f'sim1-150_freq{frequency}_B_patches_small.npy')
    mm = np.load(b_small_file, mmap_mode='r')
    test_b_small = np.asarray(mm[test_indices_final])
    save_kw = {'test_X': test_X, 'test_y': test_y, 'test_b_all_scales': test_b_all_scales, 'test_cmb_draw': test_cmb_draw, 'test_b_small': test_b_small, 'test_indices': test_indices_final, 'input_mode': np.array(input_mode), 'normalized': np.bool_(normalized), 'in_channels': np.int32(in_channels), 'frequency_GHz': np.int32(frequency)}
    if norm_stats is not None:
        for (k, v) in norm_stats.items():
            save_kw[f'norm_{k}'] = np.asarray(v)
    np.savez_compressed(path, **save_kw)
    print(f'\n[ok] Saved tensors for loaded test split (before model inference) to:\n  {path}')
    print(f'    Keys: {", ".join(sorted(save_kw.keys()))}')

def load_eval_from_npz_singlefreq(npz_path, input_mode, n_subsample, normalize_run):
    z = np.load(npz_path, allow_pickle=True)
    required = ('test_X', 'test_y', 'test_b_all_scales', 'test_cmb_draw', 'test_indices', 'test_b_small')
    for k in required:
        if k not in z.files:
            raise KeyError(f'{npz_path}: missing required array {k!r}')
    full_X = np.asarray(z['test_X'])
    if full_X.ndim != 4 or full_X.shape[1] != 3:
        raise ValueError(f'--test-data-npz expects a B+T+E export with test_X shape (N,3,H,W); got {full_X.shape}')
    stored_normalized = bool(np.asarray(z['normalized']).item()) if 'normalized' in z.files else False
    if normalize_run != stored_normalized:
        print(f"  WARNING: npz normalized={stored_normalized} but --normalize is {normalize_run}; expect mismatched scaling.")
    test_y = np.asarray(z['test_y'])
    test_b_all_scales = np.asarray(z['test_b_all_scales'])
    test_cmb_draw = np.asarray(z['test_cmb_draw'])
    test_b_small = np.asarray(z['test_b_small'])
    test_indices_final = np.asarray(z['test_indices'])
    if n_subsample is not None:
        n_cap = min(int(n_subsample), full_X.shape[0])
        full_X = full_X[:n_cap]
        test_y = test_y[:n_cap]
        test_b_all_scales = test_b_all_scales[:n_cap]
        test_cmb_draw = test_cmb_draw[:n_cap]
        test_b_small = test_b_small[:n_cap]
        test_indices_final = test_indices_final[:n_cap]
        print(f'  Subsampled npz data to first {n_cap} patches')
    if input_mode == 'b_only':
        test_X = full_X[:, 0:1, :, :]
    elif input_mode == 't_only':
        test_X = full_X[:, 1:2, :, :]
    elif input_mode == 'e_only':
        test_X = full_X[:, 2:3, :, :]
    elif input_mode == 'te_only':
        test_X = full_X[:, 1:3, :, :]
    else:
        test_X = np.asarray(full_X)
    t_all = full_X[:, 1, :, :]
    e_all = full_X[:, 2, :, :]
    norm_stats = None
    if normalize_run:
        raw = {k[5:]: np.asarray(z[k]) for k in z.files if k.startswith('norm_')}
        if not raw or 'target_mean' not in raw:
            raise ValueError('npz has no norm_* fields; use an export saved with --normalize, or run without --normalize')
        full_im = raw['input_mean']
        full_is = raw['input_std']
        if input_mode == 'bte':
            input_mean = full_im
            input_std = full_is
        elif input_mode == 'b_only':
            input_mean = full_im[:1]
            input_std = full_is[:1]
        elif input_mode == 't_only':
            input_mean = full_im[1:2]
            input_std = full_is[1:2]
        elif input_mode == 'e_only':
            input_mean = full_im[2:3]
            input_std = full_is[2:3]
        elif input_mode == 'te_only':
            input_mean = full_im[1:3]
            input_std = full_is[1:3]
        else:
            raise ValueError(input_mode)
        norm_stats = {'input_mean': input_mean, 'input_std': input_std, 'target_mean': raw['target_mean'], 'target_std': raw['target_std']}
    print(f'\n[ok] Loaded test data from npz (channels after slice: {test_X.shape[1]})')
    bundle = {'test_b_small': test_b_small, 't_all': t_all, 'e_all': e_all}
    return (test_X, test_y, test_b_all_scales, test_cmb_draw, test_indices_final, norm_stats, bundle)

def load_model(model_path, normalize=True, in_channels=3):
    print(f'\nLoading model from: {model_path}')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        print(f'  Found architecture config in checkpoint:')
        print(f"    feature_dims: {saved_config.get('feature_dims', 'Not found')}")
        print(f"    in_channels: {saved_config.get('in_channels', 'Not found')}")
        print(f"    out_channels: {saved_config.get('out_channels', 'Not found')}")
        print(f"    negative_slope: {saved_config.get('negative_slope', 'Not found')}")
        model_in_channels = saved_config.get('in_channels', in_channels)
        if model_in_channels != in_channels:
            print(f'  WARNING: WARNING: Checkpoint has in_channels={model_in_channels}, but function argument is in_channels={in_channels}')
            print(f'    Using checkpoint value: {model_in_channels}')
            in_channels = model_in_channels
        model_config = {'in_channels': in_channels, 'out_channels': saved_config.get('out_channels', CONFIG['out_channels']), 'feature_dims': saved_config.get('feature_dims', CONFIG['feature_dims']), 'negative_slope': saved_config.get('negative_slope', CONFIG['negative_slope'])}
    else:
        print(f"  WARNING: WARNING: No 'config' found in checkpoint, using default CONFIG")
        print(f"    Default feature_dims: {CONFIG['feature_dims']}")
        model_config = {'in_channels': in_channels, 'out_channels': CONFIG['out_channels'], 'feature_dims': CONFIG['feature_dims'], 'negative_slope': CONFIG['negative_slope']}
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
    model = architecture.UNET(in_channels=model_config['in_channels'], out_channels=model_config['out_channels'], feature_dims=model_config['feature_dims'], negative_slope=model_config['negative_slope'], mean_init=mean_init_path)
    print(f'  Model architecture:')
    print(f"    Input channels: {model_config['in_channels']}")
    print(f"    Output channels: {model_config['out_channels']}")
    print(f"    Feature dimensions: {model_config['feature_dims']}")
    print(f"    Negative slope: {model_config['negative_slope']}")
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

def compute_patch_mse(patch1, patch2):
    try:
        mse_val = np.mean((patch1 - patch2) ** 2)
        if np.isnan(mse_val) or np.isinf(mse_val):
            return None
        return float(mse_val)
    except Exception:
        return None

def compute_normalized_cross_spectrum(map1, map2, delta_ell=DELTA_ELL, ell_max=ELL_CUTOFF, pix_size=PIX_SIZE, N=PATCH_SIZE):
    if calculate_2d_spectrum is None:
        raise ValueError('calculate_2d_spectrum not available - cannot compute cross-spectrum')
    (ell_array, cl_cross) = calculate_2d_spectrum(map1, map2, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N, lib_mode='NumPy')
    (_, cl_map1_map1) = calculate_2d_spectrum(map1, map1, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N, lib_mode='NumPy')
    (_, cl_map2_map2) = calculate_2d_spectrum(map2, map2, delta_ell=delta_ell, ell_max=ell_max, pix_size=pix_size, N=N, lib_mode='NumPy')
    normalized_cross_ps = cl_cross / np.sqrt(cl_map1_map1 * cl_map2_map2)
    return (ell_array, normalized_cross_ps)

def visualize_cmb_reconstructions(cmb_draw, observed_b, unet_cmb_reconstructed, sample_indices, output_dir=None, ell_array=None, unet_cross_spectra_array=None, observed_cross_spectra_array=None, mean_unet_cross_ps=None, std_unet_cross_ps=None, unet_cross_ps_lower=None, unet_cross_ps_upper=None, mean_observed_cross_ps=None, std_observed_cross_ps=None, observed_cross_ps_lower=None, observed_cross_ps_upper=None, null_cross_spectra_array=None, mean_null_cross_ps_overall=None, unet_spatial_correlations=None, null_correlations_array=None, mean_null_correlation_overall=None, observed_b_mse_array=None, unet_recon_mse_array=None, index_mapping=None):
    print(f'\nCreating 1x6 plots for {len(sample_indices)} test samples (saving each separately)...')
    n_samples = len(sample_indices)
    n_cols = 6
    for (row_idx, sample_idx) in enumerate(sample_indices):
        (fig, axes) = plt.subplots(1, n_cols, figsize=(18, 3))
        axes = axes.reshape(1, -1) if n_cols > 1 else np.array([[axes]])
        if index_mapping is not None:
            if sample_idx not in index_mapping:
                print(f'  Skipping sample {sample_idx} (not in subsampled data)')
                plt.close()
                continue
            array_idx = index_mapping[sample_idx]
        else:
            array_idx = sample_idx
        pure_cmb = cmb_draw[array_idx]
        observed_b_patch = observed_b[array_idx]
        unet_cmb_patch = unet_cmb_reconstructed[array_idx]
        (vmin1, vmax1) = (pure_cmb.min(), pure_cmb.max())
        (vmin2, vmax2) = (observed_b_patch.min(), observed_b_patch.max())
        (vmin3, vmax3) = (unet_cmb_patch.min(), unet_cmb_patch.max())
        im1 = axes[0, 0].imshow(pure_cmb, cmap='RdBu_r', vmin=vmin1, vmax=vmax1, origin='lower')
        axes[0, 0].set_title(f'Sample {sample_idx}\nPrimary B-modes', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        im2 = axes[0, 1].imshow(observed_b_patch, cmap='RdBu_r', vmin=vmin2, vmax=vmax2, origin='lower')
        axes[0, 1].set_title(f'Sample {sample_idx}\nObserved B All-Scales', fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        im3 = axes[0, 2].imshow(unet_cmb_patch, cmap='RdBu_r', vmin=vmin3, vmax=vmax3, origin='lower')
        axes[0, 2].set_title(f'Sample {sample_idx}\nUNet CMB Reconstructed', fontsize=10, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        pure_flat = pure_cmb.flatten()
        observed_flat = observed_b_patch.flatten()
        unet_flat = unet_cmb_patch.flatten()
        corr_observed = np.corrcoef(pure_flat, observed_flat)[0, 1]
        corr_unet = np.corrcoef(pure_flat, unet_flat)[0, 1]
        axes[0, 2].text(0.02, 0.98, f'Observed corr: {corr_observed:.3f}\nUNet corr: {corr_unet:.3f}', transform=axes[0, 2].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
        ax4 = axes[0, 3]
        if ell_array is not None and unet_cross_spectra_array is not None and (observed_cross_spectra_array is not None):
            if array_idx < len(unet_cross_spectra_array):
                unet_cross_ps_sample = unet_cross_spectra_array[array_idx]
                observed_cross_ps_sample = observed_cross_spectra_array[array_idx]
                if unet_cross_ps_lower is not None and unet_cross_ps_upper is not None:
                    ax4.fill_between(ell_array, unet_cross_ps_lower, unet_cross_ps_upper, color='blue', alpha=0.2, label='$\\pm 1\\sigma$')
                    if mean_unet_cross_ps is not None:
                        ax4.plot(ell_array, mean_unet_cross_ps, 'b-', linewidth=2, alpha=0.6, label='UNet mean')
                elif mean_unet_cross_ps is not None and std_unet_cross_ps is not None:
                    ax4.fill_between(ell_array, mean_unet_cross_ps - std_unet_cross_ps, mean_unet_cross_ps + std_unet_cross_ps, color='orange', alpha=0.2, label='$\\pm 1\\sigma$')
                    ax4.plot(ell_array, mean_unet_cross_ps, 'b-', linewidth=2, alpha=0.6, label='UNet mean')
                if observed_cross_ps_lower is not None and observed_cross_ps_upper is not None:
                    ax4.fill_between(ell_array, observed_cross_ps_lower, observed_cross_ps_upper, color='orange', alpha=0.2, label='$\\pm 1\\sigma$')
                    if mean_observed_cross_ps is not None:
                        ax4.plot(ell_array, mean_observed_cross_ps, 'orange', linewidth=2, alpha=0.6, label='Observed mean')
                elif mean_observed_cross_ps is not None and std_observed_cross_ps is not None:
                    ax4.fill_between(ell_array, mean_observed_cross_ps - std_observed_cross_ps, mean_observed_cross_ps + std_observed_cross_ps, color='orange', alpha=0.2, label='$\\pm 1\\sigma$')
                    ax4.plot(ell_array, mean_observed_cross_ps, 'orange', linewidth=2, alpha=0.6, label='Observed mean')
                ax4.plot(ell_array, unet_cross_ps_sample, 'b-', linewidth=1.5, alpha=0.8, label='UNet (this sample)')
                ax4.plot(ell_array, observed_cross_ps_sample, 'orange', linewidth=1.5, alpha=0.8, linestyle='--', label='Observed (this sample)')
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
        if observed_b_mse_array is not None and unet_recon_mse_array is not None:
            hb = ax6.hexbin(observed_b_mse_array, unet_recon_mse_array, gridsize=12, cmap='Blues', mincnt=1, linewidths=0.1, edgecolors='gray', alpha=0.8, zorder=2)
            cbar = plt.colorbar(hb, ax=ax6, fraction=0.046, pad=0.04)
            cbar.set_label('Count', fontsize=8)
            mse_min = min(observed_b_mse_array.min(), unet_recon_mse_array.min())
            mse_max = max(observed_b_mse_array.max(), unet_recon_mse_array.max())
            ax6.plot([mse_min, mse_max], [mse_min, mse_max], 'r--', linewidth=2, label='y=x', zorder=3, alpha=0.8)
            if array_idx < len(observed_b_mse_array):
                ax6.scatter(observed_b_mse_array[array_idx], unet_recon_mse_array[array_idx], s=150, c='green', marker='*', edgecolors='black', linewidths=1.5, zorder=4, label=f'Sample {sample_idx}')
            ax6.set_xlabel('Observed B MSE', fontsize=9)
            ax6.set_ylabel('UNet CMB MSE', fontsize=9)
            ax6.set_title('MSE Comparison\n(All Patches)', fontsize=10, fontweight='bold')
            ax6.set_aspect('equal', adjustable='box')
            ax6.grid(True, alpha=0.3, zorder=1)
            ax6.legend(loc='lower right', fontsize=7)
        else:
            ax6.axis('off')
            ax6.text(0.5, 0.5, f'Column 6\n(No MSE data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
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

def visualize_input_channels(b_small, t_all, e_all, sample_indices, output_dir=None, index_mapping=None, input_mode='bte'):
    if input_mode == 'te_only':
        n_cols = 2
        print(f'\nCreating input channels plots (1x2) for {len(sample_indices)} test samples (saving each separately)...')
    elif input_mode in ['e_only', 't_only']:
        n_cols = 1
        print(f'\nCreating input channels plots (1x1) for {len(sample_indices)} test samples (saving each separately)...')
    else:
        n_cols = 3
        print(f'\nCreating input channels plots (1x3) for {len(sample_indices)} test samples (saving each separately)...')
    n_samples = len(sample_indices)
    for (row_idx, sample_idx) in enumerate(sample_indices):
        if n_cols == 1:
            (fig, ax) = plt.subplots(1, 1, figsize=(5, 5))
            axes = ax
        else:
            (fig, axes) = plt.subplots(1, n_cols, figsize=(15, 5), gridspec_kw={'wspace': 0.4})
        if index_mapping is not None:
            if sample_idx not in index_mapping:
                print(f'  Skipping sample {sample_idx} (not in subsampled data)')
                plt.close()
                continue
            array_idx = index_mapping[sample_idx]
        else:
            array_idx = sample_idx
        if t_all is not None:
            t_patch = t_all[array_idx]
        if e_all is not None:
            e_patch = e_all[array_idx]
        if input_mode == 't_only':
            (vmin, vmax) = (t_patch.min(), t_patch.max())
            im = axes.imshow(t_patch, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, origin='lower')
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('$\\mu$K', fontsize=16)
            cbar.locator = MaxNLocator(nbins=6)
            cbar.ax.tick_params(labelsize=16)
            cbar.update_ticks()
            axes.set_title('Foreground: Temperature\\\\(All Scales)', fontsize=16, fontweight='bold')
            axes.set_xticks([])
            axes.set_yticks([])
        elif input_mode == 'e_only':
            (vmin, vmax) = (e_patch.min(), e_patch.max())
            im = axes.imshow(e_patch, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, origin='lower')
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('$\\mu$K', fontsize=16)
            cbar.locator = MaxNLocator(nbins=6)
            cbar.ax.tick_params(labelsize=16)
            cbar.update_ticks()
            axes.set_title('Foreground: E-mode\\\\(All Scales)', fontsize=16, fontweight='bold')
            axes.set_xticks([])
            axes.set_yticks([])
        elif input_mode == 'te_only':
            (vmin1, vmax1) = (t_patch.min(), t_patch.max())
            (vmin2, vmax2) = (e_patch.min(), e_patch.max())
            ax1 = axes[0]
            im1 = ax1.imshow(t_patch, cmap='RdYlBu_r', vmin=vmin1, vmax=vmax1, origin='lower')
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.set_label('$\\mu$K', fontsize=16)
            cbar1.locator = MaxNLocator(nbins=6)
            cbar1.ax.tick_params(labelsize=16)
            cbar1.update_ticks()
            ax1.set_title('Foreground: Temperature\\\\(All Scales)', fontsize=16, fontweight='bold')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2 = axes[1]
            im2 = ax2.imshow(e_patch, cmap='RdYlBu_r', vmin=vmin2, vmax=vmax2, origin='lower')
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.set_label('$\\mu$K', fontsize=16)
            cbar2.locator = MaxNLocator(nbins=6)
            cbar2.ax.tick_params(labelsize=14)
            cbar2.update_ticks()
            ax2.set_title('Foreground: E-mode\\\\(All Scales)', fontsize=16, fontweight='bold')
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            b_small_patch = b_small[array_idx]
            (vmin1, vmax1) = (b_small_patch.min(), b_small_patch.max())
            (vmin2, vmax2) = (t_patch.min(), t_patch.max())
            (vmin3, vmax3) = (e_patch.min(), e_patch.max())
            ax1 = axes[0]
            im1 = ax1.imshow(b_small_patch, cmap='RdYlBu_r', vmin=vmin1, vmax=vmax1, origin='lower')
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.set_label('$\\mu$K', fontsize=16)
            cbar1.locator = MaxNLocator(nbins=6)
            cbar1.ax.tick_params(labelsize=16)
            cbar1.update_ticks()
            ax1.set_title('Foreground: B-mode\\\\Small-Scales ($\\ell > 200$)', fontsize=16, fontweight='bold')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2 = axes[1]
            im2 = ax2.imshow(t_patch, cmap='RdYlBu_r', vmin=vmin2, vmax=vmax2, origin='lower')
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.set_label('$\\mu$K', fontsize=16)
            cbar2.locator = MaxNLocator(nbins=6)
            cbar2.ax.tick_params(labelsize=14)
            cbar2.update_ticks()
            ax2.set_title('Foreground: Temperature\\\\(All Scales)', fontsize=16, fontweight='bold')
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3 = axes[2]
            im3 = ax3.imshow(e_patch, cmap='RdYlBu_r', vmin=vmin3, vmax=vmax3, origin='lower')
            divider3 = make_axes_locatable(ax3)
            cax3 = divider3.append_axes('right', size='5%', pad=0.05)
            cbar3 = plt.colorbar(im3, cax=cax3)
            cbar3.set_label('$\\mu$K', fontsize=16)
            cbar3.locator = MaxNLocator(nbins=6)
            cbar3.ax.tick_params(labelsize=14)
            cbar3.update_ticks()
            ax3.set_title('Foreground: E-mode\\\\(All Scales)', fontsize=16, fontweight='bold')
            ax3.set_xticks([])
            ax3.set_yticks([])
        fig.suptitle(f'Sample {sample_idx} - Input Channels', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        if output_dir:
            input_viz_subdir = os.path.join(output_dir, 'input_channels')
            os.makedirs(input_viz_subdir, exist_ok=True)
            sample_output_path = os.path.join(input_viz_subdir, f'input_channels_sample_{sample_idx}.png')
            plt.savefig(sample_output_path, dpi=600, bbox_inches='tight')
            print(f'  Saved sample {sample_idx} to: {sample_output_path}')
        else:
            print(f'  Displaying sample {sample_idx}...')
            plt.show()
        plt.close()

def visualize_fg_reconstructions(b_fg_all_scales, targets, predictions, sample_indices, output_dir=None, ell_array=None, pred_vs_target_cross_spectra_array=None, mean_pred_vs_target_cross_ps=None, std_pred_vs_target_cross_ps=None, pred_vs_target_cross_ps_lower=None, pred_vs_target_cross_ps_upper=None, null_cross_spectra_array=None, mean_null_cross_ps_overall=None, pred_vs_target_correlations=None, null_correlations_array=None, mean_null_correlation_overall=None, target_vs_zero_mse_array=None, pred_vs_target_mse_array=None, index_mapping=None):
    print(f'\nCreating foreground reconstruction plots (2x3) for {len(sample_indices)} test samples (saving each separately)...')
    n_samples = len(sample_indices)
    n_cols = 6
    mse_hexbin_data_fg = None
    mse_extent_fg = None
    mse_vmax_fg = None
    diagonal_coords_fg = None
    if target_vs_zero_mse_array is not None and pred_vs_target_mse_array is not None:
        print('  Pre-computing foreground MSE hexbin data...')
        gridsize = 30
        mse_min = min(target_vs_zero_mse_array.min(), pred_vs_target_mse_array.min())
        mse_max = max(target_vs_zero_mse_array.max(), pred_vs_target_mse_array.max())
        mse_range = mse_max - mse_min
        mse_min = mse_min - 0.05 * mse_range
        mse_max = mse_max + 0.05 * mse_range
        x_bins = np.linspace(mse_min, mse_max, gridsize + 1)
        y_bins = np.linspace(mse_min, mse_max, gridsize + 1)
        (mse_hexbin_data_fg, x_edges, y_edges) = np.histogram2d(target_vs_zero_mse_array, pred_vs_target_mse_array, bins=[x_bins, y_bins])
        mse_hexbin_data_fg = mse_hexbin_data_fg.T
        mse_extent_fg = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        mse_vmax_fg = mse_hexbin_data_fg.max()
        diagonal_coords_fg = [-10, mse_max]
        print(f'    Hexbin data shape: {mse_hexbin_data_fg.shape}')
    for (row_idx, sample_idx) in enumerate(sample_indices):
        (fig, axes) = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'wspace': 0.27, 'hspace': 0.2})
        if index_mapping is not None:
            if sample_idx not in index_mapping:
                print(f'  Skipping sample {sample_idx} (not in subsampled data)')
                plt.close()
                continue
            array_idx = index_mapping[sample_idx]
        else:
            array_idx = sample_idx
        b_fg_all_patch = b_fg_all_scales[array_idx]
        target_patch = targets[array_idx]
        pred_patch = predictions[array_idx]
        b_small_patch = b_fg_all_patch - target_patch
        (vmin1, vmax1) = (b_small_patch.min(), b_small_patch.max())
        (vmin2, vmax2) = (target_patch.min(), target_patch.max())
        (vmin3, vmax3) = (pred_patch.min(), pred_patch.max())
        ax = axes[0, 0]
        im = ax.imshow(b_small_patch, cmap='RdYlBu_r', vmin=vmin1, vmax=vmax1, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('$\\mu$K', fontsize=16)
        cbar.locator = MaxNLocator(nbins=6)
        cbar.ax.tick_params(labelsize=16)
        cbar.update_ticks()
        ax.set_title('Foreground: B-mode\\\\Small-Scales ($\\ell > 200$)', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax2 = axes[0, 1]
        im2 = ax2.imshow(target_patch, cmap='RdYlBu_r', vmin=vmin2, vmax=vmax2, origin='lower')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('$\\mu$K', fontsize=16)
        cbar2.locator = MaxNLocator(nbins=6)
        cbar2.ax.tick_params(labelsize=14)
        cbar2.update_ticks()
        ax2.set_title('Large-Scale B-Mode\\\\Foregrounds (Target, $\\ell < 200$)', fontsize=16, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3 = axes[0, 2]
        im3 = ax3.imshow(pred_patch, cmap='RdYlBu_r', vmin=vmin3, vmax=vmax3, origin='lower')
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
        target_flat = target_patch.flatten()
        pred_flat = pred_patch.flatten()
        corr_pred_target = np.corrcoef(target_flat, pred_flat)[0, 1]
        ax4 = axes[1, 0]
        if ell_array is not None and pred_vs_target_cross_spectra_array is not None:
            if array_idx < len(pred_vs_target_cross_spectra_array):
                pred_target_cross_ps_sample = pred_vs_target_cross_spectra_array[array_idx]
                if pred_vs_target_cross_ps_lower is not None and pred_vs_target_cross_ps_upper is not None:
                    ax4.fill_between(ell_array, pred_vs_target_cross_ps_lower, pred_vs_target_cross_ps_upper, color='#f0c030', alpha=0.2, label='$\\pm 1\\sigma$')
                    if mean_pred_vs_target_cross_ps is not None:
                        ax4.plot(ell_array, mean_pred_vs_target_cross_ps, '#2e1b01', linewidth=2, alpha=0.6, label='Mean', linestyle='--')
                elif mean_pred_vs_target_cross_ps is not None and std_pred_vs_target_cross_ps is not None:
                    ax4.fill_between(ell_array, mean_pred_vs_target_cross_ps - std_pred_vs_target_cross_ps, mean_pred_vs_target_cross_ps + std_pred_vs_target_cross_ps, color='#f0c030', alpha=0.2, label='$\\pm 1\\sigma$')
                    ax4.plot(ell_array, mean_pred_vs_target_cross_ps, '#2e1b01', linewidth=2, alpha=0.6, label='Mean', linestyle='--')
                ax4.plot(ell_array, pred_target_cross_ps_sample, 'green', linewidth=1.5, alpha=0.8, label='This sample')
                if mean_null_cross_ps_overall is not None:
                    ax4.plot(ell_array, mean_null_cross_ps_overall, 'gray', linewidth=1.5, linestyle=':', alpha=0.7, label='Null (mean)')
                if ell_array is not None and len(ell_array) > 0:
                    ell_min = ell_array[0]
                    ell_max = ell_array[-1]
                    philcox_mask = (PHILCOX_ELL >= ell_min) & (PHILCOX_ELL <= ell_max)
                    if np.any(philcox_mask):
                        ax4.errorbar(PHILCOX_ELL[philcox_mask], PHILCOX_Y[philcox_mask], yerr=PHILCOX_YERR[philcox_mask], fmt='o', markersize=7, capsize=2, capthick=1, elinewidth=1, label='Philcox+18', zorder=10, alpha=0.7, markerfacecolor='lightblue', markeredgecolor='black', ecolor='black')
                ax4.set_xlabel('$\\ell$ (Multipole)', fontsize=16)
                ax4.set_ylabel('$\\tilde{C}_\\ell^{True,Pred}$', fontsize=16)
                ax4.set_ylim([-0.1, 1.5])
                ax4.set_title('Normalized Cross-Power Spectra', fontsize=16, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.legend(fontsize=16, loc='best', ncol=2)
                ax4.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax4.yaxis.set_major_locator(MaxNLocator(nbins=6))
            else:
                ax4.axis('off')
                ax4.text(0.5, 0.5, f'Sample {sample_idx}\nNo cross-spectrum data', ha='center', va='center', fontsize=10)
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, f'Column 4\n(No cross-spectrum data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax5 = axes[1, 1]
        if null_correlations_array is not None and pred_vs_target_correlations is not None:
            if array_idx < len(pred_vs_target_correlations):
                actual_corr = pred_vs_target_correlations[array_idx]
                bins = np.linspace(-0.2, 1.0, 30)
                ax5.hist(null_correlations_array, bins=bins, alpha=0.7, color='lightblue', edgecolor='#143d80')
                if mean_null_correlation_overall is not None:
                    ax5.axvline(mean_null_correlation_overall, color='#143d80', linestyle='--', linewidth=3, label='Null mean')
                ax5.axvline(actual_corr, color='green', linestyle='-', linewidth=3, label='This sample')
                ax5.set_xlabel('Spatial Correlation', fontsize=16)
                ax5.set_ylabel('Frequency', fontsize=16)
                ax5.set_title('Null Correlation Test', fontsize=16, fontweight='bold')
                xlim = ax5.get_xlim()
                ylim = ax5.get_ylim()
                x_span = xlim[1] - xlim[0]
                y_span = ylim[1] - ylim[0]
                if pred_vs_target_correlations is not None and len(pred_vs_target_correlations) > 0:
                    mean_corr = np.mean(pred_vs_target_correlations)
                    ax5.text(0.4, 0.2, f'This sample: {actual_corr:.3f}\nMean correlation: {mean_corr:.3f}', transform=ax5.transAxes, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=16)
                ax5.legend(fontsize=16, loc='best')
            else:
                ax5.axis('off')
                ax5.text(0.5, 0.5, f'Sample {sample_idx}\nNo correlation data', ha='center', va='center', fontsize=10)
        else:
            ax5.axis('off')
            ax5.text(0.5, 0.5, f'Column 5\n(No correlation data)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax6 = axes[1, 2]
        if mse_hexbin_data_fg is not None:
            im = ax6.imshow(mse_hexbin_data_fg, origin='lower', aspect='auto', extent=mse_extent_fg, cmap='Blues', vmin=0, vmax=mse_vmax_fg, interpolation='nearest', zorder=0)
            cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
            cbar.set_label('Count', fontsize=8)
            ax6.plot(diagonal_coords_fg, diagonal_coords_fg, 'r--', linewidth=2, label='y=x', zorder=3, alpha=0.8)
            if array_idx < len(target_vs_zero_mse_array):
                ax6.scatter(target_vs_zero_mse_array[array_idx], pred_vs_target_mse_array[array_idx], s=150, c='green', marker='o', edgecolors='lightgray', linewidths=1.5, zorder=4, label=f'Sample {sample_idx}')
            ax6.set_xlabel('Uncleaned MSE', fontsize=16)
            ax6.set_ylabel('UNet MSE', fontsize=16)
            ax6.set_title('MSE Comparison\n(All Patches)', fontsize=16, fontweight='bold')
            ax6.set_ylim([-0.001, 0.002])
            ax6.set_xlim([-0.001, 0.004])
            ax6.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
            ax6.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax6.grid(True, alpha=0.3, zorder=1)
            ax6.legend(loc='lower right', fontsize=16)
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

def main():
    parser = argparse.ArgumentParser(description='Evaluate UNet predictions on test data (single-frequency model)')
    parser.add_argument('--normalize', action='store_true', help='Use normalized model (default: unnormalized)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results (default: same as model dir)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model checkpoint (default: auto-detect from output dir)')
    parser.add_argument('--b-only', action='store_true', help='Use B-only model (1 channel), skip T and E channels')
    parser.add_argument('--t-only', action='store_true', help='Use only T channel (1 channel), skip B small-scale and E channels')
    parser.add_argument('--e-only', action='store_true', help='Use only E channel (1 channel), skip B small-scale and T channels')
    parser.add_argument('--te-only', action='store_true', help='Use only T and E channels (2 channels), skip B small-scale channel')
    parser.add_argument('--te', action='store_true', help='Use T and E channels in addition to B small-scale (3 channels, default)')
    parser.add_argument('--sample-indices', type=str, default=None, help='Comma-separated list of sample indices to visualize (e.g., "0,5,10,15"). Default: first 50 samples')
    parser.add_argument('--n-subsample', type=int, default=0, help='Number of test patches to subsample for analysis (default: 0, uses all test patches). Set to a positive integer to subsample that many patches.')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization (only save data)')
    parser.add_argument('--save-loaded-test-data', type=str, default=None, metavar='PATH', help='Save test tensors right after loading (before inference) as a compressed .npz for archiving or offline use')
    parser.add_argument('--export-loaded-test-only', action='store_true', help='Load data, write --save-loaded-test-data, then exit (no checkpoint, no inference)')
    parser.add_argument('--test-data-npz', type=str, default=None, metavar='PATH', help='Load test tensors from a compressed .npz (e.g. exported teb bundle); use when evaluating without the full ILC_ML .npy tree')
    args = parser.parse_args()
    if args.export_loaded_test_only and (not args.save_loaded_test_data):
        parser.error('--export-loaded-test-only requires --save-loaded-test-data PATH')
    if args.test_data_npz and args.export_loaded_test_only:
        parser.error('--export-loaded-test-only cannot be used with --test-data-npz')
    if args.test_data_npz and args.save_loaded_test_data:
        parser.error('--save-loaded-test-data cannot be used with --test-data-npz (export requires the on-disk patch files)')
    if args.b_only:
        input_mode = 'b_only'
        channel_suffix = 'b_only'
        channel_desc = 'B small-scale only (1 channel)'
        in_channels = 1
    elif args.t_only:
        input_mode = 't_only'
        channel_suffix = 't_only'
        channel_desc = 'T only (1 channel)'
        in_channels = 1
    elif args.e_only:
        input_mode = 'e_only'
        channel_suffix = 'e_only'
        channel_desc = 'E only (1 channel)'
        in_channels = 1
    elif args.te_only:
        input_mode = 'te_only'
        channel_suffix = 'te_only'
        channel_desc = 'T + E only (2 channels)'
        in_channels = 2
    else:
        input_mode = 'bte'
        channel_suffix = 'teb'
        channel_desc = 'B small-scale + T + E (3 channels)'
        in_channels = 3
    norm_suffix = 'normalized' if args.normalize else 'unnormalized'
    print(f'\nConfiguration:')
    print(f'  Normalization: {norm_suffix}')
    print(f'  Input channels: {channel_desc}')
    print(f'  Number of channels: {in_channels}')
    n_subsample_arg = args.n_subsample if args.n_subsample > 0 else None
    if n_subsample_arg is not None:
        print(f'  Subsample size: {n_subsample_arg} patches')
    else:
        print(f'  Subsample size: All test patches (no subsampling)')
    npz_bundle = None
    if args.test_data_npz:
        (test_X, test_y, test_b_all_scales, test_cmb_draw, test_indices_final, norm_stats, npz_bundle) = load_eval_from_npz_singlefreq(args.test_data_npz, input_mode, n_subsample_arg, args.normalize)
    else:
        (test_X, test_y, test_b_all_scales, test_cmb_draw, test_indices_final, norm_stats) = load_test_data(normalize=args.normalize, input_mode=input_mode, n_subsample=n_subsample_arg)
    if args.save_loaded_test_data:
        _save_loaded_test_tensors(args.save_loaded_test_data, test_X, test_y, test_b_all_scales, test_cmb_draw, test_indices_final, norm_stats, input_mode, args.normalize, in_channels, B_SMALL_DIR, FREQUENCY)
    if args.export_loaded_test_only:
        print('\n[ok] --export-loaded-test-only: skipping model load and evaluation.')
        return
    subsample_index_map = {test_indices_final[i]: i for i in range(len(test_indices_final))}
    print(f'\n  Created index mapping for {len(subsample_index_map)} subsampled patches')
    print(f'    Original dataset indices mapped to subsampled positions (0 to {len(test_indices_final) - 1})')
    if args.model_path:
        model_path = args.model_path
    else:
        model_dir = os.path.join(OUTPUT_DIR, f'{norm_suffix}_{channel_suffix}')
        stats = np.load(STATS_FILE)
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
    print('\nComputing unet_cmb_reconstructed = observed_B - b_small - unet_predicted_b_large...')
    if npz_bundle is not None:
        test_b_small = npz_bundle['test_b_small']
        test_t_all = npz_bundle['t_all'] if input_mode in ('bte', 'te_only', 't_only') else None
        test_e_all = npz_bundle['e_all'] if input_mode in ('bte', 'te_only', 'e_only') else None
        if test_t_all is not None:
            print(f'  test_t_all shape (from npz): {test_t_all.shape}')
        if test_e_all is not None:
            print(f'  test_e_all shape (from npz): {test_e_all.shape}')
    else:
        b_small_all = np.load(f'{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_B_patches_small.npy')
        test_b_small = b_small_all[test_indices_final]
        test_t_all = None
        test_e_all = None
        if input_mode in ['bte', 'te_only', 'e_only', 't_only']:
            if input_mode in ['bte', 'te_only', 't_only']:
                print('\nLoading T test data for visualization...')
                t_all_full = np.load(f'{T_E_DIR}/sim1-150_freq{FREQUENCY}_T_patches.npy')
                test_t_all = t_all_full[test_indices_final]
                print(f'  test_t_all shape: {test_t_all.shape}')
            if input_mode in ['bte', 'te_only', 'e_only']:
                print('\nLoading E test data for visualization...')
                e_all_full = np.load(f'{T_E_DIR}/sim1-150_freq{FREQUENCY}_E_patches.npy')
                test_e_all = e_all_full[test_indices_final]
                print(f'  test_e_all shape: {test_e_all.shape}')
    unet_cmb_reconstructed = test_b_all_scales - test_b_small - predictions
    print(f'  unet_cmb_reconstructed shape: {unet_cmb_reconstructed.shape}')
    print(f'  Formula: unet_cmb_reconstructed = observed_b_all_scales - b_small - unet_predicted_b_large')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(OUTPUT_DIR, f'{norm_suffix}_{channel_suffix}')
    os.makedirs(output_dir, exist_ok=True)
    n_subsample = len(test_cmb_draw)
    unet_cross_spectra_list = []
    pred_vs_target_cross_spectra_list = []
    ell_array_ref = None
    if calculate_2d_spectrum is not None:
        print('\n' + '=' * 80)
        print('COMPUTING CROSS-POWER SPECTRA')
        print('=' * 80)
        print(f'\nComputing cross-power spectra for {n_subsample} subsampled patches...')
        print(f'  Parameters: delta_ell={DELTA_ELL}, ell_max={ELL_CUTOFF}, pix_size={PIX_SIZE}, N={PATCH_SIZE}')
        cross_spectra_success_mask = []
        temp_ell_array_ref = None
        for patch_idx in tqdm(range(n_subsample), desc='Cross-spectra'):
            try:
                pure_cmb_patch = test_cmb_draw[patch_idx]
                unet_cmb_patch = unet_cmb_reconstructed[patch_idx]
                pred_patch = predictions[patch_idx]
                target_patch = targets[patch_idx]
                (ell_array, unet_cross_ps) = compute_normalized_cross_spectrum(pure_cmb_patch, unet_cmb_patch)
                unet_cross_spectra_list.append(unet_cross_ps)
                (_, pred_vs_target_cross_ps) = compute_normalized_cross_spectrum(pred_patch, target_patch)
                pred_vs_target_cross_spectra_list.append(pred_vs_target_cross_ps)
                if ell_array_ref is None:
                    ell_array_ref = ell_array.copy()
                    temp_ell_array_ref = ell_array.copy()
                elif temp_ell_array_ref is None:
                    temp_ell_array_ref = ell_array.copy()
                cross_spectra_success_mask.append(True)
            except Exception as e:
                print(f'    Warning: Error computing cross-spectrum for patch {patch_idx}: {e}')
                if temp_ell_array_ref is not None:
                    nan_array = np.full_like(temp_ell_array_ref, np.nan)
                else:
                    estimated_nbins = int(ELL_CUTOFF / DELTA_ELL)
                    nan_array = np.full(estimated_nbins, np.nan, dtype=float)
                unet_cross_spectra_list.append(nan_array.copy())
                pred_vs_target_cross_spectra_list.append(nan_array.copy())
                cross_spectra_success_mask.append(False)
        unet_cross_spectra_array = np.array(unet_cross_spectra_list)
        observed_cross_spectra_array = None
        pred_vs_target_cross_spectra_array = np.array(pred_vs_target_cross_spectra_list)
        cross_spectra_success_mask = np.array(cross_spectra_success_mask)
        mean_unet_cross_ps = np.nanmean(unet_cross_spectra_array, axis=0)
        std_unet_cross_ps = np.nanstd(unet_cross_spectra_array, axis=0)
        unet_cross_ps_lower = np.nanpercentile(unet_cross_spectra_array, 16, axis=0)
        unet_cross_ps_upper = np.nanpercentile(unet_cross_spectra_array, 84, axis=0)
        mean_observed_cross_ps = None
        std_observed_cross_ps = None
        observed_cross_ps_lower = None
        observed_cross_ps_upper = None
        mean_pred_vs_target_cross_ps = np.nanmean(pred_vs_target_cross_spectra_array, axis=0)
        std_pred_vs_target_cross_ps = np.nanstd(pred_vs_target_cross_spectra_array, axis=0)
        pred_vs_target_cross_ps_lower = np.nanpercentile(pred_vs_target_cross_spectra_array, 16, axis=0)
        pred_vs_target_cross_ps_upper = np.nanpercentile(pred_vs_target_cross_spectra_array, 84, axis=0)
        print(f'\n  [ok] Computed 2-tailed percentiles for prediction vs target cross-spectra')
        print(f'    Lower (16th percentile) shape: {pred_vs_target_cross_ps_lower.shape}')
        print(f'    Upper (84th percentile) shape: {pred_vs_target_cross_ps_upper.shape}')
        print(f'    Lower values: {pred_vs_target_cross_ps_lower}')
        print(f'    Upper values: {pred_vs_target_cross_ps_upper}')
        n_successful = cross_spectra_success_mask.sum()
        print(f'\n  [ok] Computed cross-spectra for {n_subsample} patches')
        print(f'    Successfully computed: {n_successful}/{n_subsample} patches')
        print(f'    UNet CMB cross-spectra: {unet_cross_spectra_array.shape}')
        print(f'    Prediction vs Target cross-spectra: {pred_vs_target_cross_spectra_array.shape}')
        print('\n' + '=' * 80)
        print('COMPUTING NULL CROSS-SPECTRA FOR CMB RECONSTRUCTIONS')
        print('=' * 80)

        def compute_null_cross_spectra_cmb(cmb_patches, unet_cmb_patches, ell_array_ref, n_null=100):
            n_patches = len(cmb_patches)
            all_null_cross_spectra = []
            print(f'\nComputing null cross-spectra for {n_patches} CMB reconstruction patches...')
            print(f'  For each patch: computing cross-spectra with {min(n_null, n_patches - 1)} random CMB patches (excluding its own)')
            for i in tqdm(range(n_patches), desc='Null cross-spectra (CMB)'):
                try:
                    unet_cmb_patch = unet_cmb_patches[i]
                    available_indices = [j for j in range(n_patches) if j != i]
                    n_samples = min(n_null, len(available_indices))
                    random_indices = np.random.choice(available_indices, size=n_samples, replace=False)
                    for j in random_indices:
                        try:
                            cmb_patch = cmb_patches[j]
                            (_, null_cross_ps) = compute_normalized_cross_spectrum(cmb_patch, unet_cmb_patch, delta_ell=DELTA_ELL, ell_max=ELL_CUTOFF, pix_size=PIX_SIZE, N=PATCH_SIZE)
                            if null_cross_ps is not None and (not np.all(np.isnan(null_cross_ps))):
                                all_null_cross_spectra.append(null_cross_ps)
                        except Exception as e:
                            continue
                except Exception as e:
                    print(f'    Warning: Error computing null cross-spectrum for patch {i}: {e}')
                    continue
            if len(all_null_cross_spectra) > 0:
                return np.array(all_null_cross_spectra)
            else:
                return None
        null_cross_spectra_array = compute_null_cross_spectra_cmb(test_cmb_draw, unet_cmb_reconstructed, ell_array_ref, n_null=100)
        if null_cross_spectra_array is not None:
            mean_null_cross_ps_overall = np.nanmean(null_cross_spectra_array, axis=0)
            print(f'\n  [ok] Computed null cross-spectra distribution')
            print(f'    Total null cross-spectra: {len(null_cross_spectra_array)} (distribution size)')
            print(f'    Null cross-spectra shape: {null_cross_spectra_array.shape}')
            print(f'    Mean null cross-spectrum shape: {mean_null_cross_ps_overall.shape}')
        else:
            mean_null_cross_ps_overall = None
            print(f'\n  WARNING: Warning: No null cross-spectra computed')
    else:
        print('\nSkipping cross-power spectra (calculate_2d_spectrum not available)')
        unet_cross_spectra_array = None
        observed_cross_spectra_array = None
        pred_vs_target_cross_spectra_array = None
        mean_unet_cross_ps = None
        std_unet_cross_ps = None
        unet_cross_ps_lower = None
        unet_cross_ps_upper = None
        mean_observed_cross_ps = None
        std_observed_cross_ps = None
        observed_cross_ps_lower = None
        observed_cross_ps_upper = None
        mean_pred_vs_target_cross_ps = None
        std_pred_vs_target_cross_ps = None
        pred_vs_target_cross_ps_lower = None
        pred_vs_target_cross_ps_upper = None
        ell_array_ref = None
        null_cross_spectra_array = None
        mean_null_cross_ps_overall = None
    print('\n' + '=' * 80)
    print('COMPUTING SPATIAL PEARSON CORRELATIONS')
    print('=' * 80)
    unet_spatial_correlations = []
    pred_vs_target_correlations = []
    for patch_idx in tqdm(range(n_subsample), desc='Spatial correlations'):
        try:
            pure_cmb_patch = test_cmb_draw[patch_idx]
            unet_cmb_patch = unet_cmb_reconstructed[patch_idx]
            pred_patch = predictions[patch_idx]
            target_patch = targets[patch_idx]
            pure_flat = pure_cmb_patch.flatten()
            unet_flat = unet_cmb_patch.flatten()
            pred_flat = pred_patch.flatten()
            target_flat = target_patch.flatten()
            corr_matrix_unet = np.corrcoef(pure_flat, unet_flat)
            unet_corr = corr_matrix_unet[0, 1]
            if np.isnan(unet_corr):
                unet_corr = 0.0
            unet_spatial_correlations.append(unet_corr)
            corr_matrix_pred_target = np.corrcoef(pred_flat, target_flat)
            pred_target_corr = corr_matrix_pred_target[0, 1]
            if np.isnan(pred_target_corr):
                pred_target_corr = 0.0
            pred_vs_target_correlations.append(pred_target_corr)
        except Exception as e:
            print(f'    Warning: Error computing spatial correlation for patch {patch_idx}: {e}')
            unet_spatial_correlations.append(0.0)
            pred_vs_target_correlations.append(0.0)
    unet_spatial_correlations = np.array(unet_spatial_correlations)
    observed_spatial_correlations = None
    pred_vs_target_correlations = np.array(pred_vs_target_correlations)
    print(f'\n  [ok] Computed spatial correlations for {len(unet_spatial_correlations)} patches')
    print(f'    UNet CMB spatial correlations (unet_cmb_reconstructed vs pure_cmb):')
    print(f'      Mean: {np.mean(unet_spatial_correlations):.4f}, Std: {np.std(unet_spatial_correlations):.4f}')
    print(f'    Prediction vs Target correlations (prediction vs target, large-scale B-modes):')
    print(f'      Mean: {np.mean(pred_vs_target_correlations):.4f}, Std: {np.std(pred_vs_target_correlations):.4f}')
    print('\n' + '=' * 80)
    print('COMPUTING NULL CORRELATIONS FOR FOREGROUND PREDICTIONS')
    print('=' * 80)

    def compute_null_correlations_fg(pred_patches, target_patches, n_null=100):
        n_patches = len(pred_patches)
        all_null_corrs = []
        for i in tqdm(range(n_patches), desc='Null correlations (FG)'):
            try:
                pred_flat = pred_patches[i].flatten()
                available_indices = [j for j in range(n_patches) if j != i]
                n_samples = min(n_null, len(available_indices))
                random_indices = np.random.choice(available_indices, size=n_samples, replace=False)
                for j in random_indices:
                    try:
                        target_flat = target_patches[j].flatten()
                        corr = np.corrcoef(pred_flat, target_flat)[0, 1]
                        if not np.isnan(corr):
                            all_null_corrs.append(corr)
                    except Exception:
                        continue
            except Exception as e:
                print(f'    Warning: Error computing null correlation for patch {i}: {e}')
                continue
        return np.array(all_null_corrs)
    print(f'\nComputing null correlations for {n_subsample} foreground prediction patches...')
    print(f'  For each patch: correlating with {min(100, n_subsample - 1)} random target patches (excluding its own)')
    fg_null_correlations_array = compute_null_correlations_fg(predictions, targets, n_null=100)
    fg_mean_null_correlation_overall = np.mean(fg_null_correlations_array)
    print(f'\n  [ok] Computed null correlations distribution')
    print(f'    Total null correlations: {len(fg_null_correlations_array)} (distribution size)')
    print(f'    Mean null correlation (overall): {fg_mean_null_correlation_overall:.6f} (should be close to 0)')
    print(f'    Std of null correlations: {np.std(fg_null_correlations_array):.6f}')
    print('\n' + '=' * 80)
    print('COMPUTING NULL CORRELATIONS FOR CMB RECONSTRUCTIONS')
    print('=' * 80)

    def compute_null_correlations_cmb(cmb_patches, unet_cmb_patches, n_null=100):
        n_patches = len(cmb_patches)
        all_null_corrs = []
        for i in tqdm(range(n_patches), desc='Null correlations (CMB)'):
            try:
                unet_flat = unet_cmb_patches[i].flatten()
                available_indices = [j for j in range(n_patches) if j != i]
                n_samples = min(n_null, len(available_indices))
                random_indices = np.random.choice(available_indices, size=n_samples, replace=False)
                for j in random_indices:
                    try:
                        cmb_flat = cmb_patches[j].flatten()
                        corr = np.corrcoef(unet_flat, cmb_flat)[0, 1]
                        if not np.isnan(corr):
                            all_null_corrs.append(corr)
                    except Exception:
                        continue
            except Exception as e:
                print(f'    Warning: Error computing null correlation for patch {i}: {e}')
                continue
        return np.array(all_null_corrs)
    print(f'\nComputing null correlations for {n_subsample} CMB reconstruction patches...')
    print(f'  For each patch: correlating with {min(100, n_subsample - 1)} random CMB patches (excluding its own)')
    null_correlations_array = compute_null_correlations_cmb(test_cmb_draw, unet_cmb_reconstructed, n_null=100)
    mean_null_correlation_overall = np.mean(null_correlations_array)
    print(f'\n  [ok] Computed null correlations distribution')
    print(f'    Total null correlations: {len(null_correlations_array)} (distribution size)')
    print(f'    Mean null correlation (overall): {mean_null_correlation_overall:.6f} (should be close to 0)')
    print(f'    Std of null correlations: {np.std(null_correlations_array):.6f}')
    print('\n' + '=' * 80)
    print('COMPUTING MSE FOR EACH PATCH')
    print('=' * 80)
    observed_b_mse_list = []
    unet_recon_mse_list = []
    pred_vs_target_mse_list = []
    target_vs_zero_mse_list = []
    if npz_bundle is not None:
        test_b_large = np.asarray(targets)
    else:
        b_small_all = np.load(f'{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_B_patches_small.npy')
        test_b_small = b_small_all[test_indices_final]
        b_large_all = np.load(f'{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_B_patches_large.npy')
        test_b_large = b_large_all[test_indices_final]
    for patch_idx in tqdm(range(n_subsample), desc='MSE computation'):
        try:
            pure_cmb_patch = test_cmb_draw[patch_idx]
            observed_b_patch = test_b_all_scales[patch_idx]
            unet_cmb_patch = unet_cmb_reconstructed[patch_idx]
            pred_patch = predictions[patch_idx]
            target_patch = targets[patch_idx]
            observed_mse = compute_patch_mse(pure_cmb_patch, observed_b_patch)
            if observed_mse is not None:
                observed_b_mse_list.append(observed_mse)
            else:
                observed_b_mse_list.append(0.0)
            unet_mse = compute_patch_mse(pure_cmb_patch, unet_cmb_patch)
            if unet_mse is not None:
                unet_recon_mse_list.append(unet_mse)
            else:
                unet_recon_mse_list.append(0.0)
            pred_target_mse = compute_patch_mse(pred_patch, target_patch)
            if pred_target_mse is not None:
                pred_vs_target_mse_list.append(pred_target_mse)
            else:
                pred_vs_target_mse_list.append(0.0)
            zero_patch = np.zeros_like(target_patch)
            target_zero_mse = compute_patch_mse(target_patch, zero_patch)
            if target_zero_mse is not None:
                target_vs_zero_mse_list.append(target_zero_mse)
            else:
                target_vs_zero_mse_list.append(0.0)
        except Exception as e:
            print(f'    Warning: Error computing MSE for patch {patch_idx}: {e}')
            observed_b_mse_list.append(0.0)
            unet_recon_mse_list.append(0.0)
            pred_vs_target_mse_list.append(0.0)
            target_vs_zero_mse_list.append(0.0)
    observed_b_mse_array = np.array(observed_b_mse_list)
    unet_recon_mse_array = np.array(unet_recon_mse_list)
    pred_vs_target_mse_array = np.array(pred_vs_target_mse_list)
    target_vs_zero_mse_array = np.array(target_vs_zero_mse_list)
    print(f'\n  [ok] Computed MSEs for {len(observed_b_mse_list)} patches')
    print(f'    Observed B MSE - Mean: {np.mean(observed_b_mse_array):.6e}, Std: {np.std(observed_b_mse_array):.6e}')
    print(f'    UNet recon MSE - Mean: {np.mean(unet_recon_mse_array):.6e}, Std: {np.std(unet_recon_mse_array):.6e}')
    if np.mean(unet_recon_mse_array) > 0:
        print(f'    Improvement ratio (Observed/UNet): {np.mean(observed_b_mse_array) / np.mean(unet_recon_mse_array):.4f}x')
    print(f'\n  Foreground Prediction MSEs:')
    print(f'    MSE(prediction, target) - Mean: {np.mean(pred_vs_target_mse_array):.6e}, Std: {np.std(pred_vs_target_mse_array):.6e}')
    print(f'    MSE(target, zero) - Mean: {np.mean(target_vs_zero_mse_array):.6e}, Std: {np.std(target_vs_zero_mse_array):.6e}')
    if np.mean(target_vs_zero_mse_array) > 0:
        print(f'    Improvement ratio (target_vs_zero / pred_vs_target): {np.mean(target_vs_zero_mse_array) / np.mean(pred_vs_target_mse_array):.4f}x')
        if np.mean(pred_vs_target_mse_array) < np.mean(target_vs_zero_mse_array):
            print(f'    [ok] UNet prediction is better than zero (MSE(pred, target) < MSE(target, zero))')
        else:
            print(f'    WARNING: WARNING: UNet prediction is worse than zero (MSE(pred, target) > MSE(target, zero))')
    print('\n' + '=' * 80)
    print('SAVING TEST DATA FOR ANALYSIS')
    print('=' * 80)
    output_file = os.path.join(output_dir, 'test_results.npz')
    print(f'\nSaving test results to: {output_file}')
    np.savez(output_file, unet_predictions=predictions, unet_targets=targets, observed_b_all_scales=test_b_all_scales, cmb_draw=test_cmb_draw, unet_cmb_reconstructed=unet_cmb_reconstructed, test_b_small=test_b_small, unet_spatial_correlations=unet_spatial_correlations, pred_vs_target_correlations=pred_vs_target_correlations, null_correlations_array=null_correlations_array, fg_null_correlations_array=fg_null_correlations_array, unet_cross_spectra_array=unet_cross_spectra_array, pred_vs_target_cross_spectra_array=pred_vs_target_cross_spectra_array, ell_array=ell_array_ref, pred_vs_target_mse_array=pred_vs_target_mse_array, target_vs_zero_mse_array=target_vs_zero_mse_array, b_fg_all_scales=test_b_small + test_b_large, test_indices=test_indices_final, normalize=args.normalize, in_channels=in_channels, channel_mode=channel_suffix)
    print('  [ok] Test results saved!')
    print('\n' + '=' * 80)
    print('SAVING FIRST SAMPLE DATA FOR NOTEBOOK REPRODUCTION')
    print('=' * 80)
    first_sample_idx = test_indices_final[0]
    first_sample_array_idx = 0
    print(f'\nExtracting data for first sample:')
    print(f'  Original dataset index: {first_sample_idx}')
    print(f'  Array position: {first_sample_array_idx}')
    test_b_fg_all_scales = test_b_small + test_b_large
    first_sample_data = {'sample_idx': first_sample_idx, 'array_idx': first_sample_array_idx, 'cmb_draw': test_cmb_draw[first_sample_array_idx], 'observed_b_all_scales': test_b_all_scales[first_sample_array_idx], 'unet_cmb_reconstructed': unet_cmb_reconstructed[first_sample_array_idx], 'b_fg_all_scales': test_b_fg_all_scales[first_sample_array_idx], 'target': targets[first_sample_array_idx], 'prediction': predictions[first_sample_array_idx], 'b_small': test_b_small[first_sample_array_idx], 'unet_spatial_correlation': unet_spatial_correlations[first_sample_array_idx], 'pred_vs_target_correlation': pred_vs_target_correlations[first_sample_array_idx], 'observed_b_mse': observed_b_mse_array[first_sample_array_idx], 'unet_recon_mse': unet_recon_mse_array[first_sample_array_idx], 'pred_vs_target_mse': pred_vs_target_mse_array[first_sample_array_idx], 'target_vs_zero_mse': target_vs_zero_mse_array[first_sample_array_idx], 'ell_array': ell_array_ref}
    if unet_cross_spectra_array is not None:
        first_sample_data['unet_cross_spectrum'] = unet_cross_spectra_array[first_sample_array_idx]
        first_sample_data['mean_unet_cross_ps'] = mean_unet_cross_ps
        first_sample_data['std_unet_cross_ps'] = std_unet_cross_ps
    if pred_vs_target_cross_spectra_array is not None:
        first_sample_data['pred_vs_target_cross_spectrum'] = pred_vs_target_cross_spectra_array[first_sample_array_idx]
        first_sample_data['mean_pred_vs_target_cross_ps'] = mean_pred_vs_target_cross_ps
        first_sample_data['std_pred_vs_target_cross_ps'] = std_pred_vs_target_cross_ps
    if null_cross_spectra_array is not None:
        first_sample_data['null_cross_spectra_array'] = null_cross_spectra_array
        first_sample_data['mean_null_cross_ps_overall'] = mean_null_cross_ps_overall
    if null_correlations_array is not None:
        first_sample_data['null_correlations_array'] = null_correlations_array
        first_sample_data['mean_null_correlation_overall'] = mean_null_correlation_overall
    if fg_null_correlations_array is not None:
        first_sample_data['fg_null_correlations_array'] = fg_null_correlations_array
        first_sample_data['fg_mean_null_correlation_overall'] = fg_mean_null_correlation_overall
    first_sample_data['observed_b_mse_array'] = observed_b_mse_array
    first_sample_data['unet_recon_mse_array'] = unet_recon_mse_array
    first_sample_data['pred_vs_target_mse_array'] = pred_vs_target_mse_array
    first_sample_data['target_vs_zero_mse_array'] = target_vs_zero_mse_array
    first_sample_file = os.path.join(output_dir, 'first_sample_data.npz')
    np.savez(first_sample_file, **first_sample_data)
    print(f'\n  [ok] Saved first sample data to: {first_sample_file}')
    print(f'    This file contains all data needed to reproduce plots in a Jupyter notebook')
    if not args.no_viz:
        if args.sample_indices:
            sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
            sample_indices = [idx for idx in sample_indices if idx in subsample_index_map]
            if len(sample_indices) == 0:
                print(f'\nWarning: No valid sample indices in subsampled data. Using first 50 subsampled patches.')
                sample_indices = list(test_indices_final[:min(50, len(test_indices_final))])
        else:
            sample_indices = list(test_indices_final[:min(50, len(test_indices_final))])
        print(f'\nVisualizing samples: {sample_indices}')
        print(f'  These are original dataset indices. Mapped to subsampled positions: {[subsample_index_map[idx] for idx in sample_indices]}')
        missing_indices = [idx for idx in sample_indices if idx not in subsample_index_map]
        if missing_indices:
            print(f'\n  WARNING: WARNING: The following sample indices are not in the subsampled data: {missing_indices}')
            print(f'     Available indices: {list(subsample_index_map.keys())[:10]}...' if len(subsample_index_map) > 10 else f'     Available indices: {list(subsample_index_map.keys())}')
            sample_indices = [idx for idx in sample_indices if idx in subsample_index_map]
            if len(sample_indices) == 0:
                print(f'  ERROR: No valid sample indices remaining. Using first 50 from subsampled data.')
                sample_indices = list(test_indices_final[:min(50, len(test_indices_final))])
        visualize_cmb_reconstructions(test_cmb_draw, test_b_all_scales, unet_cmb_reconstructed, sample_indices, output_dir=output_dir, ell_array=ell_array_ref, unet_cross_spectra_array=unet_cross_spectra_array, observed_cross_spectra_array=observed_cross_spectra_array, mean_unet_cross_ps=mean_unet_cross_ps, std_unet_cross_ps=std_unet_cross_ps, unet_cross_ps_lower=unet_cross_ps_lower, unet_cross_ps_upper=unet_cross_ps_upper, mean_observed_cross_ps=mean_observed_cross_ps, std_observed_cross_ps=std_observed_cross_ps, observed_cross_ps_lower=observed_cross_ps_lower, observed_cross_ps_upper=observed_cross_ps_upper, null_cross_spectra_array=null_cross_spectra_array, mean_null_cross_ps_overall=mean_null_cross_ps_overall, unet_spatial_correlations=unet_spatial_correlations, null_correlations_array=null_correlations_array, mean_null_correlation_overall=mean_null_correlation_overall, observed_b_mse_array=observed_b_mse_array, unet_recon_mse_array=unet_recon_mse_array, index_mapping=subsample_index_map)
        visualize_fg_reconstructions(test_b_fg_all_scales, targets, predictions, sample_indices, output_dir=output_dir, ell_array=ell_array_ref, pred_vs_target_cross_spectra_array=pred_vs_target_cross_spectra_array, mean_pred_vs_target_cross_ps=mean_pred_vs_target_cross_ps, std_pred_vs_target_cross_ps=std_pred_vs_target_cross_ps, pred_vs_target_cross_ps_lower=pred_vs_target_cross_ps_lower, pred_vs_target_cross_ps_upper=pred_vs_target_cross_ps_upper, null_cross_spectra_array=None, mean_null_cross_ps_overall=None, pred_vs_target_correlations=pred_vs_target_correlations, null_correlations_array=fg_null_correlations_array, mean_null_correlation_overall=fg_mean_null_correlation_overall, target_vs_zero_mse_array=target_vs_zero_mse_array, pred_vs_target_mse_array=pred_vs_target_mse_array, index_mapping=subsample_index_map)
        if input_mode in ['bte', 'te_only', 'e_only', 't_only']:
            if input_mode in ['bte', 'te_only', 'e_only'] and test_e_all is not None or (input_mode == 't_only' and test_t_all is not None):
                visualize_input_channels(test_b_small if input_mode == 'bte' else None, test_t_all, test_e_all, sample_indices, output_dir=output_dir, index_mapping=subsample_index_map, input_mode=input_mode)
    print('\n' + '=' * 80)
    print('SUMMARY STATISTICS')
    print('=' * 80)
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    mse = np.mean((target_flat - pred_flat) ** 2)
    mae = np.mean(np.abs(target_flat - pred_flat))
    corr = np.corrcoef(target_flat, pred_flat)[0, 1]
    print(f'\nUNet Large-Scale Foreground: B-mode Prediction Metrics (overall flattened correlation - all pixels across all patches):')
    print(f'  MSE: {mse:.3e}')
    print(f'  MAE: {mae:.3e}')
    print(f'  Correlation (overall): {corr:.3f}  # Single correlation computed across ALL pixels in ALL patches')
    print(f'  # Targets are large-scale Foreground: B-modes (ell < cutoff), not CMB')
    print(f'\nDataset Statistics:')
    print(f'  Number of test samples: {len(predictions)}')
    print(f'  Spatial dimensions: {predictions.shape[1:]} (H, W)')
    print(f'\n  UNet predictions (large-scale Foreground: B-modes) - Mean: {pred_flat.mean():.3e}, Std: {pred_flat.std():.3e}')
    print(f'  UNet targets (true large-scale Foreground: B-modes) - Mean: {target_flat.mean():.3e}, Std: {target_flat.std():.3e}')
    print(f'  Observed B all-scales - Mean: {test_b_all_scales.flatten().mean():.3e}, Std: {test_b_all_scales.flatten().std():.3e}')
    print(f'  Pure CMB - Mean: {test_cmb_draw.flatten().mean():.3e}, Std: {test_cmb_draw.flatten().std():.3e}')
    print(f'  UNet CMB reconstructed - Mean: {unet_cmb_reconstructed.flatten().mean():.3e}, Std: {unet_cmb_reconstructed.flatten().std():.3e}')
    print(f'\nCMB Reconstruction Correlations (overall flattened - all pixels across all patches):')
    pure_cmb_flat = test_cmb_draw.flatten()
    unet_cmb_flat = unet_cmb_reconstructed.flatten()
    corr_unet = np.corrcoef(pure_cmb_flat, unet_cmb_flat)[0, 1]
    print(f'  UNet CMB vs Pure CMB correlation (overall): {corr_unet:.4f}')
    print(f'\nPrediction vs Target Correlations (Large-Scale Foreground B-Modes):')
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    corr_pred_target = np.corrcoef(pred_flat, target_flat)[0, 1]
    print(f'  Prediction vs Target correlation: {corr_pred_target:.4f}')
    if pred_vs_target_cross_spectra_array is not None:
        print(f'\nPrediction vs Target Cross-Spectrum Statistics:')
        print(f'  Mean cross-spectrum (across ell): {np.nanmean(mean_pred_vs_target_cross_ps):.6e}')
        print(f'  Mean cross-spectrum (across patches): {np.nanmean(pred_vs_target_cross_spectra_array):.6e}')
    if unet_cross_spectra_array is not None:
        print(f'\nUNet CMB vs Pure CMB Cross-Spectrum Statistics:')
        print(f'  Mean cross-spectrum (across ell): {np.nanmean(mean_unet_cross_ps):.6e}')
        print(f'  Mean cross-spectrum (across patches): {np.nanmean(unet_cross_spectra_array):.6e}')
    print(f'\n[ok] Test results saved to: {output_file}')
    print(f'  All datasets are aligned using the same test_indices')
    print(f'  Ready for further analysis!')
    print('\n' + '=' * 80)
    print('SAVING COMPARISON METRICS FOR MODEL COMPARISON')
    print('=' * 80)
    model_filename = os.path.basename(model_path)
    model_basename = os.path.splitext(model_filename)[0]
    model_identifier = f'{norm_suffix}_{channel_suffix}_{model_basename}'
    print(f'\nModel identifier: {model_identifier}')
    if pred_vs_target_cross_spectra_array is not None and mean_pred_vs_target_cross_ps is not None and (std_pred_vs_target_cross_ps is not None):
        cross_spectrum_file = os.path.join(output_dir, f'{model_identifier}_cross_spectrum_stats.txt')
        print(f'\nSaving cross-spectrum statistics to: {cross_spectrum_file}')
        print(f'  Debug: pred_vs_target_cross_ps_lower is None: {pred_vs_target_cross_ps_lower is None}')
        print(f'  Debug: pred_vs_target_cross_ps_upper is None: {pred_vs_target_cross_ps_upper is None}')
        if pred_vs_target_cross_ps_lower is not None:
            print(f'  Debug: pred_vs_target_cross_ps_lower shape: {pred_vs_target_cross_ps_lower.shape}')
        if pred_vs_target_cross_ps_upper is not None:
            print(f'  Debug: pred_vs_target_cross_ps_upper shape: {pred_vs_target_cross_ps_upper.shape}')
        with open(cross_spectrum_file, 'w') as f:
            f.write(f'Prediction vs Target Cross-Power Spectrum Statistics\n')
            f.write(f'Model: {model_identifier}\n')
            f.write(f'Number of patches: {len(pred_vs_target_cross_spectra_array)}\n')
            f.write(f'Ell bins: {len(mean_pred_vs_target_cross_ps)}\n')
            f.write(f"\n{'=' * 80}\n\n")
            if ell_array_ref is not None:
                f.write('Ell (multipole) array:\n')
                f.write('  [')
                for (i, ell) in enumerate(ell_array_ref):
                    if i < len(ell_array_ref) - 1:
                        f.write(f'{ell:.1f}, ')
                    else:
                        f.write(f'{ell:.1f}')
                    if (i + 1) % 8 == 0 and i < len(ell_array_ref) - 1:
                        f.write('\n   ')
                f.write(']\n')
                f.write(f"\n{'=' * 80}\n\n")
            f.write('Mean cross-spectrum (per ell bin, averaged across patches):\n')
            f.write(f'  Mean across ell: {np.nanmean(mean_pred_vs_target_cross_ps):.6e}\n')
            f.write(f'  Std across ell: {np.nanstd(mean_pred_vs_target_cross_ps):.6e}\n')
            if ell_array_ref is not None:
                f.write(f'\nPer-ell-bin statistics (mean +/- std across patches):\n')
                for (i, ell) in enumerate(ell_array_ref):
                    if not np.isnan(mean_pred_vs_target_cross_ps[i]):
                        f.write(f'  ell={ell:6.1f}: {mean_pred_vs_target_cross_ps[i]:.6e} +/- {std_pred_vs_target_cross_ps[i]:.6e}\n')
            f.write(f'\n\nStd cross-spectrum (per ell bin, std across patches):\n')
            f.write(f'  Mean std (across ell): {np.nanmean(std_pred_vs_target_cross_ps):.6e}\n')
            f.write(f'  Std of std (across ell): {np.nanstd(std_pred_vs_target_cross_ps):.6e}\n')
            if pred_vs_target_cross_ps_lower is not None and pred_vs_target_cross_ps_upper is not None:
                f.write(f'\n\n2-tailed percentiles (16th and 84th, 1-sigma equivalent):\n')
                f.write(f'  Mean lower (16th percentile, across ell): {np.nanmean(pred_vs_target_cross_ps_lower):.6e}\n')
                f.write(f'  Mean upper (84th percentile, across ell): {np.nanmean(pred_vs_target_cross_ps_upper):.6e}\n')
                if ell_array_ref is not None:
                    f.write(f'\nPer-ell-bin 2-tailed percentiles (16th-84th):\n')
                    for (i, ell) in enumerate(ell_array_ref):
                        if not np.isnan(pred_vs_target_cross_ps_lower[i]) and (not np.isnan(pred_vs_target_cross_ps_upper[i])):
                            f.write(f'  ell={ell:6.1f}: [{pred_vs_target_cross_ps_lower[i]:.6e}, {pred_vs_target_cross_ps_upper[i]:.6e}]\n')
            f.write(f'\n\nOverall statistics (across all patches and ell bins):\n')
            f.write(f'  Mean cross-spectrum: {np.nanmean(pred_vs_target_cross_spectra_array):.6e}\n')
            f.write(f'  Std cross-spectrum: {np.nanstd(pred_vs_target_cross_spectra_array):.6e}\n')
            if ell_array_ref is not None:
                f.write(f"\n\n{'=' * 80}\n")
                f.write('Data for plotting (ell, mean_cross_ps, std_cross_ps, lower_percentile, upper_percentile):\n')
                if pred_vs_target_cross_ps_lower is not None and pred_vs_target_cross_ps_upper is not None:
                    f.write('Format: ell  mean_cross_ps  std_cross_ps  lower_16th  upper_84th\n')
                else:
                    f.write('Format: ell  mean_cross_ps  std_cross_ps\n')
                f.write(f"{'=' * 80}\n")
                for (i, ell) in enumerate(ell_array_ref):
                    if not np.isnan(mean_pred_vs_target_cross_ps[i]):
                        if pred_vs_target_cross_ps_lower is not None and pred_vs_target_cross_ps_upper is not None:
                            if not np.isnan(pred_vs_target_cross_ps_lower[i]) and (not np.isnan(pred_vs_target_cross_ps_upper[i])):
                                f.write(f'{ell:6.1f}  {mean_pred_vs_target_cross_ps[i]:.6e}  {std_pred_vs_target_cross_ps[i]:.6e}  {pred_vs_target_cross_ps_lower[i]:.6e}  {pred_vs_target_cross_ps_upper[i]:.6e}\n')
                            else:
                                f.write(f'{ell:6.1f}  {mean_pred_vs_target_cross_ps[i]:.6e}  {std_pred_vs_target_cross_ps[i]:.6e}  nan  nan\n')
                        else:
                            f.write(f'{ell:6.1f}  {mean_pred_vs_target_cross_ps[i]:.6e}  {std_pred_vs_target_cross_ps[i]:.6e}\n')
        print(f'  [ok] Saved cross-spectrum statistics')
    if pred_vs_target_correlations is not None:
        correlation_file = os.path.join(output_dir, f'{model_identifier}_correlation_stats.txt')
        print(f'\nSaving correlation statistics to: {correlation_file}')
        mean_corr = np.mean(pred_vs_target_correlations)
        std_corr = np.std(pred_vs_target_correlations)
        with open(correlation_file, 'w') as f:
            f.write(f'Prediction vs Target Spatial Correlation Statistics\n')
            f.write(f'Model: {model_identifier}\n')
            f.write(f'Number of patches: {len(pred_vs_target_correlations)}\n')
            f.write(f"\n{'=' * 80}\n\n")
            f.write(f'Mean correlation: {mean_corr:.6f}\n')
            f.write(f'Std correlation: {std_corr:.6f}\n')
            f.write(f'\nMin correlation: {np.min(pred_vs_target_correlations):.6f}\n')
            f.write(f'Max correlation: {np.max(pred_vs_target_correlations):.6f}\n')
            f.write(f'Median correlation: {np.median(pred_vs_target_correlations):.6f}\n')
        print(f'  [ok] Saved correlation statistics')
        print(f'    Mean: {mean_corr:.6f}, Std: {std_corr:.6f}')
    if pred_vs_target_mse_array is not None:
        mse_file = os.path.join(output_dir, f'{model_identifier}_mse_stats.txt')
        print(f'\nSaving MSE statistics to: {mse_file}')
        mean_mse = np.mean(pred_vs_target_mse_array)
        std_mse = np.std(pred_vs_target_mse_array)
        mean_target_vs_zero_mse = np.mean(target_vs_zero_mse_array) if target_vs_zero_mse_array is not None else None
        with open(mse_file, 'w') as f:
            f.write(f'Prediction vs Target MSE Statistics\n')
            f.write(f'Model: {model_identifier}\n')
            f.write(f'Number of patches: {len(pred_vs_target_mse_array)}\n')
            f.write(f"\n{'=' * 80}\n\n")
            f.write(f'MSE(prediction, target):\n')
            f.write(f'  Mean MSE: {mean_mse:.6e}\n')
            f.write(f'  Std MSE: {std_mse:.6e}\n')
            f.write(f'  Min MSE: {np.min(pred_vs_target_mse_array):.6e}\n')
            f.write(f'  Max MSE: {np.max(pred_vs_target_mse_array):.6e}\n')
            f.write(f'  Median MSE: {np.median(pred_vs_target_mse_array):.6e}\n')
            if mean_target_vs_zero_mse is not None:
                f.write(f'\nMSE(target, zero) [baseline]:\n')
                f.write(f'  Mean MSE: {mean_target_vs_zero_mse:.6e}\n')
                f.write(f'  Std MSE: {np.std(target_vs_zero_mse_array):.6e}\n')
                if mean_target_vs_zero_mse > 0:
                    improvement_ratio = mean_target_vs_zero_mse / mean_mse
                    f.write(f'\nImprovement ratio (baseline / UNet): {improvement_ratio:.4f}x\n')
                    f.write(f'UNet MSE is {improvement_ratio:.2f}x better (lower) than baseline\n')
        print(f'  [ok] Saved MSE statistics')
        print(f'    Mean MSE: {mean_mse:.6e}, Std MSE: {std_mse:.6e}')
    print(f'\n[ok] All comparison metrics saved to: {output_dir}')
    print(f'  Files saved with model identifier: {model_identifier}')
if __name__ == '__main__':
    main()
