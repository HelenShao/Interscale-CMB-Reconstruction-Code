import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import argparse
import shutil
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import threading
import time
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from paths_config import BACKUP_DIR_INTERSCALE as BACKUP_DIR, B_SMALL_DIR, E_T_DIR, ILC_DIR, OUTPUT_DIR_INTERSCALE as OUTPUT_DIR, STATS_FILE_16CH, STATS_FILE_8CH
import architecture
plt.rcParams['text.usetex'] = False

class IndexedTensorDataset(Dataset):

    def __init__(self, tensor_x, tensor_y):
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y
        assert len(tensor_x) == len(tensor_y), 'Input and target must have same length'

    def __getitem__(self, idx):
        return (self.tensor_x[idx], self.tensor_y[idx], idx)

    def __len__(self):
        return len(self.tensor_x)
CONFIG = {'batch_size': 64, 'learning_rate': 0.0001, 'weight_decay': 1e-05, 'in_channels': 8, 'out_channels': 1, 'feature_dims': [32, 64, 128, 256, 512, 1024], 'negative_slope': 0.01, 'toy_model': 'InterscaleMultifreq', 'num_epochs': 200, 'max_grad_norm': 5.0, 'scale_lr_by_dataset_size': True, 'lr_reference_size': 100, 'use_warmup': True, 'warmup_epochs': 10, 'warmup_start_lr': 1e-06, 'log_gradients': True, 'gradient_log_freq': 10}
FREQUENCIES = [95, 145, 220, 270]
print('=' * 80)
print('UNET TRAINING: INTER-SCALE MULTI-FREQUENCY ILC CORRECTION')
print('=' * 80)
print(f'Configuration:')
for (key, value) in CONFIG.items():
    print(f'  {key}: {value}')
print('=' * 80)

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

def load_and_prepare_data(normalize=True, use_b_small=True, use_ilc=True, use_e_t=False, test_mode=False, cmb_draw_scale=1.0):
    print('\n' + '=' * 80)
    print('LOADING AND PREPARING DATA')
    print('=' * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (stop_event, warmup_thread) = start_gpu_warmup(device, verbose=True)
    if not use_ilc and (not use_b_small):
        raise ValueError('At least one of use_ilc or use_b_small must be True')
    print('\nStep 1: Opening data files with memory mapping (not loading into RAM)...')
    ILC_foregrounds_mmap = {}
    if use_ilc:
        print('\n  Opening ILC products (memory mapped)...')
        for freq in FREQUENCIES:
            file_path = f'{ILC_DIR}/ILC_foregrounds_b_{freq}.npy'
            print(f'    Opening ILC foregrounds ({freq} GHz): {file_path}')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f'ILC foregrounds file not found: {file_path}')
            ILC_foregrounds_mmap[freq] = np.load(file_path, mmap_mode='r')
            print(f'      Shape: {ILC_foregrounds_mmap[freq].shape} (memory mapped)')
    else:
        print('\n  Skipping ILC foregrounds (--b-only mode)')
    ilc_residuals_file = f'{ILC_DIR}/ilc_residuals_b.npy'
    print(f'  Opening ILC residuals (target): {ilc_residuals_file}')
    if not os.path.exists(ilc_residuals_file):
        raise FileNotFoundError(f'ILC residuals file not found: {ilc_residuals_file}')
    ilc_residuals_mmap = np.load(ilc_residuals_file, mmap_mode='r')
    print(f'    Shape: {ilc_residuals_mmap.shape} (memory mapped)')
    ilc_cmb_file = f'{ILC_DIR}/ILC_cmb_b.npy'
    print(f'  Opening ILC CMB reconstructions: {ilc_cmb_file}')
    if not os.path.exists(ilc_cmb_file):
        raise FileNotFoundError(f'ILC CMB file not found: {ilc_cmb_file}')
    ilc_cmb_mmap = np.load(ilc_cmb_file, mmap_mode='r')
    print(f'    Shape: {ilc_cmb_mmap.shape} (memory mapped)')
    cmb_draw_file = f'{B_SMALL_DIR}/sim1-150_freq220_cmb_draw_b_all_scales.npy'
    print(f'  Opening pure primordial CMB B-modes: {cmb_draw_file}')
    if not os.path.exists(cmb_draw_file):
        raise FileNotFoundError(f'CMB draw file not found: {cmb_draw_file}')
    cmb_draw_b_all_scales_mmap = np.load(cmb_draw_file, mmap_mode='r')
    print(f'    Shape: {cmb_draw_b_all_scales_mmap.shape} (memory mapped)')
    if cmb_draw_scale != 1.0:
        print(f'  Applying scaling factor {cmb_draw_scale} to cmb_draw_b_all_scales')
        cmb_draw_b_all_scales_mmap = cmb_draw_b_all_scales_mmap * cmb_draw_scale
        print(f'    Scaled data range: [{cmb_draw_b_all_scales_mmap.min():.6e}, {cmb_draw_b_all_scales_mmap.max():.6e}]')
    else:
        print(f'  No scaling applied to cmb_draw_b_all_scales (scale factor = 1.0)')
    B_small_mmap = {}
    if use_b_small:
        print('\n  Opening B small-scale patches (memory mapped)...')
        for freq in FREQUENCIES:
            file_path = f'{B_SMALL_DIR}/sim1-150_freq{freq}_B_patches_small.npy'
            print(f'    Opening B small-scale ({freq} GHz): {file_path}')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f'B small-scale file not found: {file_path}')
            B_small_mmap[freq] = np.load(file_path, mmap_mode='r')
            print(f'      Shape: {B_small_mmap[freq].shape} (memory mapped)')
    else:
        print('\n  Skipping B small-scale patches (--ilc-only mode)')
    E_modes_mmap = {}
    T_modes_mmap = {}
    if use_e_t:
        print('\n  Opening E and T mode patches (memory mapped)...')
        for freq in FREQUENCIES:
            file_path = f'{E_T_DIR}/sim1-150_freq{freq}_E_patches.npy'
            print(f'    Opening E mode ({freq} GHz): {file_path}')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f'E mode file not found: {file_path}')
            E_modes_mmap[freq] = np.load(file_path, mmap_mode='r')
            print(f'      Shape: {E_modes_mmap[freq].shape} (memory mapped)')
        for freq in FREQUENCIES:
            file_path = f'{E_T_DIR}/sim1-150_freq{freq}_T_patches.npy'
            print(f'    Opening T mode ({freq} GHz): {file_path}')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f'T mode file not found: {file_path}')
            T_modes_mmap[freq] = np.load(file_path, mmap_mode='r')
            print(f'      Shape: {T_modes_mmap[freq].shape} (memory mapped)')
    else:
        print('\n  Skipping E and T mode patches (not using --add-e-t flag)')
    print('\nStep 3: Verifying data consistency...')
    n_patches = ilc_residuals_mmap.shape[0]
    print(f'  Total patches: {n_patches}')
    ref_spatial_shape = ilc_residuals_mmap.shape[1:]
    assert ilc_cmb_mmap.shape[0] == n_patches, f'Mismatch: ILC_cmb has {ilc_cmb_mmap.shape[0]} patches, expected {n_patches}'
    assert ilc_cmb_mmap.shape[1:] == ref_spatial_shape, f'Mismatch: ILC_cmb has spatial shape {ilc_cmb_mmap.shape[1:]}, expected {ref_spatial_shape}'
    assert cmb_draw_b_all_scales_mmap.shape[0] == n_patches, f'Mismatch: cmb_draw_b_all_scales has {cmb_draw_b_all_scales_mmap.shape[0]} patches, expected {n_patches}'
    assert cmb_draw_b_all_scales_mmap.shape[1:] == ref_spatial_shape, f'Mismatch: cmb_draw_b_all_scales has spatial shape {cmb_draw_b_all_scales_mmap.shape[1:]}, expected {ref_spatial_shape}'
    for freq in FREQUENCIES:
        if use_ilc:
            assert ILC_foregrounds_mmap[freq].shape[0] == n_patches, f'Mismatch: ILC foregrounds at {freq} GHz has {ILC_foregrounds_mmap[freq].shape[0]} patches'
            assert ILC_foregrounds_mmap[freq].shape[1:] == ref_spatial_shape, f'Mismatch: ILC foregrounds at {freq} GHz has spatial shape {ILC_foregrounds_mmap[freq].shape[1:]}, expected {ref_spatial_shape}'
        if use_b_small:
            assert B_small_mmap[freq].shape[0] == n_patches, f'Mismatch: B small at {freq} GHz has {B_small_mmap[freq].shape[0]} patches'
            assert B_small_mmap[freq].shape[1:] == ref_spatial_shape, f'Mismatch: B small at {freq} GHz has spatial shape {B_small_mmap[freq].shape[1:]}, expected {ref_spatial_shape}'
        if use_e_t:
            assert E_modes_mmap[freq].shape[0] == n_patches, f'Mismatch: E mode at {freq} GHz has {E_modes_mmap[freq].shape[0]} patches'
            assert E_modes_mmap[freq].shape[1:] == ref_spatial_shape, f'Mismatch: E mode at {freq} GHz has spatial shape {E_modes_mmap[freq].shape[1:]}, expected {ref_spatial_shape}'
            assert T_modes_mmap[freq].shape[0] == n_patches, f'Mismatch: T mode at {freq} GHz has {T_modes_mmap[freq].shape[0]} patches'
            assert T_modes_mmap[freq].shape[1:] == ref_spatial_shape, f'Mismatch: T mode at {freq} GHz has spatial shape {T_modes_mmap[freq].shape[1:]}, expected {ref_spatial_shape}'
    print('\nStep 4: Loading normalization statistics and split indices...')
    stats_file = STATS_FILE_16CH if use_e_t else STATS_FILE_8CH
    print(f'  Loading from: {stats_file}')
    print(f'  WARNING: IMPORTANT: Statistics are from TRAIN SET ONLY (no data leakage)')
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Normalization statistics file not found: {stats_file}")
    stats = np.load(stats_file)
    train_indices = stats['train_indices']
    valid_indices = stats['valid_indices']
    test_indices = stats['test_indices']
    if test_mode:
        test_limit = 100
        print(f'\n  Filtering split indices for test mode (keeping only indices < {test_limit})...')
        train_indices = train_indices[train_indices < test_limit]
        valid_indices = valid_indices[valid_indices < test_limit]
        test_indices = test_indices[test_indices < test_limit]
        print(f'  After filtering:')
    print(f'  Train indices: {len(train_indices)} patches')
    print(f'  Valid indices: {len(valid_indices)} patches')
    print(f'  Test indices: {len(test_indices)} patches')
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
    channel_desc_parts = []
    if use_ilc:
        input_mean.extend(full_input_mean[0:4])
        input_std.extend(full_input_std[0:4])
        channel_desc_parts.append('channels 0-3 (ILC foregrounds)')
    if use_b_small:
        input_mean.extend(full_input_mean[4:8])
        input_std.extend(full_input_std[4:8])
        channel_desc_parts.append('channels 4-7 (B small-scale)')
    if use_e_t:
        input_mean.extend(full_input_mean[8:12])
        input_std.extend(full_input_std[8:12])
        channel_desc_parts.append('channels 8-11 (E modes)')
        input_mean.extend(full_input_mean[12:16])
        input_std.extend(full_input_std[12:16])
        channel_desc_parts.append('channels 12-15 (T modes)')
    input_mean = np.array(input_mean)
    input_std = np.array(input_std)
    channel_desc = ', '.join(channel_desc_parts)
    print(f'\nNormalization statistics (computed from train set only):')
    print(f'  Input mean (per-channel, {channel_desc}): {input_mean}')
    print(f'  Input std (per-channel, {channel_desc}): {input_std}')
    print(f'  Target mean: {target_mean:.6e}')
    print(f'  Target std: {target_std:.6e}')
    print('\nStep 5: Building UNet inputs split-by-split (memory optimized)...')
    n_channels = 4 * (int(use_ilc) + int(use_b_small) + int(use_e_t) * 2)
    channel_desc_parts_summary = []
    if use_ilc:
        channel_desc_parts_summary.append('4 ILC foregrounds')
    if use_b_small:
        channel_desc_parts_summary.append('4 B small-scale')
    if use_e_t:
        channel_desc_parts_summary.append('4 E modes')
        channel_desc_parts_summary.append('4 T modes')
    channel_desc_summary = ' + '.join(channel_desc_parts_summary)
    print(f'  Using {n_channels}-channel input ({channel_desc_summary})')

    def build_input_split(indices, split_name):
        input_channels_list = []
        if use_ilc:
            input_channels_list.append(np.array(ILC_foregrounds_mmap[95][indices]))
            input_channels_list.append(np.array(ILC_foregrounds_mmap[145][indices]))
            input_channels_list.append(np.array(ILC_foregrounds_mmap[220][indices]))
            input_channels_list.append(np.array(ILC_foregrounds_mmap[270][indices]))
        if use_b_small:
            input_channels_list.append(np.array(B_small_mmap[95][indices]))
            input_channels_list.append(np.array(B_small_mmap[145][indices]))
            input_channels_list.append(np.array(B_small_mmap[220][indices]))
            input_channels_list.append(np.array(B_small_mmap[270][indices]))
        if use_e_t:
            input_channels_list.append(np.array(E_modes_mmap[95][indices]))
            input_channels_list.append(np.array(E_modes_mmap[145][indices]))
            input_channels_list.append(np.array(E_modes_mmap[220][indices]))
            input_channels_list.append(np.array(E_modes_mmap[270][indices]))
            input_channels_list.append(np.array(T_modes_mmap[95][indices]))
            input_channels_list.append(np.array(T_modes_mmap[145][indices]))
            input_channels_list.append(np.array(T_modes_mmap[220][indices]))
            input_channels_list.append(np.array(T_modes_mmap[270][indices]))
        split_input = np.stack(input_channels_list, axis=1)
        return split_input
    import gc
    print(f'  Processing train split ({len(train_indices)} samples)...')
    train_X_np = build_input_split(train_indices, 'train')
    if normalize:
        for i in range(n_channels):
            train_X_np[:, i] = (train_X_np[:, i] - input_mean[i]) / (input_std[i] + 1e-08)
    train_X = torch.from_numpy(train_X_np).float()
    del train_X_np
    gc.collect()
    train_y_np = np.array(ilc_residuals_mmap[train_indices])[:, np.newaxis, :, :]
    if normalize:
        train_y_np = (train_y_np - target_mean) / (target_std + 1e-08)
    train_y = torch.from_numpy(train_y_np).float()
    del train_y_np
    train_ilc_cmb = torch.from_numpy(np.array(ilc_cmb_mmap[train_indices])).float()
    train_cmb_draw = torch.from_numpy(np.array(cmb_draw_b_all_scales_mmap[train_indices])).float()
    gc.collect()
    print(f'    Train split complete: {train_X.shape}')
    print(f'  Processing valid split ({len(valid_indices)} samples)...')
    valid_X_np = build_input_split(valid_indices, 'valid')
    if normalize:
        for i in range(n_channels):
            valid_X_np[:, i] = (valid_X_np[:, i] - input_mean[i]) / (input_std[i] + 1e-08)
    valid_X = torch.from_numpy(valid_X_np).float()
    del valid_X_np
    gc.collect()
    valid_y_np = np.array(ilc_residuals_mmap[valid_indices])[:, np.newaxis, :, :]
    if normalize:
        valid_y_np = (valid_y_np - target_mean) / (target_std + 1e-08)
    valid_y = torch.from_numpy(valid_y_np).float()
    del valid_y_np
    valid_ilc_cmb = torch.from_numpy(np.array(ilc_cmb_mmap[valid_indices])).float()
    valid_cmb_draw = torch.from_numpy(np.array(cmb_draw_b_all_scales_mmap[valid_indices])).float()
    gc.collect()
    print(f'    Valid split complete: {valid_X.shape}')
    print(f'  Processing test split ({len(test_indices)} samples)...')
    test_X_np = build_input_split(test_indices, 'test')
    if normalize:
        for i in range(n_channels):
            test_X_np[:, i] = (test_X_np[:, i] - input_mean[i]) / (input_std[i] + 1e-08)
    test_X = torch.from_numpy(test_X_np).float()
    del test_X_np
    gc.collect()
    test_y_np = np.array(ilc_residuals_mmap[test_indices])[:, np.newaxis, :, :]
    if normalize:
        test_y_np = (test_y_np - target_mean) / (target_std + 1e-08)
    test_y = torch.from_numpy(test_y_np).float()
    del test_y_np
    test_ilc_cmb = torch.from_numpy(np.array(ilc_cmb_mmap[test_indices])).float()
    test_cmb_draw = torch.from_numpy(np.array(cmb_draw_b_all_scales_mmap[test_indices])).float()
    gc.collect()
    print(f'    Test split complete: {test_X.shape}')
    print('\nStep 5b: Closing memory-mapped arrays...')
    if use_ilc:
        del ILC_foregrounds_mmap
    if use_b_small:
        del B_small_mmap
    if use_e_t:
        del E_modes_mmap
        del T_modes_mmap
    del ilc_residuals_mmap
    del ilc_cmb_mmap
    del cmb_draw_b_all_scales_mmap
    gc.collect()
    if normalize:
        print('\nVerifying normalization:')
        train_input_mean = train_X.float().mean().item()
        train_input_std = train_X.float().std().item()
        train_target_mean = train_y.float().mean().item()
        train_target_std = train_y.float().std().item()
        print(f'  Train input - Mean: {train_input_mean:.6f}, Std: {train_input_std:.6f}')
        print(f'  Train target - Mean: {train_target_mean:.6f}, Std: {train_target_std:.6f}')
        if abs(train_input_mean) > 0.1:
            print(f'  WARNING: WARNING: Train input mean ({train_input_mean:.6f}) is not close to 0')
        if abs(train_input_std - 1.0) > 0.1:
            print(f'  WARNING: WARNING: Train input std ({train_input_std:.6f}) is not close to 1.0')
        if abs(train_target_mean) > 0.1:
            print(f'  WARNING: WARNING: Train target mean ({train_target_mean:.6f}) is not close to 0')
        if abs(train_target_std - 1.0) > 0.1:
            print(f'  WARNING: WARNING: Train target std ({train_target_std:.6f}) is not close to 1.0')
    split_memory_gb = (train_X.element_size() * train_X.nelement() + valid_X.element_size() * valid_X.nelement() + test_X.element_size() * test_X.nelement()) / 1000000000.0
    print(f'\n  [ok] All splits processed and converted to PyTorch tensors')
    print(f'  Memory now contains only PyTorch tensors (~{split_memory_gb:.1f} GB)')
    print(f'  Train: {train_X.shape[0]} samples, Valid: {valid_X.shape[0]} samples, Test: {test_X.shape[0]} samples')
    print('\nStep 9: Creating PyTorch datasets...')
    train_dataset = IndexedTensorDataset(train_X, train_y)
    valid_dataset = IndexedTensorDataset(valid_X, valid_y)
    test_dataset = IndexedTensorDataset(test_X, test_y)
    print(f'  Train dataset: {len(train_dataset)} samples')
    print(f'  Valid dataset: {len(valid_dataset)} samples')
    print(f'  Test dataset: {len(test_dataset)} samples')
    data_dict = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset, 'stats': {'input_mean': input_mean, 'input_std': input_std, 'target_mean': target_mean, 'target_std': target_std, 'normalized': normalize}, 'indices': {'train': train_indices, 'valid': valid_indices, 'test': test_indices}, 'ilc_cmb': {'train': train_ilc_cmb, 'valid': valid_ilc_cmb, 'test': test_ilc_cmb}, 'cmb_draw': {'train': train_cmb_draw, 'valid': valid_cmb_draw, 'test': test_cmb_draw}}
    print('\n[ok] Data loading and preparation complete!')
    stop_gpu_warmup(stop_event, warmup_thread, verbose=True)
    return data_dict

def train_model(data_dict, config, resume_from_best=False, normalize=True, use_b_small=True, use_ilc=True, use_e_t=False, test_mode=False, output_dir_base=None, backup_dir_base=None):
    print('\n' + '=' * 80)
    print('TRAINING MODEL')
    print('=' * 80)
    base_output_dir = output_dir_base if output_dir_base is not None else OUTPUT_DIR
    base_backup_dir = backup_dir_base if backup_dir_base is not None else BACKUP_DIR
    norm_suffix = 'normalized' if normalize else 'unnormalized'
    if use_ilc and use_b_small and use_e_t:
        channel_suffix = 'full_et'
        channel_desc = 'ILC + B small + E + T'
    elif use_ilc and use_b_small:
        channel_suffix = 'full'
        channel_desc = 'ILC + B small'
    elif use_ilc:
        channel_suffix = 'ilc_only'
        channel_desc = 'ILC only'
    elif use_b_small:
        channel_suffix = 'b_only'
        channel_desc = 'B small only'
    else:
        raise ValueError('At least one of use_ilc or use_b_small must be True')
    feature_dims = config.get('feature_dims', CONFIG['feature_dims'])
    max_feat = max(feature_dims) if feature_dims else 1024
    feature_suffix = f'_maxfeat{max_feat}' if max_feat != 1024 else ''
    test_suffix = '_test' if test_mode else ''
    output_dir = os.path.join(base_output_dir, f'{norm_suffix}_{channel_suffix}{feature_suffix}{test_suffix}')
    backup_dir = os.path.join(base_backup_dir, f'{norm_suffix}_{channel_suffix}{feature_suffix}{test_suffix}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    print(f'\nNormalization mode: {norm_suffix}')
    print(f"Input channels: {config['in_channels']} ({channel_desc})")
    if test_mode:
        print(f"WARNING: TEST MODE: Output directory includes '_test' suffix")
    print(f'Output directory: {output_dir}')
    print(f'Backup directory: {backup_dir}')
    train_dataset = data_dict['train']
    valid_dataset = data_dict['valid']
    test_dataset = data_dict['test']
    num_workers = min(8, os.cpu_count() or 4)
    persistent_workers = num_workers > 0
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=2 if num_workers > 0 else None)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=2 if num_workers > 0 else None)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=2 if num_workers > 0 else None)
    print(f'\nDataLoader configuration:')
    print(f'  num_workers: {num_workers}')
    print(f'  persistent_workers: {persistent_workers}')
    print(f'  prefetch_factor: 2')
    print(f'\nDataset sizes:')
    print(f'  Train: {len(train_dataset)} samples, {len(train_loader)} batches')
    print(f'  Valid: {len(valid_dataset)} samples, {len(valid_loader)} batches')
    print(f'  Test:  {len(test_dataset)} samples, {len(test_loader)} batches')
    norm_tag = 'norm' if normalize else 'nonorm'
    file_id = f'{len(train_dataset)}_{len(valid_dataset)}_{norm_tag}'
    checkpoint_path = os.path.join(output_dir, f'checkpoint_{file_id}.pt')
    best_model_path = os.path.join(output_dir, f'best_model_{file_id}.pt')
    checkpoint_config = None
    if resume_from_best and os.path.exists(best_model_path):
        print(f'\nFound best model checkpoint: {best_model_path}')
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            print('  Using architecture config from best model checkpoint')
            print(f"    in_channels: {checkpoint_config.get('in_channels')}")
            print(f"    feature_dims: {checkpoint_config.get('feature_dims')}")
    elif os.path.exists(checkpoint_path):
        print(f'\nFound checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            print('  Using architecture config from checkpoint')
            print(f"    in_channels: {checkpoint_config.get('in_channels')}")
            print(f"    feature_dims: {checkpoint_config.get('feature_dims')}")
    if checkpoint_config is not None:
        print('\nWARNING: Overriding current config with checkpoint config for model architecture')
        config['in_channels'] = checkpoint_config.get('in_channels', config['in_channels'])
        config['out_channels'] = checkpoint_config.get('out_channels', config['out_channels'])
        config['feature_dims'] = checkpoint_config.get('feature_dims', config['feature_dims'])
        config['negative_slope'] = checkpoint_config.get('negative_slope', config['negative_slope'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print('  cuDNN benchmark: Enabled (optimizes for consistent input sizes)')
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.2f} GB')
    mean_init_path = None
    if not normalize:
        input_mean = data_dict['stats']['input_mean']
        input_std = data_dict['stats']['input_std']
        target_mean = data_dict['stats']['target_mean']
        target_std = data_dict['stats']['target_std']
        train_x_mean = float(np.mean(input_mean))
        train_x_std = float(np.mean(input_std))
        stats_dict = {'train_x_mean': train_x_mean, 'train_x_std': train_x_std, 'train_y_mean': float(target_mean), 'train_y_std': float(target_std)}
        mean_init_path = os.path.join(output_dir, 'mean_init_stats.pkl')
        with open(mean_init_path, 'wb') as f:
            pickle.dump(stats_dict, f)
        print(f'\nCreated mean initialization file: {mean_init_path}')
        print(f'  WARNING: Statistics are from TRAIN SET ONLY (no data leakage)')
        print(f'  Target mean for final layer initialization: {target_mean:.6e}')
    model = architecture.UNET(config['in_channels'], config['out_channels'], config['feature_dims'], config['negative_slope'], mean_init=mean_init_path).to(device)
    print(f"Model initialized with {config['in_channels']} input channels")
    print(f"Feature dimensions: {config['feature_dims']}")
    num_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Total trainable parameters: {num_params:,}')
    base_lr = config['learning_rate']
    if config.get('scale_lr_by_dataset_size', False):
        dataset_size = len(train_dataset)
        reference_size = config.get('lr_reference_size', 100)
        lr_scale = np.sqrt(dataset_size / reference_size)
        scaled_lr = base_lr * lr_scale
        print(f'\nLearning rate scaling:')
        print(f'  Base LR: {base_lr:.2e}')
        print(f'  Dataset size: {dataset_size:,}')
        print(f'  Reference size: {reference_size:,}')
        print(f'  Scale factor: {lr_scale:.4f}')
        print(f'  Scaled LR: {scaled_lr:.2e}')
        actual_lr = scaled_lr
    else:
        actual_lr = base_lr
        print(f'\nLearning rate: {actual_lr:.2e} (no scaling)')
    base_learning_rate = actual_lr
    optimizer = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    train_losses = []
    val_losses = []
    train_correlations = []
    val_correlations = []
    train_cmb_correlations = []
    val_cmb_correlations = []
    train_ilc_mse = []
    train_unet_recon_mse = []
    val_ilc_mse = []
    val_unet_recon_mse = []
    gradient_stats_history = []
    learning_rates = []
    best_val_loss = float('inf')
    best_val_correlation = -float('inf')
    best_val_cmb_correlation = -float('inf')
    start_epoch = 0
    early_stopped = False
    patience_counter = 0
    patience = 20
    use_warmup = config.get('use_warmup', False)
    warmup_epochs = config.get('warmup_epochs', 5) if use_warmup else 0
    warmup_start_lr = config.get('warmup_start_lr', 1e-05) if use_warmup else base_learning_rate
    if use_warmup:
        print(f'\nLearning rate warmup enabled:')
        print(f'  Warmup epochs: {warmup_epochs}')
        print(f'  Start LR: {warmup_start_lr:.2e}')
        print(f'  Target LR: {base_learning_rate:.2e}')
    if resume_from_best and os.path.exists(best_model_path):
        print(f'\nResuming from best model: {best_model_path}')
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f'Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.3e}')
        print(f"  Note: Optimizer and scheduler reinitialized (best model doesn't save training state)")
    elif os.path.exists(checkpoint_path):
        print(f'\nResuming from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_val_loss = checkpoint['best_val_loss']
        if 'train_correlations' in checkpoint:
            train_correlations = checkpoint['train_correlations']
        if 'val_correlations' in checkpoint:
            val_correlations = checkpoint['val_correlations']
        if 'best_val_correlation' in checkpoint:
            best_val_correlation = checkpoint['best_val_correlation']
        if 'train_cmb_correlations' in checkpoint:
            train_cmb_correlations = checkpoint['train_cmb_correlations']
        if 'val_cmb_correlations' in checkpoint:
            val_cmb_correlations = checkpoint['val_cmb_correlations']
        if 'best_val_cmb_correlation' in checkpoint:
            best_val_cmb_correlation = checkpoint['best_val_cmb_correlation']
        if 'train_ilc_mse' in checkpoint:
            train_ilc_mse = checkpoint['train_ilc_mse']
        if 'train_unet_recon_mse' in checkpoint:
            train_unet_recon_mse = checkpoint['train_unet_recon_mse']
        if 'val_ilc_mse' in checkpoint:
            val_ilc_mse = checkpoint['val_ilc_mse']
        if 'val_unet_recon_mse' in checkpoint:
            val_unet_recon_mse = checkpoint['val_unet_recon_mse']
        if 'learning_rates' in checkpoint:
            learning_rates = checkpoint['learning_rates']
        if 'gradient_stats_history' in checkpoint:
            gradient_stats_history = checkpoint['gradient_stats_history']
        print(f'Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.3e}')
        if best_val_correlation > -float('inf'):
            print(f'  Best val correlation: {best_val_correlation:.3f}')
        print(f'  Optimizer and scheduler states restored')
        if 'learning_rates' in checkpoint:
            print(f'  Learning rate history restored ({len(learning_rates)} entries)')
        if 'gradient_stats_history' in checkpoint:
            print(f'  Gradient stats history restored ({len(gradient_stats_history)} entries)')

    def compute_gradient_stats(model):
        total_norm = 0.0
        max_norm = 0.0
        param_count = 0
        grad_norms = []
        for (name, param) in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_norm = max(max_norm, param_norm.item())
                grad_norms.append(param_norm.item())
                param_count += 1
        total_norm = total_norm ** (1.0 / 2)
        return {'total_norm': total_norm, 'max_norm': max_norm, 'mean_norm': np.mean(grad_norms) if grad_norms else 0.0, 'param_count': param_count, 'grad_norms': grad_norms}

    def compute_correlation(target, predicted):
        target_np = target.detach().cpu().numpy()
        pred_np = predicted.detach().cpu().numpy()
        batch_size = target_np.shape[0]
        correlations = []
        for i in range(batch_size):
            target_flat = target_np[i].flatten()
            pred_flat = pred_np[i].flatten()
            try:
                corr = np.corrcoef(target_flat, pred_flat)[0, 1]
                if not (np.isnan(corr) or np.isinf(corr)):
                    correlations.append(corr)
            except (ValueError, RuntimeWarning):
                pass
        if len(correlations) == 0:
            print(f'Warning: Could not compute valid correlation for any sample in batch')
            return None
        avg_correlation = np.mean(correlations)
        return avg_correlation

    def mse(patch1, patch2):
        patch1_np = patch1.detach().cpu().numpy() if isinstance(patch1, torch.Tensor) else patch1
        patch2_np = patch2.detach().cpu().numpy() if isinstance(patch2, torch.Tensor) else patch2
        if len(patch1_np.shape) == 4:
            patch1_np = patch1_np.squeeze(1)
        if len(patch2_np.shape) == 4:
            patch2_np = patch2_np.squeeze(1)
        batch_size = patch1_np.shape[0]
        mse_values = []
        for i in range(batch_size):
            mse_val = np.mean((patch1_np[i] - patch2_np[i]) ** 2)
            if not (np.isnan(mse_val) or np.isinf(mse_val)):
                mse_values.append(mse_val)
        if len(mse_values) == 0:
            return None
        return np.mean(mse_values)

    def compute_cmb_reconstruction_correlation(ilc_cmb_batch, unet_pred_batch, cmb_draw_batch, normalize=True, target_mean=None, target_std=None):
        ilc_cmb_np = ilc_cmb_batch.detach().cpu().numpy()
        unet_pred_np = unet_pred_batch.detach().cpu().numpy()
        cmb_draw_np = cmb_draw_batch.detach().cpu().numpy()
        if len(unet_pred_np.shape) == 4:
            unet_pred_np = unet_pred_np.squeeze(1)
        if normalize:
            if target_mean is None or target_std is None:
                raise ValueError('target_mean and target_std must be provided when normalize=True')
            unet_pred_np = unet_pred_np * target_std + target_mean
        unet_cmb_reconstructed = ilc_cmb_np - unet_pred_np
        batch_size = unet_cmb_reconstructed.shape[0]
        if not hasattr(compute_cmb_reconstruction_correlation, '_diagnostic_printed'):
            unet_cmb_reconstructed_alt = ilc_cmb_np + unet_pred_np
            if batch_size > 0:
                unet_cmb_flat = unet_cmb_reconstructed[0].flatten()
                unet_cmb_alt_flat = unet_cmb_reconstructed_alt[0].flatten()
                cmb_draw_flat = cmb_draw_np[0].flatten()
                try:
                    corr_sub = np.corrcoef(unet_cmb_flat, cmb_draw_flat)[0, 1]
                    corr_add = np.corrcoef(unet_cmb_alt_flat, cmb_draw_flat)[0, 1]
                except:
                    pass
            compute_cmb_reconstruction_correlation._diagnostic_printed = True
        correlations = []
        for i in range(batch_size):
            unet_cmb_flat = unet_cmb_reconstructed[i].flatten()
            cmb_draw_flat = cmb_draw_np[i].flatten()
            try:
                corr = np.corrcoef(unet_cmb_flat, cmb_draw_flat)[0, 1]
                if not (np.isnan(corr) or np.isinf(corr)):
                    correlations.append(corr)
            except (ValueError, RuntimeWarning):
                pass
        if len(correlations) == 0:
            return None
        avg_correlation = np.mean(correlations)
        return avg_correlation
    print('\n' + '=' * 80)
    print('STARTING TRAINING')
    print('=' * 80)
    for epoch in range(start_epoch, config['num_epochs']):
        if use_warmup and epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = warmup_start_lr + (base_learning_rate - warmup_start_lr) * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        model.train()
        batch_train_losses = []
        train_corr_list = []
        epoch_gradient_stats = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Train]")
        for (batch_idx, (x, y, indices)) in enumerate(train_pbar):
            (x, y) = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            if config.get('log_gradients', False):
                grad_log_freq = config.get('gradient_log_freq', 10)
                if grad_log_freq == 0 or (batch_idx + 1) % grad_log_freq == 0 or batch_idx == len(train_loader) - 1:
                    grad_stats = compute_gradient_stats(model)
                    epoch_gradient_stats.append(grad_stats)
                    if grad_log_freq > 0:
                        train_pbar.set_postfix({'loss': f'{loss.item():.3e}', 'grad_norm': f"{grad_stats['total_norm']:.2e}"})
            optimizer.step()
            loss_value = loss.item()
            batch_train_losses.append(loss_value)
            with torch.no_grad():
                batch_corr = compute_correlation(y.detach(), pred.detach())
                if batch_corr is not None:
                    train_corr_list.append(batch_corr)
            if not config.get('log_gradients', False) or config.get('gradient_log_freq', 10) == 0:
                train_pbar.set_postfix({'loss': f'{loss_value:.3e}'})
        if config.get('log_gradients', False) and epoch_gradient_stats:
            avg_grad_stats = {'total_norm': np.mean([s['total_norm'] for s in epoch_gradient_stats]), 'max_norm': np.max([s['max_norm'] for s in epoch_gradient_stats]), 'mean_norm': np.mean([s['mean_norm'] for s in epoch_gradient_stats]), 'param_count': epoch_gradient_stats[0]['param_count'] if epoch_gradient_stats else 0}
            gradient_stats_history.append(avg_grad_stats)
        else:
            gradient_stats_history.append(None)
        avg_train_loss = np.mean(batch_train_losses)
        train_losses.append(avg_train_loss)
        avg_train_corr = np.mean(train_corr_list) if train_corr_list else None
        train_correlations.append(avg_train_corr)
        model.eval()
        train_cmb_corr_list = []
        train_ilc_mse_list = []
        train_unet_recon_mse_list = []
        with torch.no_grad():
            train_loader_ordered = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=min(8, os.cpu_count() or 4), pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=2 if num_workers > 0 else None)
            for (x, y, indices) in train_loader_ordered:
                (x, y) = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                pred = model(x)
                indices_np = indices.numpy()
                ilc_cmb_batch = data_dict['ilc_cmb']['train'][indices_np].to(device, non_blocking=True)
                cmb_draw_batch = data_dict['cmb_draw']['train'][indices_np].to(device, non_blocking=True)
                target_mean = data_dict['stats']['target_mean'] if normalize else None
                target_std = data_dict['stats']['target_std'] if normalize else None
                batch_cmb_corr = compute_cmb_reconstruction_correlation(ilc_cmb_batch, pred, cmb_draw_batch, normalize=normalize, target_mean=target_mean, target_std=target_std)
                if batch_cmb_corr is not None:
                    train_cmb_corr_list.append(batch_cmb_corr)
                batch_ilc_mse = mse(cmb_draw_batch, ilc_cmb_batch)
                if batch_ilc_mse is not None:
                    train_ilc_mse_list.append(batch_ilc_mse)
                target_mean = data_dict['stats']['target_mean'] if normalize else None
                target_std = data_dict['stats']['target_std'] if normalize else None
                pred_denorm = pred.squeeze(1) if len(pred.shape) == 4 else pred
                if normalize and target_mean is not None and (target_std is not None):
                    pred_denorm = pred_denorm * target_std + target_mean
                unet_cmb_reconstructed = ilc_cmb_batch - pred_denorm
                batch_unet_recon_mse = mse(cmb_draw_batch, unet_cmb_reconstructed)
                if batch_unet_recon_mse is not None:
                    train_unet_recon_mse_list.append(batch_unet_recon_mse)
        avg_train_cmb_corr = np.mean(train_cmb_corr_list) if train_cmb_corr_list else None
        train_cmb_correlations.append(avg_train_cmb_corr)
        avg_train_ilc_mse = np.mean(train_ilc_mse_list) if train_ilc_mse_list else None
        train_ilc_mse.append(avg_train_ilc_mse)
        avg_train_unet_recon_mse = np.mean(train_unet_recon_mse_list) if train_unet_recon_mse_list else None
        train_unet_recon_mse.append(avg_train_unet_recon_mse)
        model.eval()
        batch_val_losses = []
        val_corr_list = []
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Valid]")
            for (x, y, indices) in valid_pbar:
                (x, y) = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                pred = model(x)
                loss = F.mse_loss(pred, y)
                loss_value = loss.item()
                batch_val_losses.append(loss_value)
                batch_corr = compute_correlation(y, pred)
                if batch_corr is not None:
                    val_corr_list.append(batch_corr)
                valid_pbar.set_postfix({'loss': f'{loss_value:.3e}'})
        avg_val_loss = np.mean(batch_val_losses)
        val_losses.append(avg_val_loss)
        avg_val_corr = np.mean(val_corr_list) if val_corr_list else None
        val_correlations.append(avg_val_corr)
        if epoch == 0:
            print('\n' + '=' * 80)
            print('FIRST EPOCH DIAGNOSTICS')
            print('=' * 80)
            if batch_train_losses and len(batch_train_losses) > 0:
                first_batch_loss = batch_train_losses[0]
                loss_decrease = first_batch_loss - avg_train_loss
                print(f'First batch loss: {first_batch_loss:.6e}')
                print(f'Average epoch loss: {avg_train_loss:.6e}')
                print(f'Loss decreased during epoch: {loss_decrease:.6e}')
            else:
                print(f'Average epoch loss: {avg_train_loss:.6e}')
            if config.get('log_gradients', False) and gradient_stats_history[-1] is not None:
                grad_stats = gradient_stats_history[-1]
                print(f'\nGradient stats:')
                print(f"  Total norm: {grad_stats['total_norm']:.6e}")
                print(f"  Max norm: {grad_stats['max_norm']:.6e}")
                print(f"  Mean norm: {grad_stats['mean_norm']:.6e}")
                print(f"  Parameters with gradients: {grad_stats['param_count']}")
                if grad_stats['total_norm'] < 1e-06:
                    print(f'  WARNING: Gradients may be vanishing (total norm < 1e-6)')
                elif grad_stats['total_norm'] > 100:
                    print(f'  WARNING: Gradients may be exploding (total norm > 100)')
            else:
                print('\nGradient stats: Not available (gradient logging may be disabled)')
            print(f'\nLearning rate: {current_lr:.2e}')
            print(f'Train loss: {avg_train_loss:.6e}')
            print(f'Val loss: {avg_val_loss:.6e}')
            train_corr_str = f'{avg_train_corr:.6f}' if avg_train_corr is not None else 'N/A'
            val_corr_str = f'{avg_val_corr:.6f}' if avg_val_corr is not None else 'N/A'
            print(f'Train correlation: {train_corr_str}')
            print(f'Val correlation: {val_corr_str}')
            print('=' * 80)
        val_cmb_corr_list = []
        val_ilc_mse_list = []
        val_unet_recon_mse_list = []
        with torch.no_grad():
            for (x, y, indices) in valid_loader:
                (x, y) = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                pred = model(x)
                indices_np = indices.numpy()
                ilc_cmb_batch = data_dict['ilc_cmb']['valid'][indices_np].to(device, non_blocking=True)
                cmb_draw_batch = data_dict['cmb_draw']['valid'][indices_np].to(device, non_blocking=True)
                target_mean = data_dict['stats']['target_mean'] if normalize else None
                target_std = data_dict['stats']['target_std'] if normalize else None
                batch_cmb_corr = compute_cmb_reconstruction_correlation(ilc_cmb_batch, pred, cmb_draw_batch, normalize=normalize, target_mean=target_mean, target_std=target_std)
                if batch_cmb_corr is not None:
                    val_cmb_corr_list.append(batch_cmb_corr)
                batch_ilc_mse = mse(cmb_draw_batch, ilc_cmb_batch)
                if batch_ilc_mse is not None:
                    val_ilc_mse_list.append(batch_ilc_mse)
                target_mean = data_dict['stats']['target_mean'] if normalize else None
                target_std = data_dict['stats']['target_std'] if normalize else None
                pred_denorm = pred.squeeze(1) if len(pred.shape) == 4 else pred
                if normalize and target_mean is not None and (target_std is not None):
                    pred_denorm = pred_denorm * target_std + target_mean
                unet_cmb_reconstructed = ilc_cmb_batch - pred_denorm
                batch_unet_recon_mse = mse(cmb_draw_batch, unet_cmb_reconstructed)
                if batch_unet_recon_mse is not None:
                    val_unet_recon_mse_list.append(batch_unet_recon_mse)
        avg_val_cmb_corr = np.mean(val_cmb_corr_list) if val_cmb_corr_list else None
        val_cmb_correlations.append(avg_val_cmb_corr)
        avg_val_ilc_mse = np.mean(val_ilc_mse_list) if val_ilc_mse_list else None
        val_ilc_mse.append(avg_val_ilc_mse)
        avg_val_unet_recon_mse = np.mean(val_unet_recon_mse_list) if val_unet_recon_mse_list else None
        val_unet_recon_mse.append(avg_val_unet_recon_mse)
        scheduler.step(avg_val_loss)
        if avg_val_corr is not None:
            if avg_val_corr > best_val_correlation:
                best_val_correlation = avg_val_corr
        save_best_model = False
        save_reason = ''
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_best_model = True
            save_reason = 'val_loss'
            patience_counter = 0
        else:
            patience_counter += 1
        if save_best_model:
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'val_loss': avg_val_loss, 'best_val_loss': best_val_loss, 'val_correlation': avg_val_corr, 'best_val_correlation': best_val_correlation, 'config': config}, best_model_path)
            print(f'\n[ok] New best model saved! Reason: {save_reason}')
            val_corr_str = f'{avg_val_corr:.3f}' if avg_val_corr is not None else 'N/A'
            print(f'  Val loss: {avg_val_loss:.3e}, Val correlation: {val_corr_str}')
            print(f'  Primary: {best_model_path}')
            backup_best_model_path = os.path.join(backup_dir, f'best_model_{file_id}.pt')
            shutil.copy2(best_model_path, backup_best_model_path)
            print(f'  Backup:  {backup_best_model_path}')
        checkpoint_data = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'val_loss': avg_val_loss, 'best_val_loss': best_val_loss, 'val_correlation': avg_val_corr, 'best_val_correlation': best_val_correlation, 'train_losses': train_losses, 'val_losses': val_losses, 'train_correlations': train_correlations, 'val_correlations': val_correlations, 'train_cmb_correlations': train_cmb_correlations, 'val_cmb_correlations': val_cmb_correlations, 'best_val_cmb_correlation': best_val_cmb_correlation, 'train_ilc_mse': train_ilc_mse, 'train_unet_recon_mse': train_unet_recon_mse, 'val_ilc_mse': val_ilc_mse, 'val_unet_recon_mse': val_unet_recon_mse, 'learning_rates': learning_rates, 'config': config}
        if config.get('log_gradients', False):
            checkpoint_data['gradient_stats_history'] = gradient_stats_history
        torch.save(checkpoint_data, checkpoint_path)
        if avg_val_cmb_corr is not None:
            if avg_val_cmb_corr > best_val_cmb_correlation:
                best_val_cmb_correlation = avg_val_cmb_corr
        train_corr_str = f'{avg_train_corr:.3f}' if avg_train_corr is not None else 'N/A'
        val_corr_str = f'{avg_val_corr:.3f}' if avg_val_corr is not None else 'N/A'
        best_corr_str = f'{best_val_correlation:.3f}' if best_val_correlation > -float('inf') else 'N/A'
        train_cmb_corr_str = f'{avg_train_cmb_corr:.3f}' if avg_train_cmb_corr is not None else 'N/A'
        val_cmb_corr_str = f'{avg_val_cmb_corr:.3f}' if avg_val_cmb_corr is not None else 'N/A'
        best_cmb_corr_str = f'{best_val_cmb_correlation:.3f}' if best_val_cmb_correlation > -float('inf') else 'N/A'
        train_ilc_mse_str = f'{avg_train_ilc_mse:.3e}' if avg_train_ilc_mse is not None else 'N/A'
        train_unet_recon_mse_str = f'{avg_train_unet_recon_mse:.3e}' if avg_train_unet_recon_mse is not None else 'N/A'
        val_ilc_mse_str = f'{avg_val_ilc_mse:.3e}' if avg_val_ilc_mse is not None else 'N/A'
        val_unet_recon_mse_str = f'{avg_val_unet_recon_mse:.3e}' if avg_val_unet_recon_mse is not None else 'N/A'
        grad_stats_str = ''
        if config.get('log_gradients', False) and gradient_stats_history[-1] is not None:
            grad_stats = gradient_stats_history[-1]
            grad_stats_str = f", Grad Norm: {grad_stats['total_norm']:.2e}"
        warmup_str = ' [WARMUP]' if use_warmup and epoch < warmup_epochs else ''
        val_loss_improving = avg_val_loss < best_val_loss
        improvement_str = '[ok]' if val_loss_improving else '[fail]'
        epochs_since_best = patience_counter
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}{warmup_str} - Train Loss: {avg_train_loss:.3e}, Val Loss: {avg_val_loss:.3e} {improvement_str}, Best Val Loss: {best_val_loss:.3e} (epoch {np.argmin(val_losses) + 1})")
        print(f'  LR: {current_lr:.2e}{grad_stats_str}')
        if epochs_since_best > 0:
            print(f"  WARNING: Val loss hasn't improved for {epochs_since_best} epochs (patience: {patience})")
        print(f'  Train Correlation: {train_corr_str}, Val Correlation: {val_corr_str}, Best Val Correlation: {best_corr_str}')
        print(f'  Train CMB Corr: {train_cmb_corr_str}, Val CMB Corr: {val_cmb_corr_str}, Best Val CMB Corr: {best_cmb_corr_str}')
        print(f'  Train ILC MSE: {train_ilc_mse_str}, Train UNet Recon MSE: {train_unet_recon_mse_str}')
        print(f'  Val ILC MSE: {val_ilc_mse_str}, Val UNet Recon MSE: {val_unet_recon_mse_str}')
        if patience_counter >= patience:
            print(f"\n[ok] Early stopping triggered: Validation loss hasn't improved for {patience} epochs")
            print(f'  Best validation loss: {best_val_loss:.3e}')
            print(f'  Current validation loss: {avg_val_loss:.3e}')
            early_stopped = True
            break
        losses_path = os.path.join(output_dir, f'losses_{file_id}.txt')
        with open(losses_path, 'w') as f:
            header = 'epoch,train_loss,val_loss,min_val_loss,train_corr,val_corr,max_val_corr,train_cmb_corr,val_cmb_corr,max_val_cmb_corr,train_ilc_mse,train_unet_recon_mse,val_ilc_mse,val_unet_recon_mse,learning_rate'
            if config.get('log_gradients', False):
                header += ',grad_total_norm,grad_max_norm,grad_mean_norm'
            f.write(header + '\n')
            for i in range(len(train_losses)):
                train_corr_val = f'{train_correlations[i]:.3f}' if train_correlations[i] is not None else 'nan'
                val_corr_val = f'{val_correlations[i]:.3f}' if val_correlations[i] is not None else 'nan'
                max_corr_val = f'{max([c for c in val_correlations[:i + 1] if c is not None]):.3f}' if any((c is not None for c in val_correlations[:i + 1])) else 'nan'
                train_cmb_corr_val = f'{train_cmb_correlations[i]:.3f}' if i < len(train_cmb_correlations) and train_cmb_correlations[i] is not None else 'nan'
                val_cmb_corr_val = f'{val_cmb_correlations[i]:.3f}' if i < len(val_cmb_correlations) and val_cmb_correlations[i] is not None else 'nan'
                max_cmb_corr_val = f'{max([c for c in val_cmb_correlations[:i + 1] if c is not None]):.3f}' if any((c is not None for c in val_cmb_correlations[:i + 1])) else 'nan'
                train_ilc_mse_val = f'{train_ilc_mse[i]:.3e}' if i < len(train_ilc_mse) and train_ilc_mse[i] is not None else 'nan'
                train_unet_recon_mse_val = f'{train_unet_recon_mse[i]:.3e}' if i < len(train_unet_recon_mse) and train_unet_recon_mse[i] is not None else 'nan'
                val_ilc_mse_val = f'{val_ilc_mse[i]:.3e}' if i < len(val_ilc_mse) and val_ilc_mse[i] is not None else 'nan'
                val_unet_recon_mse_val = f'{val_unet_recon_mse[i]:.3e}' if i < len(val_unet_recon_mse) and val_unet_recon_mse[i] is not None else 'nan'
                lr_val = f'{learning_rates[i]:.2e}' if i < len(learning_rates) else 'nan'
                row = f'{i + 1},{train_losses[i]:.3e},{val_losses[i]:.3e},{min(val_losses[:i + 1]):.3e},{train_corr_val},{val_corr_val},{max_corr_val},{train_cmb_corr_val},{val_cmb_corr_val},{max_cmb_corr_val},{train_ilc_mse_val},{train_unet_recon_mse_val},{val_ilc_mse_val},{val_unet_recon_mse_val},{lr_val}'
                if config.get('log_gradients', False):
                    if i < len(gradient_stats_history) and gradient_stats_history[i] is not None:
                        grad_stats = gradient_stats_history[i]
                        row += f",{grad_stats['total_norm']:.2e},{grad_stats['max_norm']:.2e},{grad_stats['mean_norm']:.2e}"
                    else:
                        row += ',nan,nan,nan'
                f.write(row + '\n')
    final_epoch = epoch + 1 if early_stopped else config['num_epochs']
    final_model_path = os.path.join(output_dir, f'final_model_{file_id}.pt')
    torch.save({'epoch': final_epoch, 'model_state_dict': model.state_dict(), 'val_loss': avg_val_loss, 'best_val_loss': best_val_loss, 'val_correlation': avg_val_corr if 'avg_val_corr' in locals() else None, 'best_val_correlation': best_val_correlation, 'early_stopped': early_stopped, 'config': config}, final_model_path)
    print(f'\nFinal model saved to {final_model_path}')
    print(f'  Final epoch: {final_epoch}, Early stopped: {early_stopped}')
    print(f'  Final val loss: {avg_val_loss:.3e}, Best val loss: {best_val_loss:.3e}')
    if 'avg_val_corr' in locals() and avg_val_corr is not None:
        print(f'  Final val correlation: {avg_val_corr:.3f}, Best val correlation: {best_val_correlation:.3f}')
    print('\n' + '=' * 80)
    print('EVALUATING ON TEST SET')
    print('=' * 80)
    if not os.path.exists(best_model_path):
        print(f'\nWARNING: Best model not found at {best_model_path}')
        print('  Skipping test evaluation (no best model to evaluate)')
        test_loss = float('inf')
    else:
        print(f'\nLoading best model from: {best_model_path}')
        best_checkpoint = torch.load(best_model_path)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        model.eval()
        batch_test_losses = []
        test_cmb_corr_list = []
        test_ilc_mse_list = []
        test_unet_recon_mse_list = []
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc='Test Set Evaluation')
            for (x, y, indices) in test_pbar:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)
                loss = F.mse_loss(pred, y)
                batch_test_losses.append(loss.item())
                indices_np = indices.numpy()
                ilc_cmb_batch = data_dict['ilc_cmb']['test'][indices_np].to(device, non_blocking=True)
                cmb_draw_batch = data_dict['cmb_draw']['test'][indices_np].to(device, non_blocking=True)
                target_mean = data_dict['stats']['target_mean'] if normalize else None
                target_std = data_dict['stats']['target_std'] if normalize else None
                batch_cmb_corr = compute_cmb_reconstruction_correlation(ilc_cmb_batch, pred, cmb_draw_batch, normalize=normalize, target_mean=target_mean, target_std=target_std)
                if batch_cmb_corr is not None:
                    test_cmb_corr_list.append(batch_cmb_corr)
                batch_ilc_mse = mse(cmb_draw_batch, ilc_cmb_batch)
                if batch_ilc_mse is not None:
                    test_ilc_mse_list.append(batch_ilc_mse)
                pred_denorm = pred.squeeze(1) if len(pred.shape) == 4 else pred
                if normalize and target_mean is not None and (target_std is not None):
                    pred_denorm = pred_denorm * target_std + target_mean
                unet_cmb_reconstructed = ilc_cmb_batch - pred_denorm
                batch_unet_recon_mse = mse(cmb_draw_batch, unet_cmb_reconstructed)
                if batch_unet_recon_mse is not None:
                    test_unet_recon_mse_list.append(batch_unet_recon_mse)
                test_pbar.set_postfix({'loss': f'{loss.item():.3e}'})
        test_loss = np.mean(batch_test_losses)
        test_cmb_corr = np.mean(test_cmb_corr_list) if test_cmb_corr_list else None
        test_ilc_mse = np.mean(test_ilc_mse_list) if test_ilc_mse_list else None
        test_unet_recon_mse = np.mean(test_unet_recon_mse_list) if test_unet_recon_mse_list else None
        print(f'\nTest set evaluation complete')
        print(f'  Test loss (from best model): {test_loss:.3e}')
        test_cmb_corr_str = f'{test_cmb_corr:.3f}' if test_cmb_corr is not None else 'N/A'
        print(f'  Test CMB correlation: {test_cmb_corr_str}')
        test_ilc_mse_str = f'{test_ilc_mse:.3e}' if test_ilc_mse is not None else 'N/A'
        test_unet_recon_mse_str = f'{test_unet_recon_mse:.3e}' if test_unet_recon_mse is not None else 'N/A'
        print(f'  Test ILC MSE: {test_ilc_mse_str}, Test UNet Recon MSE: {test_unet_recon_mse_str}')
    return (train_losses, val_losses, test_loss, output_dir, train_correlations, val_correlations, train_cmb_correlations, val_cmb_correlations)

def plot_training_curves(train_losses, val_losses, output_dir):
    print('\nPlotting training curves...')
    (fig, axes) = plt.subplots(1, 2, figsize=(15, 5))
    ax1 = axes[0]
    epochs = np.arange(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    ax2.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('Training and Validation Loss (Log Scale)', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {plot_path}')

def print_training_summary(train_losses, val_losses, test_loss, output_dir, train_correlations=None, val_correlations=None, train_cmb_correlations=None, val_cmb_correlations=None):
    print('\n' + '=' * 80)
    print('TRAINING SUMMARY')
    print('=' * 80)
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    best_train_loss = train_losses[best_epoch - 1]
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    overfitting_gap = best_train_loss - best_val_loss
    best_corr_epoch = None
    best_val_corr = None
    if val_correlations and any((c is not None for c in val_correlations)):
        valid_corrs = [(i, c) for (i, c) in enumerate(val_correlations) if c is not None]
        if valid_corrs:
            (best_corr_epoch, best_val_corr) = max(valid_corrs, key=lambda x: x[1])
            best_corr_epoch += 1
    print(f'\nBest epoch (by validation loss): {best_epoch}')
    print(f'  Train loss: {best_train_loss:.3e}')
    print(f'  Val loss: {best_val_loss:.3e}')
    print(f'  Overfitting gap: {overfitting_gap:.3e}')
    if best_corr_epoch is not None:
        print(f'\nBest epoch (by validation correlation): {best_corr_epoch}')
        print(f'  Val correlation: {best_val_corr:.3f}')
        if train_correlations and train_correlations[best_corr_epoch - 1] is not None:
            print(f'  Train correlation: {train_correlations[best_corr_epoch - 1]:.3f}')
    best_cmb_corr_epoch = None
    best_val_cmb_corr = None
    if val_cmb_correlations and any((c is not None for c in val_cmb_correlations)):
        valid_cmb_corrs = [(i, c) for (i, c) in enumerate(val_cmb_correlations) if c is not None]
        if valid_cmb_corrs:
            (best_cmb_corr_epoch, best_val_cmb_corr) = max(valid_cmb_corrs, key=lambda x: x[1])
            best_cmb_corr_epoch += 1
            print(f'\nBest epoch (by validation CMB correlation): {best_cmb_corr_epoch}')
            print(f'  Val CMB correlation: {best_val_cmb_corr:.3f}')
            if train_cmb_correlations and train_cmb_correlations[best_cmb_corr_epoch - 1] is not None:
                print(f'  Train CMB correlation: {train_cmb_correlations[best_cmb_corr_epoch - 1]:.3f}')
    print(f'\nFinal epoch: {len(train_losses)}')
    print(f'  Train loss: {final_train_loss:.3e}')
    print(f'  Val loss: {final_val_loss:.3e}')
    if val_correlations and val_correlations[-1] is not None:
        print(f'  Val correlation: {val_correlations[-1]:.3f}')
    if train_correlations and train_correlations[-1] is not None:
        print(f'  Train correlation: {train_correlations[-1]:.3f}')
    if val_cmb_correlations and val_cmb_correlations[-1] is not None:
        print(f'  Val CMB correlation: {val_cmb_correlations[-1]:.3f}')
    if train_cmb_correlations and train_cmb_correlations[-1] is not None:
        print(f'  Train CMB correlation: {train_cmb_correlations[-1]:.3f}')
    print(f'\nTest set (held-out, evaluated once):')
    print(f'  Test loss (from best model): {test_loss:.3e}')
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('TRAINING SUMMARY\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'Best epoch (by validation loss): {best_epoch}\n')
        f.write(f'  Train loss: {best_train_loss:.3e}\n')
        f.write(f'  Val loss: {best_val_loss:.3e}\n')
        f.write(f'  Overfitting gap: {overfitting_gap:.3e}\n\n')
        if best_corr_epoch is not None:
            f.write(f'Best epoch (by validation correlation): {best_corr_epoch}\n')
            f.write(f'  Val correlation: {best_val_corr:.3f}\n\n')
        best_cmb_corr_epoch = None
        best_val_cmb_corr = None
        if val_cmb_correlations and any((c is not None for c in val_cmb_correlations)):
            valid_cmb_corrs = [(i, c) for (i, c) in enumerate(val_cmb_correlations) if c is not None]
            if valid_cmb_corrs:
                (best_cmb_corr_epoch, best_val_cmb_corr) = max(valid_cmb_corrs, key=lambda x: x[1])
                best_cmb_corr_epoch += 1
                f.write(f'Best epoch (by validation CMB correlation): {best_cmb_corr_epoch}\n')
                f.write(f'  Val CMB correlation: {best_val_cmb_corr:.3f}\n\n')
        f.write(f'Final epoch: {len(train_losses)}\n')
        f.write(f'  Train loss: {final_train_loss:.3e}\n')
        f.write(f'  Val loss: {final_val_loss:.3e}\n')
        if val_correlations and val_correlations[-1] is not None:
            f.write(f'  Val correlation: {val_correlations[-1]:.3f}\n')
        if train_correlations and train_correlations[-1] is not None:
            f.write(f'  Train correlation: {train_correlations[-1]:.3f}\n')
        if val_cmb_correlations and val_cmb_correlations[-1] is not None:
            f.write(f'  Val CMB correlation: {val_cmb_correlations[-1]:.3f}\n')
        if train_cmb_correlations and train_cmb_correlations[-1] is not None:
            f.write(f'  Train CMB correlation: {train_cmb_correlations[-1]:.3f}\n')
        f.write(f'\nTest set (held-out, evaluated once):\n')
        f.write(f'  Test loss (from best model): {test_loss:.3e}\n')
        f.write('=' * 80 + '\n')
    print(f'\nSummary saved to {summary_path}')
    print('=' * 80)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train UNet for inter-scale multi-frequency ILC correction', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-normalize', action='store_true', help='Disable data normalization (default: normalize using train stats)')
    parser.add_argument('--resume-best', action='store_true', help='Resume training from best saved model')
    parser.add_argument('--ilc-only', action='store_true', help='Use only ILC foreground channels (4 channels), skip B small-scale channels (channels 4-7)')
    parser.add_argument('--b-only', action='store_true', help='Use only B small-scale channels (4 channels), skip ILC foreground channels (channels 0-3)')
    parser.add_argument('--add-e-t', action='store_true', help='Add E and T mode channels (8 additional channels: channels 8-15)')
    parser.add_argument('--test-mode', action='store_true', help='Test mode: Use only first 100 indices from loaded data (for debugging overfitting)')
    parser.add_argument('--cmb-draw-scale', type=float, default=1.0, help='Scaling factor for cmb_draw_b_all_scales (ground truth CMB). Default: 1.0')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory (default: uses OUTPUT_DIR from config)')
    parser.add_argument('--backup-dir', type=str, default=None, help='Override backup directory (default: uses BACKUP_DIR from config)')
    parser.add_argument('--max-features', type=int, default=None, help='Maximum feature dimension (bottleneck size). Will generate feature_dims automatically. Default: use CONFIG value')
    args = parser.parse_args()
    if args.ilc_only and args.b_only:
        parser.error('--ilc-only and --b-only cannot be used together')
    return args

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    args = parse_arguments()
    normalize = not args.no_normalize
    resume_from_best = args.resume_best
    test_mode = args.test_mode
    cmb_draw_scale = args.cmb_draw_scale
    use_e_t = args.add_e_t
    if args.ilc_only:
        use_ilc = True
        use_b_small = False
        base_channels = 4
        channel_desc = '4 (ILC only)'
    elif args.b_only:
        use_ilc = False
        use_b_small = True
        base_channels = 4
        channel_desc = '4 (B small only)'
    else:
        use_ilc = True
        use_b_small = True
        base_channels = 8
        channel_desc = '8 (ILC + B small)'
    if use_e_t:
        additional_channels = 8
        total_channels = base_channels + additional_channels
        channel_desc += f' + 8 (E + T) = {total_channels} total'
    else:
        total_channels = base_channels
    print(f"\nNormalization: {('Enabled' if normalize else 'Disabled')}")
    print(f"Resume from best: {('Yes' if resume_from_best else 'No')}")
    print(f'Input channels: {channel_desc}')
    print(f"Test mode: {('Enabled (using first 100 indices)' if test_mode else 'Disabled')}")
    print(f'CMB draw scale factor: {cmb_draw_scale}')
    if test_mode:
        print('\nWARNING: TEST MODE ENABLED: This will limit all datasets to first 100 indices')
        print('  Use this mode to test for overfitting with a smaller dataset')
    config = CONFIG.copy()
    config['in_channels'] = total_channels
    if args.max_features is not None:
        max_features = args.max_features
        feature_dims = []
        current = 32
        while current <= max_features:
            feature_dims.append(current)
            if current == max_features:
                break
            current *= 2
            if current > max_features:
                feature_dims.append(max_features)
                break
        config['feature_dims'] = feature_dims
        print(f'\nUsing custom feature dimensions (max_features={max_features}): {feature_dims}')
    output_dir_override = args.output_dir if args.output_dir else OUTPUT_DIR
    backup_dir_override = args.backup_dir if args.backup_dir else BACKUP_DIR
    if args.output_dir:
        print(f'Using custom output directory: {output_dir_override}')
    if args.backup_dir:
        print(f'Using custom backup directory: {backup_dir_override}')
    data_dict = load_and_prepare_data(normalize=normalize, use_b_small=use_b_small, use_ilc=use_ilc, use_e_t=use_e_t, test_mode=test_mode, cmb_draw_scale=cmb_draw_scale)
    (train_losses, val_losses, test_loss, output_dir, train_correlations, val_correlations, train_cmb_correlations, val_cmb_correlations) = train_model(data_dict, config, resume_from_best=resume_from_best, normalize=normalize, use_b_small=use_b_small, use_ilc=use_ilc, use_e_t=use_e_t, test_mode=test_mode, output_dir_base=output_dir_override, backup_dir_base=backup_dir_override)
    plot_training_curves(train_losses, val_losses, output_dir)
    print_training_summary(train_losses, val_losses, test_loss, output_dir, train_correlations, val_correlations, train_cmb_correlations, val_cmb_correlations)
    print('\n' + '=' * 80)
    print('TRAINING COMPLETE!')
    print('=' * 80)
    print(f'All results saved to: {output_dir}')
    print('=' * 80)
if __name__ == '__main__':
    main()
