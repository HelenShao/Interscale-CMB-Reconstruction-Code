#!/usr/bin/env python3
"""Single-frequency inter-scale UNet (220 GHz; ell_split 200)."""
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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from paths_config import BACKUP_DIR_SINGLEFREQ as BACKUP_DIR, B_SMALL_DIR, OUTPUT_DIR_SINGLEFREQ as OUTPUT_DIR, STATS_FILE_SINGLEFREQ as STATS_FILE, T_E_DIR
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
FREQUENCY = 220
CONFIG = {'batch_size': 32, 'learning_rate': 0.001, 'weight_decay': 0.0005, 'in_channels': 3, 'out_channels': 1, 'feature_dims': [32, 64, 128, 256, 512, 1024], 'negative_slope': 0.01, 'toy_model': 'SingleFreqInterscale', 'num_epochs': 200, 'max_grad_norm': 1.0, 'use_warmup': True, 'warmup_epochs': 5, 'warmup_start_lr': 1e-05, 'log_gradients': True, 'early_stopping_patience': 20}
print('=' * 80)
print('UNET TRAINING: SINGLE-FREQUENCY INTER-SCALE (B, T, E) @ 220 GHz')
print('=' * 80)
print(f'Configuration:')
for (key, value) in CONFIG.items():
    print(f'  {key}: {value}')
print('=' * 80)

def load_and_prepare_data(normalize=True, input_mode='bte', test_mode=False):
    print('\n' + '=' * 80)
    print('LOADING AND PREPARING DATA')
    print('=' * 80)
    print('\nStep 1: Loading single-frequency (220 GHz) data...')
    if input_mode in ['bte', 'b_only']:
        b_small_file = f'{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_B_patches_small.npy'
        print(f'  Loading B small-scale: {b_small_file}')
        if not os.path.exists(b_small_file):
            raise FileNotFoundError(f'B small-scale file not found: {b_small_file}')
        b_small = np.load(b_small_file)
        print(f'    Shape: {b_small.shape}')
    else:
        b_small = None
    b_large_file = f'{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_B_patches_large.npy'
    print(f'  Loading B large-scale (target): {b_large_file}')
    if not os.path.exists(b_large_file):
        raise FileNotFoundError(f'B large-scale file not found: {b_large_file}')
    b_large = np.load(b_large_file)
    print(f'    Shape: {b_large.shape}')
    if input_mode in ['bte', 'te_only', 't_only']:
        t_file = f'{T_E_DIR}/sim1-150_freq{FREQUENCY}_T_patches.npy'
        print(f'  Loading T all-scale @ {FREQUENCY} GHz: {t_file}')
        if not os.path.exists(t_file):
            raise FileNotFoundError(f'T patches file not found: {t_file}')
        t_all = np.load(t_file)
        print(f'    Shape: {t_all.shape}')
    else:
        t_all = None
    if input_mode in ['bte', 'te_only', 'e_only']:
        e_file = f'{T_E_DIR}/sim1-150_freq{FREQUENCY}_E_patches.npy'
        print(f'  Loading E all-scale @ {FREQUENCY} GHz: {e_file}')
        if not os.path.exists(e_file):
            raise FileNotFoundError(f'E patches file not found: {e_file}')
        e_all = np.load(e_file)
        print(f'    Shape: {e_all.shape}')
    else:
        print('  Skipping E channel (--b-only mode)')
        e_all = None
    cmb_draw_file = f'{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_cmb_draw_b_all_scales.npy'
    print(f'  Loading pure primordial CMB B-modes: {cmb_draw_file}')
    if not os.path.exists(cmb_draw_file):
        raise FileNotFoundError(f'CMB draw file not found: {cmb_draw_file}')
    cmb_draw_b_all_scales = np.load(cmb_draw_file)
    print(f'    Shape: {cmb_draw_b_all_scales.shape}')
    print('\nStep 2: Verifying data consistency...')
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
    if test_mode:
        test_limit = 100
        print(f'\nStep 3: TEST MODE ENABLED - Limiting all datasets to first {test_limit} indices')
        print(f'  Original number of patches: {n_patches}')
        if n_patches < test_limit:
            print(f'  WARNING: WARNING: Dataset has only {n_patches} patches, which is less than {test_limit}')
            print(f'  Using all available patches: {n_patches}')
            test_limit = n_patches
        if b_small is not None:
            b_small = b_small[:test_limit]
        b_large = b_large[:test_limit]
        cmb_draw_b_all_scales = cmb_draw_b_all_scales[:test_limit]
        if t_all is not None:
            t_all = t_all[:test_limit]
        if e_all is not None:
            e_all = e_all[:test_limit]
        n_patches = test_limit
        print(f'  [ok] All datasets limited to {n_patches} patches')
    print('\nStep 4: Constructing UNet input and target...')
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
    print(f'  UNet input shape: {UNet_input.shape}')
    print(f'  UNet target shape: {UNet_target.shape}')
    print('\nStep 5: Loading normalization statistics and split indices...')
    print(f'  Loading from: {STATS_FILE}')
    if not os.path.exists(STATS_FILE):
        raise FileNotFoundError(f'Normalization statistics file not found: {STATS_FILE}\nPlease generate this file first using the data preparation script.')
    stats = np.load(STATS_FILE)
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
    if input_mode == 'bte':
        input_mean = full_input_mean
        input_std = full_input_std
        channel_desc = 'all 3 channels (B small-scale + T + E)'
    elif input_mode == 'b_only':
        input_mean = full_input_mean[:1] if len(full_input_mean) > 1 else full_input_mean
        input_std = full_input_std[:1] if len(full_input_std) > 1 else full_input_std
        channel_desc = 'channel 0 (B small-scale only)'
    elif input_mode == 't_only':
        if len(full_input_mean) >= 3:
            input_mean = full_input_mean[1:2]
            input_std = full_input_std[1:2]
        else:
            raise ValueError(f'Expected 3-channel stats for t_only mode, got {len(full_input_mean)} channels')
        channel_desc = 'channel 1 (T only)'
    elif input_mode == 'e_only':
        if len(full_input_mean) >= 3:
            input_mean = full_input_mean[2:3]
            input_std = full_input_std[2:3]
        else:
            raise ValueError(f'Expected 3-channel stats for e_only mode, got {len(full_input_mean)} channels')
        channel_desc = 'channel 2 (E only)'
    else:
        if len(full_input_mean) >= 3:
            input_mean = full_input_mean[1:3]
            input_std = full_input_std[1:3]
        else:
            raise ValueError(f'Expected 3-channel stats for te_only mode, got {len(full_input_mean)} channels')
        channel_desc = 'channels 1-2 (T + E only)'
    print(f'\nNormalization statistics (computed from train set only):')
    print(f'  Input mean (per-channel, {channel_desc}): {input_mean}')
    print(f'  Input std (per-channel, {channel_desc}): {input_std}')
    print(f'  Target mean: {target_mean:.6e}')
    print(f'  Target std: {target_std:.6e}')
    print('\nStep 6: Splitting data into train/valid/test...')
    train_X = UNet_input[train_indices]
    valid_X = UNet_input[valid_indices]
    test_X = UNet_input[test_indices]
    train_y = UNet_target[train_indices]
    valid_y = UNet_target[valid_indices]
    test_y = UNet_target[test_indices]
    train_cmb_draw = cmb_draw_b_all_scales[train_indices]
    valid_cmb_draw = cmb_draw_b_all_scales[valid_indices]
    test_cmb_draw = cmb_draw_b_all_scales[test_indices]
    print(f'  Train: {train_X.shape[0]} samples')
    print(f'  Valid: {valid_X.shape[0]} samples')
    print(f'  Test: {test_X.shape[0]} samples')
    if normalize:
        print('\nStep 7: Normalizing data using train-only statistics...')
        n_channels = train_X.shape[1]
        print(f'  Normalizing input (per-channel, {n_channels} channels)...')
        for i in range(n_channels):
            train_X[:, i] = (train_X[:, i] - input_mean[i]) / (input_std[i] + 1e-08)
            valid_X[:, i] = (valid_X[:, i] - input_mean[i]) / (input_std[i] + 1e-08)
            test_X[:, i] = (test_X[:, i] - input_mean[i]) / (input_std[i] + 1e-08)
        print('  Normalizing target...')
        train_y = (train_y - target_mean) / (target_std + 1e-08)
        valid_y = (valid_y - target_mean) / (target_std + 1e-08)
        test_y = (test_y - target_mean) / (target_std + 1e-08)
        print('  [ok] Normalization complete')
        print('\nVerifying normalization:')
        train_input_mean = train_X.mean()
        train_input_std = train_X.std()
        train_target_mean = train_y.mean()
        train_target_std = train_y.std()
        print(f'  Train input - Mean: {train_input_mean:.6f}, Std: {train_input_std:.6f}')
        print(f'  Train target - Mean: {train_target_mean:.6f}, Std: {train_target_std:.6f}')
        print(f'  Valid input - Mean: {valid_X.mean():.6f}, Std: {valid_X.std():.6f}')
        print(f'  Valid target - Mean: {valid_y.mean():.6f}, Std: {valid_y.std():.6f}')
        if abs(train_input_mean) > 0.1:
            print(f'  WARNING: WARNING: Train input mean ({train_input_mean:.6f}) is not close to 0')
        if abs(train_input_std - 1.0) > 0.1:
            print(f'  WARNING: WARNING: Train input std ({train_input_std:.6f}) is not close to 1.0')
        if abs(train_target_mean) > 0.1:
            print(f'  WARNING: WARNING: Train target mean ({train_target_mean:.6f}) is not close to 0')
        if abs(train_target_std - 1.0) > 0.1:
            print(f'  WARNING: WARNING: Train target std ({train_target_std:.6f}) is not close to 1.0')
    else:
        print('\nStep 7: Normalization disabled (using raw data)')
    print('\nStep 8: Converting to PyTorch tensors (one at a time to minimize memory)...')
    import gc
    train_X_torch = torch.from_numpy(train_X).float()
    del train_X
    gc.collect()
    valid_X_torch = torch.from_numpy(valid_X).float()
    del valid_X
    gc.collect()
    test_X_torch = torch.from_numpy(test_X).float()
    del test_X
    gc.collect()
    train_y_torch = torch.from_numpy(train_y).float()
    del train_y
    gc.collect()
    valid_y_torch = torch.from_numpy(valid_y).float()
    del valid_y
    gc.collect()
    test_y_torch = torch.from_numpy(test_y).float()
    del test_y
    gc.collect()
    train_cmb_draw_torch = torch.from_numpy(train_cmb_draw).float()
    del train_cmb_draw
    gc.collect()
    valid_cmb_draw_torch = torch.from_numpy(valid_cmb_draw).float()
    del valid_cmb_draw
    gc.collect()
    test_cmb_draw_torch = torch.from_numpy(test_cmb_draw).float()
    del test_cmb_draw
    gc.collect()
    del b_large
    if t_all is not None:
        del t_all
    if e_all is not None:
        del e_all
    if b_small is not None:
        del b_small
    del cmb_draw_b_all_scales
    del UNet_input
    del UNet_target
    gc.collect()
    print('  [ok] Conversion complete and NumPy arrays freed')
    print('\nStep 9: Creating PyTorch datasets...')
    train_dataset = IndexedTensorDataset(train_X_torch, train_y_torch)
    valid_dataset = IndexedTensorDataset(valid_X_torch, valid_y_torch)
    test_dataset = IndexedTensorDataset(test_X_torch, test_y_torch)
    print(f'  Train dataset: {len(train_dataset)} samples')
    print(f'  Valid dataset: {len(valid_dataset)} samples')
    print(f'  Test dataset: {len(test_dataset)} samples')
    data_dict = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset, 'stats': {'input_mean': input_mean, 'input_std': input_std, 'target_mean': target_mean, 'target_std': target_std, 'normalized': normalize}, 'indices': {'train': train_indices, 'valid': valid_indices, 'test': test_indices}, 'cmb_draw': {'train': train_cmb_draw_torch, 'valid': valid_cmb_draw_torch, 'test': test_cmb_draw_torch}}
    print('\n[ok] Data loading and preparation complete!')
    return data_dict

def train_model(data_dict, config, resume_from_best=False, normalize=True, input_mode='bte', test_mode=False):
    print('\n' + '=' * 80)
    print('TRAINING MODEL')
    print('=' * 80)
    norm_suffix = 'normalized' if normalize else 'unnormalized'
    if input_mode == 'bte':
        channel_suffix = 'teb'
        channel_desc = 'B small + T + E'
    elif input_mode == 'b_only':
        channel_suffix = 'b_only'
        channel_desc = 'B small only'
    elif input_mode == 't_only':
        channel_suffix = 't_only'
        channel_desc = 'T only'
    elif input_mode == 'e_only':
        channel_suffix = 'e_only'
        channel_desc = 'E only'
    else:
        channel_suffix = 'te_only'
        channel_desc = 'T + E only'
    test_suffix = '_test' if test_mode else ''
    output_dir = os.path.join(OUTPUT_DIR, f'{norm_suffix}_{channel_suffix}{test_suffix}')
    backup_dir = os.path.join(BACKUP_DIR, f'{norm_suffix}_{channel_suffix}{test_suffix}')
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
        print(f'  Target mean for initialization: {target_mean:.6e}')
    model = architecture.UNET(config['in_channels'], config['out_channels'], config['feature_dims'], config['negative_slope'], mean_init=mean_init_path).to(device)
    print(f"Model initialized with {config['in_channels']} input channels")
    print(f"Feature dimensions: {config['feature_dims']}")
    num_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Total trainable parameters: {num_params:,}')
    actual_lr = config['learning_rate']
    print(f'\nLearning rate: {actual_lr:.2e}')
    base_learning_rate = actual_lr
    optimizer = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    train_losses = []
    val_losses = []
    train_correlations = []
    val_correlations = []
    train_cmb_correlations = []
    val_cmb_correlations = []
    train_target_mse = []
    train_pred_vs_target_mse = []
    val_target_mse = []
    val_pred_vs_target_mse = []
    gradient_stats_history = []
    learning_rates = []
    best_val_loss = float('inf')
    best_val_correlation = -1.0
    best_val_cmb_correlation = -1.0
    start_epoch = 0
    early_stopped = False
    patience_counter = 0
    patience = config.get('early_stopping_patience', 20)
    use_warmup = config.get('use_warmup', False)
    warmup_epochs = config.get('warmup_epochs', 5) if use_warmup else 0
    warmup_start_lr = config.get('warmup_start_lr', 1e-05) if use_warmup else base_learning_rate
    if use_warmup:
        print(f'\nLearning rate warmup enabled:')
        print(f'  Warmup epochs: {warmup_epochs}')
        print(f'  Start LR: {warmup_start_lr:.2e}')
        print(f'  Target LR: {base_learning_rate:.2e}')
    norm_tag = 'norm' if normalize else 'nonorm'
    file_id = f'{len(train_dataset)}_{len(valid_dataset)}_{norm_tag}'
    checkpoint_path = os.path.join(output_dir, f'checkpoint_{file_id}.pt')
    best_model_path = os.path.join(output_dir, f'best_model_{file_id}.pt')
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
        if 'train_target_mse' in checkpoint:
            train_target_mse = checkpoint['train_target_mse']
        if 'train_pred_vs_target_mse' in checkpoint:
            train_pred_vs_target_mse = checkpoint['train_pred_vs_target_mse']
        if 'val_target_mse' in checkpoint:
            val_target_mse = checkpoint['val_target_mse']
        if 'val_pred_vs_target_mse' in checkpoint:
            val_pred_vs_target_mse = checkpoint['val_pred_vs_target_mse']
        if 'learning_rates' in checkpoint:
            learning_rates = checkpoint['learning_rates']
        if 'gradient_stats_history' in checkpoint:
            gradient_stats_history = checkpoint['gradient_stats_history']
        print(f'Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.3e}')
        if best_val_correlation > -1.0:
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
        if len(target_np.shape) == 4:
            target_np = target_np.squeeze(1)
        if len(pred_np.shape) == 4:
            pred_np = pred_np.squeeze(1)
        batch_size = target_np.shape[0]
        corrs = []
        for i in range(batch_size):
            target_patch = target_np[i].flatten()
            pred_patch = pred_np[i].flatten()
            try:
                corr = np.corrcoef(target_patch, pred_patch)[0, 1]
                if np.isnan(corr) or np.isinf(corr):
                    continue
                corrs.append(corr)
            except (ValueError, RuntimeWarning):
                continue
        return corrs

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

    def compute_cmb_reconstruction_correlation(b_large_scales_true, b_large_scales_pred, cmb_true):
        b_large_scales_true_np = b_large_scales_true.detach().cpu().numpy()
        b_large_scales_pred_np = b_large_scales_pred.detach().cpu().numpy()
        cmb_true_np = cmb_true.detach().cpu().numpy()
        if len(b_large_scales_pred_np.shape) == 4:
            b_large_scales_pred_np = b_large_scales_pred_np.squeeze(1)
        if len(b_large_scales_true_np.shape) == 4:
            b_large_scales_true_np = b_large_scales_true_np.squeeze(1)
        if len(cmb_true_np.shape) == 4:
            cmb_true_np = cmb_true_np.squeeze(1)
        batch_size = b_large_scales_true_np.shape[0]
        normalized_corrs = []
        for i in range(batch_size):
            map_before_cleaning = cmb_true_np[i] + b_large_scales_true_np[i]
            cmb_reconstructed = map_before_cleaning - b_large_scales_pred_np[i]
            cmb_reconstructed_flat = cmb_reconstructed.flatten()
            cmb_true_flat = cmb_true_np[i].flatten()
            try:
                corr = np.corrcoef(cmb_reconstructed_flat, cmb_true_flat)[0, 1]
                if np.isnan(corr) or np.isinf(corr):
                    continue
                normalized_corr = (corr + 1.0) / 2.0
                normalized_corrs.append(normalized_corr)
            except (ValueError, RuntimeWarning):
                continue
        return normalized_corrs
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
        train_cmb_corr_list = []
        train_target_mse_list = []
        train_pred_vs_target_mse_list = []
        epoch_gradient_stats = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Train]")
        for (batch_idx, (x, y, indices)) in enumerate(train_pbar):
            (x, y) = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            should_log_grad = False
            if config.get('log_gradients', False) and batch_idx == len(train_loader) - 1:
                grad_stats = compute_gradient_stats(model)
                epoch_gradient_stats.append(grad_stats)
                should_log_grad = True
                train_pbar.set_postfix({'loss': f'{loss.item():.3e}', 'grad_norm': f"{grad_stats['total_norm']:.2e}"})
            optimizer.step()
            loss_value = loss.item()
            batch_train_losses.append(loss_value)
            with torch.no_grad():
                batch_corrs = compute_correlation(y.detach(), pred.detach())
                train_corr_list.extend(batch_corrs)
                indices_np = indices.numpy()
                cmb_draw_batch = data_dict['cmb_draw']['train'][indices_np].to(device, non_blocking=True)
                y_flat = y.detach().squeeze(1) if len(y.shape) == 4 else y.detach()
                pred_flat = pred.detach().squeeze(1) if len(pred.shape) == 4 else pred.detach()
                batch_cmb_corrs = compute_cmb_reconstruction_correlation(y_flat, pred_flat, cmb_draw_batch)
                train_cmb_corr_list.extend(batch_cmb_corrs)
                zero_batch = torch.zeros_like(y_flat)
                batch_target_mse = mse(y_flat, zero_batch)
                if batch_target_mse is not None:
                    train_target_mse_list.append(batch_target_mse)
                batch_pred_vs_target_mse = mse(pred_flat, y_flat)
                if batch_pred_vs_target_mse is not None:
                    train_pred_vs_target_mse_list.append(batch_pred_vs_target_mse)
            if not should_log_grad:
                train_pbar.set_postfix({'loss': f'{loss_value:.3e}'})
        if config.get('log_gradients', False) and epoch_gradient_stats:
            final_grad_stats = epoch_gradient_stats[-1]
            gradient_stats_history.append(final_grad_stats)
        else:
            gradient_stats_history.append(None)
        avg_train_loss = np.mean(batch_train_losses)
        train_losses.append(avg_train_loss)
        avg_train_corr = np.mean(train_corr_list) if train_corr_list else None
        train_correlations.append(avg_train_corr)
        avg_train_cmb_corr = np.mean(train_cmb_corr_list) if train_cmb_corr_list else None
        train_cmb_correlations.append(avg_train_cmb_corr)
        avg_train_target_mse = np.mean(train_target_mse_list) if train_target_mse_list else None
        train_target_mse.append(avg_train_target_mse)
        avg_train_pred_vs_target_mse = np.mean(train_pred_vs_target_mse_list) if train_pred_vs_target_mse_list else None
        train_pred_vs_target_mse.append(avg_train_pred_vs_target_mse)
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
                batch_corrs = compute_correlation(y, pred)
                val_corr_list.extend(batch_corrs)
                valid_pbar.set_postfix({'loss': f'{loss_value:.3e}'})
        avg_val_loss = np.mean(batch_val_losses)
        val_losses.append(avg_val_loss)
        avg_val_corr = np.mean(val_corr_list) if val_corr_list else None
        val_correlations.append(avg_val_corr)
        val_cmb_corr_list = []
        val_target_mse_list = []
        val_pred_vs_target_mse_list = []
        with torch.no_grad():
            for (x, y, indices) in valid_loader:
                (x, y) = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                pred = model(x)
                indices_np = indices.numpy()
                cmb_draw_batch = data_dict['cmb_draw']['valid'][indices_np].to(device, non_blocking=True)
                y_flat = y.squeeze(1) if len(y.shape) == 4 else y
                pred_flat = pred.squeeze(1) if len(pred.shape) == 4 else pred
                batch_cmb_corrs = compute_cmb_reconstruction_correlation(y_flat, pred_flat, cmb_draw_batch)
                val_cmb_corr_list.extend(batch_cmb_corrs)
                zero_batch = torch.zeros_like(y_flat)
                batch_target_mse = mse(y_flat, zero_batch)
                if batch_target_mse is not None:
                    val_target_mse_list.append(batch_target_mse)
                batch_pred_vs_target_mse = mse(pred_flat, y_flat)
                if batch_pred_vs_target_mse is not None:
                    val_pred_vs_target_mse_list.append(batch_pred_vs_target_mse)
        avg_val_cmb_corr = np.mean(val_cmb_corr_list) if val_cmb_corr_list else None
        val_cmb_correlations.append(avg_val_cmb_corr)
        avg_val_target_mse = np.mean(val_target_mse_list) if val_target_mse_list else None
        val_target_mse.append(avg_val_target_mse)
        avg_val_pred_vs_target_mse = np.mean(val_pred_vs_target_mse_list) if val_pred_vs_target_mse_list else None
        val_pred_vs_target_mse.append(avg_val_pred_vs_target_mse)
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
        checkpoint_data = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'val_loss': avg_val_loss, 'best_val_loss': best_val_loss, 'val_correlation': avg_val_corr, 'best_val_correlation': best_val_correlation, 'train_losses': train_losses, 'val_losses': val_losses, 'train_correlations': train_correlations, 'val_correlations': val_correlations, 'train_cmb_correlations': train_cmb_correlations, 'val_cmb_correlations': val_cmb_correlations, 'best_val_cmb_correlation': best_val_cmb_correlation, 'train_target_mse': train_target_mse, 'train_pred_vs_target_mse': train_pred_vs_target_mse, 'val_target_mse': val_target_mse, 'val_pred_vs_target_mse': val_pred_vs_target_mse, 'learning_rates': learning_rates, 'config': config}
        if config.get('log_gradients', False):
            checkpoint_data['gradient_stats_history'] = gradient_stats_history
        torch.save(checkpoint_data, checkpoint_path)
        if avg_val_cmb_corr is not None:
            if avg_val_cmb_corr > best_val_cmb_correlation:
                best_val_cmb_correlation = avg_val_cmb_corr
        train_corr_str = f'{avg_train_corr:.3f}' if avg_train_corr is not None else 'N/A'
        val_corr_str = f'{avg_val_corr:.3f}' if avg_val_corr is not None else 'N/A'
        best_corr_str = f'{best_val_correlation:.3f}' if best_val_correlation > -1.0 else 'N/A'
        train_cmb_corr_str = f'{avg_train_cmb_corr:.3f}' if avg_train_cmb_corr is not None else 'N/A'
        val_cmb_corr_str = f'{avg_val_cmb_corr:.3f}' if avg_val_cmb_corr is not None else 'N/A'
        best_cmb_corr_str = f'{best_val_cmb_correlation:.3f}' if best_val_cmb_correlation > 0.0 else 'N/A'
        train_target_mse_str = f'{avg_train_target_mse:.3e}' if avg_train_target_mse is not None else 'N/A'
        train_pred_vs_target_mse_str = f'{avg_train_pred_vs_target_mse:.3e}' if avg_train_pred_vs_target_mse is not None else 'N/A'
        val_target_mse_str = f'{avg_val_target_mse:.3e}' if avg_val_target_mse is not None else 'N/A'
        val_pred_vs_target_mse_str = f'{avg_val_pred_vs_target_mse:.3e}' if avg_val_pred_vs_target_mse is not None else 'N/A'
        grad_stats_str = ''
        if config.get('log_gradients', False) and gradient_stats_history[-1] is not None:
            grad_stats = gradient_stats_history[-1]
            grad_stats_str = f", Grad Norm: {grad_stats['total_norm']:.2e}"
        warmup_str = ' [WARMUP]' if use_warmup and epoch < warmup_epochs else ''
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}{warmup_str} - Train Loss: {avg_train_loss:.3e}, Val Loss: {avg_val_loss:.3e}, Best Val Loss: {best_val_loss:.3e}")
        print(f'  LR: {current_lr:.2e}{grad_stats_str}')
        print(f'  Train Correlation: {train_corr_str}, Val Correlation: {val_corr_str}, Best Val Correlation: {best_corr_str}')
        print(f'  Train CMB Corr: {train_cmb_corr_str}, Val CMB Corr: {val_cmb_corr_str}, Best Val CMB Corr: {best_cmb_corr_str}')
        print(f'  Train Target MSE: {train_target_mse_str}, Train Pred vs Target MSE: {train_pred_vs_target_mse_str}')
        print(f'  Val Target MSE: {val_target_mse_str}, Val Pred vs Target MSE: {val_pred_vs_target_mse_str}')
        if patience_counter >= patience:
            print(f"\n[ok] Early stopping triggered: Validation loss hasn't improved for {patience} epochs")
            print(f'  Best validation loss: {best_val_loss:.3e}')
            print(f'  Current validation loss: {avg_val_loss:.3e}')
            early_stopped = True
            break
        losses_path = os.path.join(output_dir, f'losses_{file_id}.txt')
        with open(losses_path, 'w') as f:
            header = 'epoch,train_loss,val_loss,min_val_loss,train_corr,val_corr,max_val_corr,train_cmb_corr,val_cmb_corr,max_val_cmb_corr,train_target_mse,train_pred_vs_target_mse,val_target_mse,val_pred_vs_target_mse,learning_rate'
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
                train_target_mse_val = f'{train_target_mse[i]:.3e}' if i < len(train_target_mse) and train_target_mse[i] is not None else 'nan'
                train_pred_vs_target_mse_val = f'{train_pred_vs_target_mse[i]:.3e}' if i < len(train_pred_vs_target_mse) and train_pred_vs_target_mse[i] is not None else 'nan'
                val_target_mse_val = f'{val_target_mse[i]:.3e}' if i < len(val_target_mse) and val_target_mse[i] is not None else 'nan'
                val_pred_vs_target_mse_val = f'{val_pred_vs_target_mse[i]:.3e}' if i < len(val_pred_vs_target_mse) and val_pred_vs_target_mse[i] is not None else 'nan'
                lr_val = f'{learning_rates[i]:.2e}' if i < len(learning_rates) else 'nan'
                row = f'{i + 1},{train_losses[i]:.3e},{val_losses[i]:.3e},{min(val_losses[:i + 1]):.3e},{train_corr_val},{val_corr_val},{max_corr_val},{train_cmb_corr_val},{val_cmb_corr_val},{max_cmb_corr_val},{train_target_mse_val},{train_pred_vs_target_mse_val},{val_target_mse_val},{val_pred_vs_target_mse_val},{lr_val}'
                if config.get('log_gradients', False):
                    if i < len(gradient_stats_history) and gradient_stats_history[i] is not None:
                        grad_stats = gradient_stats_history[i]
                        row += f",{grad_stats['total_norm']:.2e},{grad_stats['max_norm']:.2e},{grad_stats['mean_norm']:.2e}"
                    else:
                        row += ',nan,nan,nan'
                f.write(row + '\n')
    final_epoch = epoch + 1 if early_stopped else config['num_epochs']
    final_model_path = os.path.join(output_dir, f'final_model_{file_id}.pt')
    torch.save({'epoch': final_epoch, 'model_state_dict': model.state_dict(), 'val_loss': avg_val_loss, 'best_val_loss': best_val_loss, 'val_correlation': avg_val_corr, 'best_val_correlation': best_val_correlation, 'early_stopped': early_stopped, 'config': config}, final_model_path)
    print(f'\n[ok] Final model saved to {final_model_path}')
    print(f'  Final epoch: {final_epoch}, Early stopped: {early_stopped}')
    print(f'  Final val loss: {avg_val_loss:.3e}, Best val loss: {best_val_loss:.3e}')
    if avg_val_corr is not None:
        print(f'  Final val correlation: {avg_val_corr:.3f}, Best val correlation: {best_val_correlation:.3f}')
    print('\n' + '=' * 80)
    print('EVALUATING ON TEST SET (HELD-OUT DATA)')
    print('=' * 80)
    if not os.path.exists(best_model_path):
        print(f'\nWARNING: WARNING: Best model not found at {best_model_path}')
        print('  Skipping test evaluation (no best model to evaluate)')
        test_loss = float('inf')
    else:
        print(f'\nLoading best model from: {best_model_path}')
        best_checkpoint = torch.load(best_model_path)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        model.eval()
        batch_test_losses = []
        test_cmb_corr_list = []
        test_target_mse_list = []
        test_pred_vs_target_mse_list = []
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc='Test Set Evaluation')
            for (x, y, indices) in test_pbar:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)
                loss = F.mse_loss(pred, y)
                batch_test_losses.append(loss.item())
                indices_np = indices.numpy()
                cmb_draw_batch = data_dict['cmb_draw']['test'][indices_np].to(device, non_blocking=True)
                y_flat = y.squeeze(1) if len(y.shape) == 4 else y
                pred_flat = pred.squeeze(1) if len(pred.shape) == 4 else pred
                batch_cmb_corrs = compute_cmb_reconstruction_correlation(y_flat, pred_flat, cmb_draw_batch)
                test_cmb_corr_list.extend(batch_cmb_corrs)
                zero_batch = torch.zeros_like(y_flat)
                batch_target_mse = mse(y_flat, zero_batch)
                if batch_target_mse is not None:
                    test_target_mse_list.append(batch_target_mse)
                batch_pred_vs_target_mse = mse(pred_flat, y_flat)
                if batch_pred_vs_target_mse is not None:
                    test_pred_vs_target_mse_list.append(batch_pred_vs_target_mse)
                test_pbar.set_postfix({'loss': f'{loss.item():.3e}'})
        test_loss = np.mean(batch_test_losses)
        test_cmb_corr = np.mean(test_cmb_corr_list) if test_cmb_corr_list else None
        test_target_mse = np.mean(test_target_mse_list) if test_target_mse_list else None
        test_pred_vs_target_mse = np.mean(test_pred_vs_target_mse_list) if test_pred_vs_target_mse_list else None
        print(f'\n[ok] Test set evaluation complete')
        print(f'  Test loss (from best model): {test_loss:.3e}')
        test_cmb_corr_str = f'{test_cmb_corr:.3f}' if test_cmb_corr is not None else 'N/A'
        print(f'  Test CMB correlation: {test_cmb_corr_str}')
        test_target_mse_str = f'{test_target_mse:.3e}' if test_target_mse is not None else 'N/A'
        test_pred_vs_target_mse_str = f'{test_pred_vs_target_mse:.3e}' if test_pred_vs_target_mse is not None else 'N/A'
        print(f'  Test Target MSE: {test_target_mse_str}, Test Pred vs Target MSE: {test_pred_vs_target_mse_str}')
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
    print(f'[ok] Training curves saved to {plot_path}')

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
    print(f'\n[ok] Summary saved to {summary_path}')
    print('=' * 80)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train UNet for single-frequency inter-scale (B, T, E) @ 220 GHz', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-normalize', action='store_true', help='Disable data normalization (default: normalize using train stats)')
    parser.add_argument('--resume-best', action='store_true', help='Resume training from best saved model')
    parser.add_argument('--b-only', action='store_true', help='Use only B small-scale channel (1 channel), skip T and E channels')
    parser.add_argument('--t-only', action='store_true', help='Use only T channel (1 channel), skip B small-scale and E channels')
    parser.add_argument('--e-only', action='store_true', help='Use only E channel (1 channel), skip B small-scale and T channels')
    parser.add_argument('--te-only', action='store_true', help='Use only T and E channels (2 channels), skip B small-scale channel')
    parser.add_argument('--te', action='store_true', help='Use T and E channels in addition to B small-scale (3 channels, default)')
    parser.add_argument('--test-mode', action='store_true', help='Test mode: Use only first 100 indices from loaded data (for debugging overfitting)')
    args = parser.parse_args()
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
    if args.b_only:
        input_mode = 'b_only'
        channel_desc = '1 (B small-scale only)'
        n_channels = 1
    elif args.t_only:
        input_mode = 't_only'
        channel_desc = '1 (T only)'
        n_channels = 1
    elif args.e_only:
        input_mode = 'e_only'
        channel_desc = '1 (E only)'
        n_channels = 1
    elif args.te_only:
        input_mode = 'te_only'
        channel_desc = '2 (T + E only)'
        n_channels = 2
    else:
        input_mode = 'bte'
        channel_desc = '3 (B small-scale + T + E)'
        n_channels = 3
    print(f"\nNormalization: {('Enabled' if normalize else 'Disabled')}")
    print(f"Resume from best: {('Yes' if resume_from_best else 'No')}")
    print(f'Input channels: {channel_desc}')
    print(f"Test mode: {('Enabled (using first 100 indices)' if test_mode else 'Disabled')}")
    if test_mode:
        print('\nWARNING: TEST MODE ENABLED: This will limit all datasets to first 100 indices')
        print('  Use this mode to test for overfitting with a smaller dataset')
    config = CONFIG.copy()
    config['in_channels'] = n_channels
    data_dict = load_and_prepare_data(normalize=normalize, input_mode=input_mode, test_mode=test_mode)
    (train_losses, val_losses, test_loss, output_dir, train_correlations, val_correlations, train_cmb_correlations, val_cmb_correlations) = train_model(data_dict, config, resume_from_best=resume_from_best, normalize=normalize, input_mode=input_mode, test_mode=test_mode)
    plot_training_curves(train_losses, val_losses, output_dir)
    print_training_summary(train_losses, val_losses, test_loss, output_dir, train_correlations, val_correlations, train_cmb_correlations, val_cmb_correlations)
    print('\n' + '=' * 80)
    print('TRAINING COMPLETE!')
    print('=' * 80)
    print(f'All results saved to: {output_dir}')
    print('=' * 80)
if __name__ == '__main__':
    main()
