#!/usr/bin/env python3
"""
Generate normalization statistics file for inter-scale multi-frequency UNet training.

This script creates interscale_multifreq_normalization_stats.npz which contains:
- Per-channel input mean and std (for 8 or 16 channels)
- Target mean and std (for ILC residuals)
- Train/valid/test split indices
- Metadata

Channel layout (16 channels):
- Channels 0-3: ILC foreground reconstructions @ 95, 145, 220, 270 GHz
- Channels 4-7: Small-scale B-modes (ℓ > 200) @ 95, 145, 220, 270 GHz
- Channels 8-11: All-scale E-modes @ 95, 145, 220, 270 GHz
- Channels 12-15: All-scale T-modes @ 95, 145, 220, 270 GHz

Usage:
    # Generate with 16 channels (includes E and T modes)
    python generate_interscale_normalization_stats.py --include-e-t

    # Generate with 8 channels (ILC + B small only, default)
    python generate_interscale_normalization_stats.py
"""

import numpy as np
import os
import argparse

import paths_config

ILC_DIR = paths_config.ILC_DIR
B_SMALL_DIR = paths_config.B_SMALL_DIR
E_T_DIR = paths_config.T_E_DIR

STATS_FILE_8CH = paths_config.STATS_FILE_8CH
STATS_FILE_16CH = paths_config.STATS_FILE_16CH

# Frequencies
FREQUENCIES = [95, 145, 220, 270]

# Split ratios
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

def load_data(include_e_t=False):
    """Load all data files needed for statistics computation."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load ILC foregrounds (channels 0-3)
    ILC_foregrounds = {}
    print("\nLoading ILC foregrounds...")
    for freq in FREQUENCIES:
        file_path = f"{ILC_DIR}/ILC_foregrounds_b_{freq}.npy"
        print(f"  Loading {freq} GHz: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ILC foregrounds file not found: {file_path}")
        ILC_foregrounds[freq] = np.load(file_path)
        print(f"    Shape: {ILC_foregrounds[freq].shape}")
    
    # Load B small-scale (channels 4-7)
    B_small = {}
    print("\nLoading B small-scale patches...")
    for freq in FREQUENCIES:
        file_path = f"{B_SMALL_DIR}/sim1-150_freq{freq}_B_patches_small.npy"
        print(f"  Loading {freq} GHz: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"B small-scale file not found: {file_path}")
        B_small[freq] = np.load(file_path)
        print(f"    Shape: {B_small[freq].shape}")
    
    # Load E and T modes (channels 8-15) if requested
    E_modes = {}
    T_modes = {}
    if include_e_t:
        print("\nLoading E and T mode patches...")
        for freq in FREQUENCIES:
            # E modes (channels 8-11)
            e_file = f"{E_T_DIR}/sim1-150_freq{freq}_E_patches.npy"
            print(f"  Loading E mode {freq} GHz: {e_file}")
            if not os.path.exists(e_file):
                raise FileNotFoundError(f"E mode file not found: {e_file}")
            E_modes[freq] = np.load(e_file)
            print(f"    Shape: {E_modes[freq].shape}")
            
            # T modes (channels 12-15)
            t_file = f"{E_T_DIR}/sim1-150_freq{freq}_T_patches.npy"
            print(f"  Loading T mode {freq} GHz: {t_file}")
            if not os.path.exists(t_file):
                raise FileNotFoundError(f"T mode file not found: {t_file}")
            T_modes[freq] = np.load(t_file)
            print(f"    Shape: {T_modes[freq].shape}")
    
    # Load ILC residuals (target)
    ilc_residuals_file = f"{ILC_DIR}/ilc_residuals_b.npy"
    print(f"\nLoading ILC residuals (target): {ilc_residuals_file}")
    if not os.path.exists(ilc_residuals_file):
        raise FileNotFoundError(f"ILC residuals file not found: {ilc_residuals_file}")
    ilc_residuals = np.load(ilc_residuals_file)
    print(f"  Shape: {ilc_residuals.shape}")
    
    # Verify consistency
    n_patches = ilc_residuals.shape[0]
    ref_spatial_shape = ilc_residuals.shape[1:]
    
    for freq in FREQUENCIES:
        assert ILC_foregrounds[freq].shape[0] == n_patches, \
            f"Mismatch: ILC foregrounds at {freq} GHz has {ILC_foregrounds[freq].shape[0]} patches"
        assert ILC_foregrounds[freq].shape[1:] == ref_spatial_shape, \
            f"Mismatch: ILC foregrounds at {freq} GHz spatial shape mismatch"
        
        assert B_small[freq].shape[0] == n_patches, \
            f"Mismatch: B small at {freq} GHz has {B_small[freq].shape[0]} patches"
        assert B_small[freq].shape[1:] == ref_spatial_shape, \
            f"Mismatch: B small at {freq} GHz spatial shape mismatch"
        
        if include_e_t:
            assert E_modes[freq].shape[0] == n_patches, \
                f"Mismatch: E mode at {freq} GHz has {E_modes[freq].shape[0]} patches"
            assert E_modes[freq].shape[1:] == ref_spatial_shape, \
                f"Mismatch: E mode at {freq} GHz spatial shape mismatch"
            
            assert T_modes[freq].shape[0] == n_patches, \
                f"Mismatch: T mode at {freq} GHz has {T_modes[freq].shape[0]} patches"
            assert T_modes[freq].shape[1:] == ref_spatial_shape, \
                f"Mismatch: T mode at {freq} GHz spatial shape mismatch"
    
    print(f"\n✓ All datasets have {n_patches} patches with spatial shape {ref_spatial_shape}")
    
    return ILC_foregrounds, B_small, E_modes, T_modes, ilc_residuals, n_patches

def create_train_valid_test_split(n_patches, random_seed=42):
    """Create train/valid/test split indices."""
    print("\n" + "="*80)
    print("CREATING TRAIN/VALID/TEST SPLIT")
    print("="*80)
    
    np.random.seed(random_seed)
    
    # Create shuffled indices
    all_indices = np.arange(n_patches)
    np.random.shuffle(all_indices)
    
    # Calculate split sizes
    n_train = int(n_patches * TRAIN_RATIO)
    n_valid = int(n_patches * VALID_RATIO)
    n_test = n_patches - n_train - n_valid  # Remaining goes to test
    
    # Split indices
    train_indices = all_indices[:n_train]
    valid_indices = all_indices[n_train:n_train + n_valid]
    test_indices = all_indices[n_train + n_valid:]
    
    print(f"  Total patches: {n_patches}")
    print(f"  Train: {len(train_indices)} ({len(train_indices)/n_patches*100:.1f}%)")
    print(f"  Valid: {len(valid_indices)} ({len(valid_indices)/n_patches*100:.1f}%)")
    print(f"  Test:  {len(test_indices)} ({len(test_indices)/n_patches*100:.1f}%)")
    
    return train_indices, valid_indices, test_indices

def compute_statistics(ILC_foregrounds, B_small, E_modes, T_modes, ilc_residuals, train_indices, include_e_t=False):
    """Compute normalization statistics from training set only.
    
    IMPORTANT: This function computes ALL statistics (mean, std, min, max) using
    only the training set indices. This prevents data leakage from validation/test sets.
    The train_indices are used to extract training data before computing any statistics.
    
    Args:
        train_indices: Array of indices for the training set (from train/valid/test split)
        ... (other args)
    
    Returns:
        Dictionary containing statistics computed from TRAIN SET ONLY
    """
    print("\n" + "="*80)
    print("COMPUTING NORMALIZATION STATISTICS (TRAIN SET ONLY)")
    print("="*80)
    print("⚠ IMPORTANT: All statistics are computed from TRAIN SET ONLY")
    print(f"  Using {len(train_indices)} training samples (validation/test sets excluded)")
    
    # Extract training data
    train_ilc_residuals = ilc_residuals[train_indices]
    
    # Build input channels
    input_channels = []
    
    # Channels 0-3: ILC foregrounds
    for freq in FREQUENCIES:
        input_channels.append(ILC_foregrounds[freq][train_indices])
    
    # Channels 4-7: B small-scale
    for freq in FREQUENCIES:
        input_channels.append(B_small[freq][train_indices])
    
    # Channels 8-15: E and T modes (if included)
    if include_e_t:
        # Channels 8-11: E modes
        for freq in FREQUENCIES:
            input_channels.append(E_modes[freq][train_indices])
        # Channels 12-15: T modes
        for freq in FREQUENCIES:
            input_channels.append(T_modes[freq][train_indices])
    
    # Stack input channels
    train_input = np.stack(input_channels, axis=1)  # (n_train, n_channels, H, W)
    
    # Compute per-channel input statistics
    input_mean = np.mean(train_input, axis=(0, 2, 3))  # (n_channels,)
    input_std = np.std(train_input, axis=(0, 2, 3))     # (n_channels,)
    input_min = np.min(train_input, axis=(0, 2, 3))    # (n_channels,)
    input_max = np.max(train_input, axis=(0, 2, 3))    # (n_channels,)
    
    # Compute target statistics
    target_mean = np.mean(train_ilc_residuals)
    target_std = np.std(train_ilc_residuals)
    target_min = np.min(train_ilc_residuals)
    target_max = np.max(train_ilc_residuals)
    
    # Global input statistics (across all channels)
    global_input_mean = np.mean(train_input)
    global_input_std = np.std(train_input)
    
    n_channels = len(input_channels)
    print(f"\nInput statistics ({n_channels} channels, per-channel):")
    
    # Print ILC foregrounds (channels 0-3)
    for i, freq in enumerate(FREQUENCIES):
        print(f"  Channel {i} (ILC foreground {freq} GHz):")
        print(f"    Mean: {input_mean[i]:.6e}, Std: {input_std[i]:.6e}")
        print(f"    Min:  {input_min[i]:.6e}, Max: {input_max[i]:.6e}")
    
    # Print B small-scale (channels 4-7)
    for i, freq in enumerate(FREQUENCIES):
        ch_idx = 4 + i
        print(f"  Channel {ch_idx} (B small-scale {freq} GHz):")
        print(f"    Mean: {input_mean[ch_idx]:.6e}, Std: {input_std[ch_idx]:.6e}")
        print(f"    Min:  {input_min[ch_idx]:.6e}, Max: {input_max[ch_idx]:.6e}")
    
    # Print E and T modes (channels 8-15) if included
    if include_e_t:
        # E modes (channels 8-11)
        for i, freq in enumerate(FREQUENCIES):
            ch_idx = 8 + i
            print(f"  Channel {ch_idx} (E mode {freq} GHz):")
            print(f"    Mean: {input_mean[ch_idx]:.6e}, Std: {input_std[ch_idx]:.6e}")
            print(f"    Min:  {input_min[ch_idx]:.6e}, Max: {input_max[ch_idx]:.6e}")
        
        # T modes (channels 12-15)
        for i, freq in enumerate(FREQUENCIES):
            ch_idx = 12 + i
            print(f"  Channel {ch_idx} (T mode {freq} GHz):")
            print(f"    Mean: {input_mean[ch_idx]:.6e}, Std: {input_std[ch_idx]:.6e}")
            print(f"    Min:  {input_min[ch_idx]:.6e}, Max: {input_max[ch_idx]:.6e}")
    
    print(f"\nGlobal input statistics:")
    print(f"  Mean: {global_input_mean:.6e}, Std: {global_input_std:.6e}")
    
    print(f"\nTarget statistics (ILC residuals):")
    print(f"  Mean: {target_mean:.6e}, Std: {target_std:.6e}")
    print(f"  Min:  {target_min:.6e}, Max: {target_max:.6e}")
    
    return {
        'input_mean': input_mean,
        'input_std': input_std,
        'input_min': input_min,
        'input_max': input_max,
        'target_mean': target_mean,
        'target_std': target_std,
        'target_min': target_min,
        'target_max': target_max,
        'global_input_mean': global_input_mean,
        'global_input_std': global_input_std
    }

def save_statistics(stats, train_indices, valid_indices, test_indices, n_patches, include_e_t=False):
    """Save normalization statistics to .npz file.
    
    IMPORTANT: All statistics (mean, std, min, max) are computed from TRAIN SET ONLY.
    This ensures that normalization statistics do not leak information from validation/test sets.
    """
    print("\n" + "="*80)
    print("SAVING STATISTICS")
    print("="*80)
    
    n_channels = 16 if include_e_t else 8
    
    # Select output file based on number of channels
    stats_file = STATS_FILE_16CH if include_e_t else STATS_FILE_8CH
    
    # Create channel names
    channel_names = []
    # Channels 0-3: ILC foregrounds
    for freq in FREQUENCIES:
        channel_names.append(f'ILC_foreground_{freq}GHz')
    # Channels 4-7: B small-scale
    for freq in FREQUENCIES:
        channel_names.append(f'B_small_{freq}GHz')
    # Channels 8-15: E and T modes (if included)
    if include_e_t:
        # Channels 8-11: E modes
        for freq in FREQUENCIES:
            channel_names.append(f'E_mode_{freq}GHz')
        # Channels 12-15: T modes
        for freq in FREQUENCIES:
            channel_names.append(f'T_mode_{freq}GHz')
    
    input_channel_names = np.array(channel_names, dtype='<U20')
    
    # Prepare data to save
    save_dict = {
        'input_mean': stats['input_mean'],
        'input_std': stats['input_std'],
        'input_min': stats['input_min'],
        'input_max': stats['input_max'],
        'input_channel_names': input_channel_names,
        'target_mean': np.float32(stats['target_mean']),
        'target_std': np.float32(stats['target_std']),
        'target_min': np.float32(stats['target_min']),
        'target_max': np.float32(stats['target_max']),
        'train_indices': train_indices,
        'valid_indices': valid_indices,
        'test_indices': test_indices,
        'n_train': len(train_indices),
        'n_valid': len(valid_indices),
        'n_test': len(test_indices),
        'n_total': n_patches,
        'n_channels': n_channels,
        'random_seed': RANDOM_SEED,
        'global_input_mean': stats['global_input_mean'],
        'global_input_std': stats['global_input_std'],
        'include_e_t': include_e_t,
        'computed_from_train_only': True  # Flag to indicate stats are from train set only
    }
    
    # Save to file
    print(f"\nSaving to: {stats_file}")
    print(f"  ⚠ IMPORTANT: All statistics computed from TRAIN SET ONLY (no data leakage)")
    np.savez(stats_file, **save_dict)
    
    print(f"✓ Statistics saved successfully!")
    print(f"\nFile contents:")
    print(f"  - input_mean: {stats['input_mean'].shape} (per-channel mean, {n_channels} channels)")
    print(f"  - input_std: {stats['input_std'].shape} (per-channel std, {n_channels} channels)")
    print(f"  - target_mean: scalar (target mean)")
    print(f"  - target_std: scalar (target std)")
    print(f"  - train_indices: {len(train_indices)} indices")
    print(f"  - valid_indices: {len(valid_indices)} indices")
    print(f"  - test_indices: {len(test_indices)} indices")
    print(f"  - include_e_t: {include_e_t}")
    print(f"  - computed_from_train_only: True (all stats from train set)")
    return stats_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate normalization statistics for inter-scale multi-frequency UNet training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--include-e-t', action='store_true',
                       help='Include E and T mode channels (16 channels total instead of 8)')
    
    args = parser.parse_args()
    
    # Determine output file
    stats_file = STATS_FILE_16CH if args.include_e_t else STATS_FILE_8CH
    
    print("="*80)
    print("GENERATE INTER-SCALE MULTI-FREQUENCY NORMALIZATION STATISTICS")
    print("="*80)
    print(f"Include E and T modes: {args.include_e_t}")
    print(f"Number of channels: {16 if args.include_e_t else 8}")
    print(f"Output file: {stats_file}")
    print("="*80)
    
    # Load data
    ILC_foregrounds, B_small, E_modes, T_modes, ilc_residuals, n_patches = load_data(include_e_t=args.include_e_t)
    
    # Create train/valid/test split
    train_indices, valid_indices, test_indices = create_train_valid_test_split(n_patches, RANDOM_SEED)
    
    # Compute statistics from training set only
    # IMPORTANT: Only train_indices are used to compute statistics (no data leakage)
    stats = compute_statistics(ILC_foregrounds, B_small, E_modes, T_modes, ilc_residuals, 
                              train_indices, include_e_t=args.include_e_t)
    
    # Save statistics (returns the actual file path used)
    saved_file = save_statistics(stats, train_indices, valid_indices, test_indices, n_patches, include_e_t=args.include_e_t)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Normalization statistics file created: {saved_file}")
    print(f"Number of channels: {16 if args.include_e_t else 8}")
    print("⚠ IMPORTANT: All statistics computed from TRAIN SET ONLY (no data leakage)")
    print("\nYou can now run train_unet_interscale.py")
    if args.include_e_t:
        print("  Use --add-e-t flag when training to use all 16 channels")
        print(f"  The training script will look for: {STATS_FILE_16CH}")
    else:
        print(f"  The training script will look for: {STATS_FILE_8CH}")
    print("="*80)

if __name__ == "__main__":
    main()

