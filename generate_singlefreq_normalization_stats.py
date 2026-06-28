#!/usr/bin/env python3
"""
Generate normalization statistics file for single-frequency UNet training.

This script creates singlefreq_normalization_stats.npz which contains:
- Per-channel input mean and std (for B small-scale, T, E @ 220 GHz)
- Target mean and std (for B large-scale @ 220 GHz)
- Train/valid/test split indices
- Metadata

Usage:
    python generate_singlefreq_normalization_stats.py
"""

import numpy as np
import os
import argparse

import paths_config

B_SMALL_DIR = paths_config.B_SMALL_DIR
T_E_DIR = paths_config.T_E_DIR
STATS_FILE = paths_config.STATS_FILE_SINGLEFREQ

# Single frequency
FREQUENCY = 220  # GHz

# Split ratios
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

def load_data():
    """Load all data files needed for statistics computation."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load B small-scale (input channel 0)
    b_small_file = f"{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_B_patches_small.npy"
    print(f"\nLoading B small-scale: {b_small_file}")
    if not os.path.exists(b_small_file):
        raise FileNotFoundError(f"B small-scale file not found: {b_small_file}")
    b_small = np.load(b_small_file)
    print(f"  Shape: {b_small.shape}")
    
    # Load T all-scale (input channel 1)
    t_file = f"{T_E_DIR}/sim1-150_freq{FREQUENCY}_T_patches.npy"
    print(f"\nLoading T all-scale: {t_file}")
    if not os.path.exists(t_file):
        raise FileNotFoundError(f"T patches file not found: {t_file}")
    t_all = np.load(t_file)
    print(f"  Shape: {t_all.shape}")
    
    # Load E all-scale (input channel 2)
    e_file = f"{T_E_DIR}/sim1-150_freq{FREQUENCY}_E_patches.npy"
    print(f"\nLoading E all-scale: {e_file}")
    if not os.path.exists(e_file):
        raise FileNotFoundError(f"E patches file not found: {e_file}")
    e_all = np.load(e_file)
    print(f"  Shape: {e_all.shape}")
    
    # Load B large-scale (target)
    b_large_file = f"{B_SMALL_DIR}/sim1-150_freq{FREQUENCY}_B_patches_large.npy"
    print(f"\nLoading B large-scale (target): {b_large_file}")
    if not os.path.exists(b_large_file):
        raise FileNotFoundError(f"B large-scale file not found: {b_large_file}")
    b_large = np.load(b_large_file)
    print(f"  Shape: {b_large.shape}")
    
    # Verify consistency
    n_patches = b_small.shape[0]
    assert b_large.shape[0] == n_patches, f"Mismatch: B large has {b_large.shape[0]} patches, expected {n_patches}"
    assert t_all.shape[0] == n_patches, f"Mismatch: T all has {t_all.shape[0]} patches, expected {n_patches}"
    assert e_all.shape[0] == n_patches, f"Mismatch: E all has {e_all.shape[0]} patches, expected {n_patches}"
    
    print(f"\n✓ All datasets have {n_patches} patches")
    
    return b_small, t_all, e_all, b_large, n_patches

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

def compute_statistics(b_small, t_all, e_all, b_large, train_indices):
    """Compute normalization statistics from training set only."""
    print("\n" + "="*80)
    print("COMPUTING NORMALIZATION STATISTICS (TRAIN SET ONLY)")
    print("="*80)
    
    # Extract training data
    train_b_small = b_small[train_indices]
    train_t_all = t_all[train_indices]
    train_e_all = e_all[train_indices]
    train_b_large = b_large[train_indices]
    
    # Stack input channels: [B_small, T, E]
    train_input = np.stack([train_b_small, train_t_all, train_e_all], axis=1)  # (n_train, 3, H, W)
    
    # Compute per-channel input statistics
    input_mean = np.mean(train_input, axis=(0, 2, 3))  # (3,)
    input_std = np.std(train_input, axis=(0, 2, 3))     # (3,)
    input_min = np.min(train_input, axis=(0, 2, 3))    # (3,)
    input_max = np.max(train_input, axis=(0, 2, 3))    # (3,)
    
    # Compute target statistics
    target_mean = np.mean(train_b_large)
    target_std = np.std(train_b_large)
    target_min = np.min(train_b_large)
    target_max = np.max(train_b_large)
    
    # Global input statistics (across all channels)
    global_input_mean = np.mean(train_input)
    global_input_std = np.std(train_input)
    
    print(f"\nInput statistics (per-channel):")
    print(f"  Channel 0 (B small-scale):")
    print(f"    Mean: {input_mean[0]:.6e}, Std: {input_std[0]:.6e}")
    print(f"    Min:  {input_min[0]:.6e}, Max: {input_max[0]:.6e}")
    print(f"  Channel 1 (T all-scale):")
    print(f"    Mean: {input_mean[1]:.6e}, Std: {input_std[1]:.6e}")
    print(f"    Min:  {input_min[1]:.6e}, Max: {input_max[1]:.6e}")
    print(f"  Channel 2 (E all-scale):")
    print(f"    Mean: {input_mean[2]:.6e}, Std: {input_std[2]:.6e}")
    print(f"    Min:  {input_min[2]:.6e}, Max: {input_max[2]:.6e}")
    
    print(f"\nGlobal input statistics:")
    print(f"  Mean: {global_input_mean:.6e}, Std: {global_input_std:.6e}")
    
    print(f"\nTarget statistics (B large-scale):")
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

def save_statistics(stats, train_indices, valid_indices, test_indices, n_patches):
    """Save normalization statistics to .npz file."""
    print("\n" + "="*80)
    print("SAVING STATISTICS")
    print("="*80)
    
    # Channel names
    input_channel_names = np.array(['B_small_220GHz', 'T_220GHz', 'E_220GHz'], dtype='<U14')
    
    # Prepare data to save
    save_dict = {
        'input_mean': stats['input_mean'],
        'input_std': stats['input_std'],
        'input_min': stats['input_min'],
        'input_max': stats['input_max'],
        'input_channel_names': input_channel_names,
        'target_mean': np.float32(stats['target_mean']),  # Match existing format
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
        'random_seed': RANDOM_SEED,
        'global_input_mean': stats['global_input_mean'],
        'global_input_std': stats['global_input_std']
    }
    
    # Save to file
    print(f"\nSaving to: {STATS_FILE}")
    np.savez(STATS_FILE, **save_dict)
    
    print(f"✓ Statistics saved successfully!")
    print(f"\nFile contents:")
    print(f"  - input_mean: {stats['input_mean'].shape} (per-channel mean)")
    print(f"  - input_std: {stats['input_std'].shape} (per-channel std)")
    print(f"  - target_mean: scalar (target mean)")
    print(f"  - target_std: scalar (target std)")
    print(f"  - train_indices: {len(train_indices)} indices")
    print(f"  - valid_indices: {len(valid_indices)} indices")
    print(f"  - test_indices: {len(test_indices)} indices")

def main():
    """Main function."""
    print("="*80)
    print("GENERATE SINGLE-FREQUENCY NORMALIZATION STATISTICS")
    print("="*80)
    print(f"Frequency: {FREQUENCY} GHz")
    print(f"Output file: {STATS_FILE}")
    print("="*80)
    
    # Load data
    b_small, t_all, e_all, b_large, n_patches = load_data()
    
    # Create train/valid/test split
    train_indices, valid_indices, test_indices = create_train_valid_test_split(n_patches, RANDOM_SEED)
    
    # Compute statistics from training set only
    stats = compute_statistics(b_small, t_all, e_all, b_large, train_indices)
    
    # Save statistics
    save_statistics(stats, train_indices, valid_indices, test_indices, n_patches)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Normalization statistics file created: {STATS_FILE}")
    print("You can now run train_unet_singlefreq.py")
    print("="*80)

if __name__ == "__main__":
    main()

