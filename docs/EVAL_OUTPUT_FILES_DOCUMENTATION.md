# Evaluation Script Output Files Documentation

This document provides a comprehensive explanation of all files saved by `eval_test_results_singlefreq.py`.

## Overview

The evaluation script saves results in the output directory. The output includes:
- **2 .npz files**: Binary NumPy archive files containing arrays and data
- **3 .txt files**: Text files with statistics and summary information
- **Visualization directories**: PNG files for individual sample visualizations (if `--no-viz` is not used)

## Output Directory Structure

### Base Output Directory

The base output directory is defined in the script:
```
OUTPUT_DIR = "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled"
```

### Output Subdirectories

Results are saved in subdirectories based on normalization mode and input channel configuration:

**Format**: `{OUTPUT_DIR}/{norm_suffix}_{channel_suffix}/`

Where:
- `norm_suffix`: `"normalized"` or `"unnormalized"` (default: `"unnormalized"`)
- `channel_suffix`: 
  - `"teb"` (default, 3 channels: B + T + E)
  - `"te_only"` (2 channels: T + E only)
  - `"e_only"` (1 channel: E only)
  - `"t_only"` (1 channel: T only)
  - `"b_only"` (1 channel: B only)

### Complete Output Directory Paths

#### Unnormalized Models (Default)

1. **B + T + E (3 channels, default)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/
   ```

2. **T + E only (2 channels)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_te_only/
   ```

3. **E only (1 channel)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/
   ```

4. **T only (1 channel)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_t_only/
   ```

5. **B only (1 channel)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_b_only/
   ```

#### Normalized Models (if `--normalize` flag is used)

1. **B + T + E (3 channels)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_teb/
   ```

2. **T + E only (2 channels)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_te_only/
   ```

3. **E only (1 channel)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_e_only/
   ```

4. **T only (1 channel)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_t_only/
   ```

5. **B only (1 channel)**:
   ```
   /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_b_only/
   ```

### Custom Output Directory

You can override the default output directory using the `--output-dir` argument:
```bash
python eval_test_results_singlefreq.py --e-only --output-dir /path/to/custom/directory
```

### Files Saved in Each Output Directory

Each output directory contains:

**Binary Files (.npz)**:
- `test_results.npz` - All test patches data
- `first_sample_data.npz` - First sample + summary statistics

**Text Files (.txt)**:
- `{model_identifier}_cross_spectrum_stats.txt` - Cross-spectrum statistics
- `{model_identifier}_correlation_stats.txt` - Correlation statistics
- `{model_identifier}_mse_stats.txt` - MSE statistics

**Visualization Directories** (if `--no-viz` is not used):
- `cmb_reconstructions/` - CMB reconstruction plots
- `fg_reconstructions/` - Foreground reconstruction plots
- `input_channels/` - Input channel visualizations

**Model Identifier Format**: `{norm_suffix}_{channel_suffix}_{model_basename}`
- Example: `unnormalized_e_only_best_model_153600_19200_nonorm`
- Example: `normalized_te_only_best_model_153600_19200_norm`

### Summary of All Output Directories

| Input Mode | Normalization | Full Output Directory Path |
|------------|---------------|----------------------------|
| B + T + E (default) | Unnormalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/` |
| T + E only | Unnormalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_te_only/` |
| E only | Unnormalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/` |
| T only | Unnormalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_t_only/` |
| B only | Unnormalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_b_only/` |
| B + T + E (default) | Normalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_teb/` |
| T + E only | Normalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_te_only/` |
| E only | Normalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_e_only/` |
| T only | Normalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_t_only/` |
| B only | Normalized | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/normalized_b_only/` |

---

## 1. Binary Data Files (.npz)

### 1.1 `test_results.npz`

**Purpose**: Main test results file containing all test set data for comprehensive analysis.

**Location**: 
- Full path example: `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/test_results.npz`
- Relative to output directory: `{output_dir}/test_results.npz`

**Contents**:

#### UNet Inputs and Outputs
- **`unet_predictions`** (numpy array, shape: `(n_subsample, H, W)`)
  - UNet predictions of large-scale foreground B-modes (ℓ < 200)
  - This is what the model predicts for the large-scale B-mode foreground component
  - Spatial resolution: 128×128 pixels per patch
  - Units: μK (microkelvin)

- **`unet_targets`** (numpy array, shape: `(n_subsample, H, W)`)
  - True large-scale foreground B-modes (ℓ < 200) - the ground truth target
  - This is what the model is trying to predict
  - Same spatial resolution as predictions
  - Used to compute prediction accuracy

#### Additional Datasets (All Aligned Using Same test_indices)
- **`observed_b_all_scales`** (numpy array, shape: `(n_subsample, H, W)`)
  - Total observed B-mode signal: `CMB + B_large + B_small` (foregrounds at all scales)
  - Represents what would be observed in a real observation
  - Formula: `observed_b_all_scales = cmb_draw + b_large + b_small`
  - Used for CMB reconstruction: `unet_cmb_reconstructed = observed_b_all_scales - b_small - unet_predicted_b_large`

- **`cmb_draw`** (numpy array, shape: `(n_subsample, H, W)`)
  - Pure primordial CMB B-modes (ground truth CMB, no foregrounds)
  - This is the "true" CMB signal that we want to recover
  - Scaled by 0.05 (as in training)
  - Used to evaluate CMB reconstruction quality

- **`unet_cmb_reconstructed`** (numpy array, shape: `(n_subsample, H, W)`)
  - UNet-improved CMB reconstruction
  - Formula: `unet_cmb_reconstructed = observed_B - b_small - unet_predicted_b_large`
  - This is the CMB signal after removing predicted foregrounds
  - Compare with `cmb_draw` to assess CMB reconstruction quality

- **`test_b_small`** (numpy array, shape: `(n_subsample, H, W)`)
  - Small-scale foreground B-modes (ℓ > 200)
  - Part of UNet input (if using B channel)
  - Used in CMB reconstruction formula

#### Correlations (All Shape: `(n_subsample,)`)
- **`unet_spatial_correlations`** (numpy array, shape: `(n_subsample,)`)
  - Spatial correlation between `unet_cmb_reconstructed` and `pure_cmb` for each patch
  - Values range from -1 to 1
  - Higher values indicate better CMB reconstruction
  - Computed using: `np.corrcoef(unet_cmb_reconstructed.flatten(), cmb_draw.flatten())[0, 1]`

- **`pred_vs_target_correlations`** (numpy array, shape: `(n_subsample,)`)
  - Spatial correlation between `prediction` and `target` for each patch
  - Values range from -1 to 1
  - Higher values indicate better foreground prediction
  - Computed using: `np.corrcoef(prediction.flatten(), target.flatten())[0, 1]`

#### Null Correlations (All Shape: `(n_subsample,)`)
- **`null_correlations_array`** (numpy array, shape: `(n_subsample,)`)
  - Null correlations for CMB reconstructions
  - For each patch: correlation between `unet_cmb_reconstructed` and a random CMB patch (not matched)
  - Used to assess statistical significance
  - Should be close to 0 if the reconstruction is meaningful
  - Distribution of these values provides a null hypothesis baseline

- **`fg_null_correlations_array`** (numpy array, shape: `(n_subsample,)`)
  - Null correlations for foreground predictions
  - For each patch: correlation between `unet_prediction` and a random target patch (not matched)
  - Used to assess statistical significance of foreground predictions
  - Should be close to 0 if predictions are meaningful

#### Cross-Spectra (All Shape: `(n_subsample, n_ell_bins)` or `None`)
- **`unet_cross_spectra_array`** (numpy array, shape: `(n_subsample, n_ell_bins)` or `None`)
  - Cross-power spectrum: `unet_cmb_reconstructed × pure_cmb`
  - Normalized cross-power spectrum per ell bin, per patch
  - Each row is one patch's cross-spectrum across ell bins
  - Values are normalized: `cl_cross / sqrt(cl_map1_map1 * cl_map2_map2)`
  - Used to assess CMB reconstruction quality in Fourier space
  - `None` if `calculate_2d_spectrum` is not available

- **`pred_vs_target_cross_spectra_array`** (numpy array, shape: `(n_subsample, n_ell_bins)` or `None`)
  - Cross-power spectrum: `prediction × target`
  - Normalized cross-power spectrum per ell bin, per patch
  - Each row is one patch's cross-spectrum across ell bins
  - Used to assess foreground prediction quality in Fourier space
  - `None` if `calculate_2d_spectrum` is not available

- **`ell_array`** (numpy array, shape: `(n_ell_bins,)` or `None`)
  - Ell (multipole) values corresponding to cross-spectrum bins
  - Used for plotting cross-spectra
  - Typically ranges from ~25 to 200 (ELL_CUTOFF) with DELTA_ELL=50 bin width
  - `None` if cross-spectra are not computed

#### MSE Arrays (All Shape: `(n_subsample,)`)
- **`pred_vs_target_mse_array`** (numpy array, shape: `(n_subsample,)`)
  - MSE(prediction, target) for each patch
  - Mean Squared Error between UNet predictions and true targets
  - Lower values indicate better predictions
  - Units: (μK)²

- **`target_vs_zero_mse_array`** (numpy array, shape: `(n_subsample,)`)
  - MSE(target, zero_patch) for each patch
  - Baseline: MSE if we predicted zero (no signal)
  - Used to compute improvement ratio: `target_vs_zero_mse / pred_vs_target_mse`
  - If improvement ratio > 1, UNet is better than predicting zero
  - Units: (μK)²

#### Foreground All-Scales
- **`b_fg_all_scales`** (numpy array, shape: `(n_subsample, H, W)`)
  - Foreground B-mode all-scales: `b_small + b_large`
  - Total foreground signal at all scales
  - Used for foreground reconstruction visualization

#### Test Indices and Metadata
- **`test_indices`** (numpy array, shape: `(n_subsample,)`)
  - Indices into the full dataset that correspond to saved arrays
  - Allows mapping back to original dataset indices
  - After subsampling (if used), these are the subset of test indices
  - Example: If test set has indices [100, 101, 102, ..., 199] and you subsample 50, 
    `test_indices` might be [100, 101, ..., 149]

- **`normalize`** (boolean)
  - Whether data was normalized during evaluation
  - `True` if `--normalize` flag was used
  - `False` if unnormalized data was used (default)

- **`in_channels`** (integer)
  - Number of input channels used by the model
  - Values: 1 (b_only, t_only, e_only), 2 (te_only), or 3 (bte/default)

- **`channel_mode`** (string)
  - Channel mode suffix identifying the input configuration
  - Values: "b_only", "t_only", "e_only", "te_only", or "teb" (default)

**Number of Patches**:
- By default (`--n-subsample 0` or not specified): **All test patches** from the test set
- If `--n-subsample N` is specified: **First N patches** from the test set
- The actual number of test patches depends on the train/valid/test split defined in `singlefreq_normalization_stats.npz`
- Typical test set size: ~19,200 patches (out of ~153,600 total patches)

**Usage Example**:
```python
import numpy as np

# Load all data
with np.load('test_results.npz') as data:
    predictions = data['unet_predictions']  # Shape: (n_subsample, 128, 128)
    targets = data['unet_targets']          # Shape: (n_subsample, 128, 128)
    correlations = data['pred_vs_target_correlations']  # Shape: (n_subsample,)
    mse_values = data['pred_vs_target_mse_array']       # Shape: (n_subsample,)
    test_indices = data['test_indices']                 # Shape: (n_subsample,)
    
    print(f"Number of patches: {len(predictions)}")
    print(f"Mean correlation: {np.mean(correlations):.4f}")
    print(f"Mean MSE: {np.mean(mse_values):.6e}")
```

---

### 1.2 `first_sample_data.npz`

**Purpose**: Contains data for the first test sample plus summary statistics for all patches. Designed for easy notebook reproduction of plots.

**Location**: 
- Full path example: `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/first_sample_data.npz`
- Relative to output directory: `{output_dir}/first_sample_data.npz`

**Contents**:

#### Sample Identification
- **`sample_idx`** (integer)
  - Original dataset index of the first sample
  - This is the index in the full dataset (before subsampling)
  - Example: If test_indices = [100, 101, 102, ...], then sample_idx = 100

- **`array_idx`** (integer)
  - Position in subsampled arrays (always 0 for first sample)
  - This is the index in the arrays after subsampling
  - Always 0 because it's the first sample

#### CMB Reconstruction Patches (Shape: `(H, W)` = `(128, 128)`)
- **`cmb_draw`** (numpy array, shape: `(128, 128)`)
  - Pure primordial CMB B-mode patch for first sample
  - Ground truth CMB signal (no foregrounds)
  - Scaled by 0.05

- **`observed_b_all_scales`** (numpy array, shape: `(128, 128)`)
  - Observed B all-scales patch for first sample
  - Total observed signal: CMB + foregrounds

- **`unet_cmb_reconstructed`** (numpy array, shape: `(128, 128)`)
  - UNet CMB reconstructed patch for first sample
  - CMB after foreground removal

#### Foreground Reconstruction Patches (Shape: `(H, W)` = `(128, 128)`)
- **`b_fg_all_scales`** (numpy array, shape: `(128, 128)`)
  - Foreground B-mode all-scales patch for first sample
  - Total foreground: b_small + b_large

- **`target`** (numpy array, shape: `(128, 128)`)
  - True large-scale foreground B-modes patch for first sample
  - Ground truth target

- **`prediction`** (numpy array, shape: `(128, 128)`)
  - UNet prediction patch for first sample
  - Model's prediction of large-scale foreground

- **`b_small`** (numpy array, shape: `(128, 128)`)
  - Small-scale foreground B-modes patch for first sample
  - Part of UNet input (if using B channel)

#### Correlations for First Sample (Scalars)
- **`unet_spatial_correlation`** (float)
  - Spatial correlation: `unet_cmb_reconstructed` vs `pure_cmb` for first sample
  - Value in range [-1, 1]
  - Single scalar value

- **`pred_vs_target_correlation`** (float)
  - Spatial correlation: `prediction` vs `target` for first sample
  - Value in range [-1, 1]
  - Single scalar value

#### MSE Values for First Sample (Scalars)
- **`observed_b_mse`** (float)
  - MSE(pure_cmb, observed_b) for first sample
  - Single scalar value

- **`unet_recon_mse`** (float)
  - MSE(pure_cmb, unet_cmb_reconstructed) for first sample
  - Single scalar value

- **`pred_vs_target_mse`** (float)
  - MSE(prediction, target) for first sample
  - Single scalar value

- **`target_vs_zero_mse`** (float)
  - MSE(target, zero) for first sample
  - Baseline MSE value

#### Cross-Spectra for First Sample (If Available)
- **`ell_array`** (numpy array, shape: `(n_ell_bins,)` or `None`)
  - Ell values for cross-spectra
  - Same as in `test_results.npz`

- **`unet_cross_spectrum`** (numpy array, shape: `(n_ell_bins,)` or `None`)
  - UNet CMB × pure CMB cross-spectrum for first sample only
  - One row from `unet_cross_spectra_array[0]`

- **`mean_unet_cross_ps`** (numpy array, shape: `(n_ell_bins,)` or `None`)
  - Mean UNet cross-spectrum across all patches
  - Computed as: `np.nanmean(unet_cross_spectra_array, axis=0)`
  - Used for plotting mean ± error bands

- **`std_unet_cross_ps`** (numpy array, shape: `(n_ell_bins,)` or `None`)
  - Standard deviation of UNet cross-spectrum across all patches
  - Computed as: `np.nanstd(unet_cross_spectra_array, axis=0)`
  - Used for plotting error bands

- **`pred_vs_target_cross_spectrum`** (numpy array, shape: `(n_ell_bins,)` or `None`)
  - Prediction × target cross-spectrum for first sample only
  - One row from `pred_vs_target_cross_spectra_array[0]`

- **`mean_pred_vs_target_cross_ps`** (numpy array, shape: `(n_ell_bins,)` or `None`)
  - Mean prediction vs target cross-spectrum across all patches
  - Computed as: `np.nanmean(pred_vs_target_cross_spectra_array, axis=0)`

- **`std_pred_vs_target_cross_ps`** (numpy array, shape: `(n_ell_bins,)` or `None`)
  - Standard deviation of prediction vs target cross-spectrum across all patches
  - Computed as: `np.nanstd(pred_vs_target_cross_spectra_array, axis=0)`

#### Null Cross-Spectra (If Available)
- **`null_cross_spectra_array`** (numpy array, shape: `(n_subsample, n_ell_bins)` or `None`)
  - Full distribution of null cross-spectra for CMB reconstructions
  - For each patch: cross-spectrum between `unet_cmb_reconstructed` and random CMB patches
  - Used to assess statistical significance

- **`mean_null_cross_ps_overall`** (numpy array, shape: `(n_ell_bins,)` or `None`)
  - Overall mean null cross-spectrum
  - Computed as: `np.mean(null_cross_spectra_array, axis=0)`
  - Should be close to 0

#### Null Correlations (If Available)
- **`null_correlations_array`** (numpy array, shape: `(n_subsample,)` or `None`)
  - Full distribution of null correlations for CMB reconstructions
  - For each patch: correlation between `unet_cmb_reconstructed` and random CMB patches
  - Used to assess statistical significance

- **`mean_null_correlation_overall`** (float or `None`)
  - Overall mean null correlation for CMB
  - Should be close to 0

- **`fg_null_correlations_array`** (numpy array, shape: `(n_subsample,)` or `None`)
  - Full distribution of null correlations for foreground predictions
  - For each patch: correlation between `unet_prediction` and random target patches

- **`fg_mean_null_correlation_overall`** (float or `None`)
  - Overall mean null correlation for foregrounds
  - Should be close to 0

#### MSE Arrays for All Patches (Shape: `(n_subsample,)`)
**Note**: These are included even though they're for all patches (not just first sample) because they're needed for hexbin plots and distribution analysis.

- **`observed_b_mse_array`** (numpy array, shape: `(n_subsample,)`)
  - MSE values for all patches: MSE(pure_cmb, observed_b)
  - One value per patch

- **`unet_recon_mse_array`** (numpy array, shape: `(n_subsample,)`)
  - MSE values for all patches: MSE(pure_cmb, unet_cmb_reconstructed)
  - One value per patch

- **`pred_vs_target_mse_array`** (numpy array, shape: `(n_subsample,)`)
  - MSE values for all patches: MSE(prediction, target)
  - One value per patch
  - **This is the main MSE array for plotting**

- **`target_vs_zero_mse_array`** (numpy array, shape: `(n_subsample,)`)
  - MSE values for all patches: MSE(target, zero)
  - One value per patch
  - **This is the baseline MSE array for plotting**

**Usage Example**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load first sample data
with np.load('first_sample_data.npz') as data:
    # First sample patches
    cmb_patch = data['cmb_draw']  # Shape: (128, 128)
    pred_patch = data['prediction']  # Shape: (128, 128)
    
    # First sample metrics
    correlation = data['pred_vs_target_correlation']  # Scalar
    mse = data['pred_vs_target_mse']  # Scalar
    
    # Summary statistics for all patches
    mean_cross_ps = data['mean_pred_vs_target_cross_ps']  # Shape: (n_ell_bins,)
    std_cross_ps = data['std_pred_vs_target_cross_ps']  # Shape: (n_ell_bins,)
    ell_array = data['ell_array']  # Shape: (n_ell_bins,)
    
    # All patches MSE arrays (for plotting)
    all_mse_pred = data['pred_vs_target_mse_array']  # Shape: (n_subsample,)
    all_mse_baseline = data['target_vs_zero_mse_array']  # Shape: (n_subsample,)
    
    # Create scatter plot
    plt.scatter(all_mse_baseline, all_mse_pred, alpha=0.5)
    plt.xlabel('MSE(target, zero)')
    plt.ylabel('MSE(prediction, target)')
    plt.show()
```

---

## 2. Text Statistics Files (.txt)

### 2.1 `{model_identifier}_cross_spectrum_stats.txt`

**Purpose**: Detailed cross-power spectrum statistics for prediction vs target.

**Location**: 
- Full path example: `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/unnormalized_e_only_best_model_153600_19200_nonorm_cross_spectrum_stats.txt`
- Relative to output directory: `{output_dir}/{model_identifier}_cross_spectrum_stats.txt`

**Model Identifier Format**: `{norm_suffix}_{channel_suffix}_{model_basename}`
- Example: `unnormalized_e_only_best_model_153600_19200_nonorm`

**Contents**:

#### Header Information
- Model identifier
- Number of patches analyzed
- Number of ell bins

#### Ell Array
- Complete list of ell (multipole) values used for cross-spectrum analysis
- Format: Python list format `[ell1, ell2, ell3, ...]`
- Typically ranges from ~25 to 200 with DELTA_ELL=50 bin width

#### Mean Cross-Spectrum Statistics
- **Mean cross-spectrum (per ell bin, averaged across patches)**
  - Mean across ell: Overall mean value
  - Std across ell: Standard deviation across ell bins
  - Per-ell-bin statistics: `ell=XX: mean ± std` for each ell bin

#### Standard Deviation Cross-Spectrum Statistics
- **Std cross-spectrum (per ell bin, std across patches)**
  - Mean std (across ell): Average standard deviation across ell bins
  - Std of std (across ell): Standard deviation of the standard deviations

#### 2-Tailed Percentiles (16th and 84th, 1-sigma equivalent)
- **Mean lower (16th percentile, across ell)**: Average of 16th percentiles across ell bins
- **Mean upper (84th percentile, across ell)**: Average of 84th percentiles across ell bins
- **Per-ell-bin 2-tailed percentiles**: `ell=XX: [lower_16th, upper_84th]` for each ell bin
- These percentiles are used for error bands in plots (instead of mean ± std)

#### Overall Statistics
- **Mean cross-spectrum**: Mean across all patches and ell bins
- **Std cross-spectrum**: Standard deviation across all patches and ell bins

#### Plotting Data Format
- **Format**: `ell  mean_cross_ps  std_cross_ps  lower_16th  upper_84th`
- One line per ell bin
- Easy to parse for plotting scripts
- Includes both std and percentile values

**Example Content**:
```
Prediction vs Target Cross-Power Spectrum Statistics
Model: unnormalized_e_only_best_model_153600_19200_nonorm
Number of patches: 19200
Ell bins: 4

================================================================================

Ell (multipole) array:
  [25.0, 75.0, 125.0, 175.0]

================================================================================

Mean cross-spectrum (per ell bin, averaged across patches):
  Mean across ell: 1.234567e-03
  Std across ell: 5.678901e-04

Per-ell-bin statistics (mean ± std across patches):
  ell=  25.0: 1.100000e-03 ± 2.000000e-04
  ell=  75.0: 1.200000e-03 ± 2.100000e-04
  ...

2-tailed percentiles (16th and 84th, 1-sigma equivalent):
  Mean lower (16th percentile, across ell): 9.000000e-04
  Mean upper (84th percentile, across ell): 1.500000e-03

Per-ell-bin 2-tailed percentiles (16th-84th):
  ell=  25.0: [9.000000e-04, 1.300000e-03]
  ell=  75.0: [1.000000e-03, 1.400000e-03]
  ...

Data for plotting (ell, mean_cross_ps, std_cross_ps, lower_percentile, upper_percentile):
Format: ell  mean_cross_ps  std_cross_ps  lower_16th  upper_84th
================================================================================
  25.0  1.100000e-03  2.000000e-04  9.000000e-04  1.300000e-03
  75.0  1.200000e-03  2.100000e-04  1.000000e-03  1.400000e-03
  ...
```

---

### 2.2 `{model_identifier}_correlation_stats.txt`

**Purpose**: Spatial correlation statistics for prediction vs target.

**Location**: 
- Full path example: `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/unnormalized_e_only_best_model_153600_19200_nonorm_correlation_stats.txt`
- Relative to output directory: `{output_dir}/{model_identifier}_correlation_stats.txt`

**Contents**:

#### Header Information
- Model identifier
- Number of patches analyzed

#### Correlation Statistics
- **Mean correlation**: Average correlation across all patches
- **Std correlation**: Standard deviation of correlations
- **Min correlation**: Minimum correlation value
- **Max correlation**: Maximum correlation value
- **Median correlation**: Median correlation value

**Note**: Individual patch correlation values are **NOT** saved in this text file. To get per-patch correlations, load from `test_results.npz`:
```python
data = np.load('test_results.npz')
correlations = data['pred_vs_target_correlations']  # Shape: (n_subsample,)
```

**Example Content**:
```
Prediction vs Target Spatial Correlation Statistics
Model: unnormalized_e_only_best_model_153600_19200_nonorm
Number of patches: 19200

================================================================================

Mean correlation: 0.823456
Std correlation: 0.123456
Min correlation: 0.123456
Max correlation: 0.987654
Median correlation: 0.845678
```

---

### 2.3 `{model_identifier}_mse_stats.txt`

**Purpose**: Mean Squared Error (MSE) statistics for prediction vs target.

**Location**: 
- Full path example: `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/unnormalized_e_only_best_model_153600_19200_nonorm_mse_stats.txt`
- Relative to output directory: `{output_dir}/{model_identifier}_mse_stats.txt`

**Contents**:

#### Header Information
- Model identifier
- Number of patches analyzed

#### MSE(prediction, target) Statistics
- **Mean MSE**: Average MSE across all patches
- **Std MSE**: Standard deviation of MSE values
- **Min MSE**: Minimum MSE value
- **Max MSE**: Maximum MSE value
- **Median MSE**: Median MSE value

#### MSE(target, zero) Statistics (Baseline)
- **Mean MSE**: Average baseline MSE (if target_vs_zero_mse_array is available)
- **Std MSE**: Standard deviation of baseline MSE

#### Improvement Ratio
- **Improvement ratio (baseline / UNet)**: `mean_target_vs_zero_mse / mean_pred_vs_target_mse`
- If ratio > 1, UNet is better than predicting zero
- Example: If ratio = 2.5, UNet MSE is 2.5× better (lower) than baseline

**Note**: Individual patch MSE values are **NOT** saved in this text file. To get per-patch MSE values, load from `.npz` files:
```python
# From test_results.npz
data = np.load('test_results.npz')
pred_vs_target_mse = data['pred_vs_target_mse_array']  # Shape: (n_subsample,)
target_vs_zero_mse = data['target_vs_zero_mse_array']  # Shape: (n_subsample,)

# From first_sample_data.npz (also has all patches)
data = np.load('first_sample_data.npz')
pred_vs_target_mse = data['pred_vs_target_mse_array']  # Shape: (n_subsample,)
target_vs_zero_mse = data['target_vs_zero_mse_array']  # Shape: (n_subsample,)
```

**Example Content**:
```
Prediction vs Target MSE Statistics
Model: unnormalized_e_only_best_model_153600_19200_nonorm
Number of patches: 19200

================================================================================

MSE(prediction, target):
  Mean MSE: 1.234567e-06
  Std MSE: 2.345678e-07
  Min MSE: 5.678901e-07
  Max MSE: 3.456789e-06
  Median MSE: 1.123456e-06

MSE(target, zero) [baseline]:
  Mean MSE: 3.456789e-06
  Std MSE: 4.567890e-07

Improvement ratio (baseline / UNet): 2.8000x
UNet MSE is 2.80x better (lower) than baseline
```

---

## 3. Visualization Files (.png)

**Location**: Subdirectories within `{output_dir}/`
- Full path examples:
  - `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/cmb_reconstructions/`
  - `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/fg_reconstructions/`
  - `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only/input_channels/`

### 3.1 `cmb_reconstructions/` Directory

**Files**: `cmb_reconstruction_sample_{idx}.png`

**Contents**: 6-column visualization for each sample showing:
1. Pure primordial CMB B-modes patch
2. Observed B all-scales patch
3. UNet CMB reconstructed patch
4. Cross-power spectra plot (with error bands)
5. Null correlation histogram
6. MSE comparison plot

**Saved if**: `--no-viz` flag is **NOT** used

---

### 3.2 `fg_reconstructions/` Directory

**Files**: `fg_reconstruction_sample_{idx}.png`

**Contents**: 6-column visualization for each sample showing:
1. Foreground B-mode all-scales patch
2. Large-scale foreground B-modes (target)
3. UNet prediction patch (with correlation annotation)
4. Cross-power spectra plot (prediction vs target)
5. Null correlation histogram
6. MSE comparison plot

**Saved if**: `--no-viz` flag is **NOT** used

---

### 3.3 `input_channels/` Directory

**Files**: `input_channels_sample_{idx}.png`

**Contents**: Visualization of UNet input channels:
- **For bte mode**: 3 panels (B small-scale, T, E)
- **For te_only mode**: 2 panels (T, E)
- **For e_only mode**: 1 panel (E)
- **For t_only mode**: 1 panel (T)
- **For b_only mode**: 1 panel (B small-scale)

**Saved if**: `--no-viz` flag is **NOT** used and T/E channels are used

---

## Number of Test Patches

### Default Behavior (All Test Patches)

**By default**, the script saves **ALL test patches** from the test set:
- Default: `--n-subsample 0` (or not specified) → uses all test patches
- The number of test patches is determined by the train/valid/test split in `singlefreq_normalization_stats.npz`
- Typical test set size: **~19,200 patches** (out of ~153,600 total patches)
- This represents approximately **12.5%** of the total dataset

**Important**: There is **NO hardcoded limit of 500 patches**. The script saves all test patches by default unless you explicitly specify `--n-subsample N`.

### Subsampling (Optional)

If you want to use **fewer patches** for faster analysis:
```bash
python eval_test_results_singlefreq.py --n-subsample 1000
```
This will use only the first 1000 patches from the test set.

**Note**: If you previously ran the script with `--n-subsample 500`, then only 500 patches would be saved. But by default (without this flag), all test patches are saved.

### To Include All Test Patches

**To ensure you're using all test patches**, either:
1. **Don't specify `--n-subsample`** (default behavior uses all patches)
2. **Explicitly set `--n-subsample 0`** (converted to None, uses all patches)

**Example command using all test patches**:
```bash
python eval_test_results_singlefreq.py --e-only --model-path /path/to/model.pt
# or explicitly:
python eval_test_results_singlefreq.py --e-only --n-subsample 0 --model-path /path/to/model.pt
```

### How to Verify How Many Patches Were Saved

**To check how many patches are actually saved in your .npz file**:
```python
import numpy as np

# Load the saved file
with np.load('test_results.npz') as data:
    n_patches = len(data['unet_predictions'])
    test_indices = data['test_indices']
    print(f"Number of patches saved: {n_patches}")
    print(f"Test indices range: {test_indices.min()} to {test_indices.max()}")
    print(f"Number of unique test indices: {len(test_indices)}")
    
    # Compare with total test set size
    stats = np.load('singlefreq_normalization_stats.npz')
    total_test_size = len(stats['test_indices'])
    print(f"Total test set size: {total_test_size}")
    
    if n_patches == total_test_size:
        print("✓ All test patches are included!")
    else:
        print(f"⚠ Only {n_patches}/{total_test_size} patches are included.")
        print(f"  This means the script was run with --n-subsample {n_patches}")
```

**The script will also print this information when saving**:
- Look for: `"Number of test samples: {n_patches}"` in the output
- Look for: `"Subsample size: All test patches (no subsampling)"` if all patches are used
- Look for: `"Subsample size: {N} patches"` if subsampling was used

---

## File Summary Table

| File | Type | Contains | Shape/Format |
|------|------|----------|--------------|
| `test_results.npz` | Binary | All test patches data | Arrays: `(n_subsample, H, W)` or `(n_subsample,)` |
| `first_sample_data.npz` | Binary | First sample + summary stats | Mix of scalars, `(H, W)`, and `(n_subsample,)` |
| `{model_id}_cross_spectrum_stats.txt` | Text | Cross-spectrum statistics | Human-readable text with tables |
| `{model_id}_correlation_stats.txt` | Text | Correlation statistics | Human-readable text (summary only) |
| `{model_id}_mse_stats.txt` | Text | MSE statistics | Human-readable text (summary only) |
| `cmb_reconstructions/*.png` | Image | CMB reconstruction plots | PNG images (if `--no-viz` not used) |
| `fg_reconstructions/*.png` | Image | Foreground reconstruction plots | PNG images (if `--no-viz` not used) |
| `input_channels/*.png` | Image | Input channel visualizations | PNG images (if `--no-viz` not used) |

---

## Important Notes

1. **Alignment**: All arrays in `test_results.npz` are aligned using the same `test_indices`, ensuring proper correspondence between predictions, targets, correlations, etc.

2. **Subsampling**: If `--n-subsample N` is used, only the first N patches from the test set are saved. The `test_indices` array tells you which original dataset indices these correspond to.

3. **Individual Values in Text Files**: The `.txt` files contain **summary statistics only** (mean, std, min, max, median). To get **individual patch values** for plotting, you must load from the `.npz` files.

4. **Percentile Values**: The 16th and 84th percentile values are saved in the cross-spectrum stats text file, but **NOT** in the `.npz` files (as per your edits).

5. **MSE Values for Plotting**: To create scatter plots of MSE values, load from `.npz` files:
   ```python
   data = np.load('test_results.npz')
   pred_mse = data['pred_vs_target_mse_array']  # All patches
   baseline_mse = data['target_vs_zero_mse_array']  # All patches
   ```

6. Table values: 16/84th percentile values for normalized cross correlation coefficient or harmonic correlation coefficient are listed in: /scratch/gpfs/JDUNKLEY/hshao/ILC_ML/cross_spectrum_percentile_averages.txt