# UNet-Enhanced ILC Evaluation

## Overview

This evaluation compares **vanilla Internal Linear Combination (ILC)** with **UNet-enhanced ILC** for CMB B-mode reconstruction from multi-frequency observations.

## Analysis of `apply_ilc_to_single_patch_with_unet` Function

### Mathematical Correctness ✓

The function implements ILC with an additional synthetic channel. The implementation is **mathematically correct** with the following key points:

1. **Covariance Matrix**: Correctly computed across n_freq + 1 channels
   ```python
   covariance = np.cov(observed_flat)  # Shape: (n_freq+1, n_freq+1)
   ```

2. **ILC Weights Formula**: Properly implements `w = (a^T C^{-1} a)^{-1} · a^T C^{-1}`
   - Minimizes variance while preserving CMB signal
   - Constraint: `a^T w = 1` ensures unit response to CMB

3. **Constraint Vector**: `a_matrix = np.ones(n_freq + 1)`
   - Assumes CMB has same amplitude in all channels (including synthetic UNet channel)
   - **CRITICAL**: This is valid ONLY if the synthetic channel contains the **same CMB signal**

4. **Regularization**: Properly scaled by (n_freq + 1) to avoid singular matrices

### Critical Requirements

For the function to work correctly:

1. **Same CMB Signal**: The synthetic UNet channel MUST contain the exact same CMB realization as the multi-frequency observations
   ```python
   # Correct construction:
   synthetic_channel = scaled_primary_cmb + b_small + unet_predicted_b_large
   
   # where scaled_primary_cmb is the SAME CMB added to all frequency channels
   ```

2. **Synthetic Channel Composition**:
   - Scaled primary CMB (same across all channels)
   - Small-scale B-mode foregrounds (ℓ > cutoff)
   - UNet predicted large-scale B-mode foregrounds (ℓ < cutoff)

3. **Parameter Naming Issue**: The parameter `unet_cmb_prediction` is misleading
   - Should be `unet_synthetic_observation` or `unet_channel`
   - It's not just the CMB prediction, but a full synthetic observation

## Implementation: `eval_ilc_enhancement.py`

### Data Flow

```
1. Load multi-frequency B-mode patches (95, 150, 220, 270 GHz)
   ├─> Foreground patches at each frequency
   
2. Load primary CMB (same for all frequencies)
   ├─> Scale by factor (default: 0.001)
   
3. Create observed signals
   ├─> observed_freq[i] = foreground_freq[i] + scaled_cmb
   
4. Load UNet components
   ├─> UNet predictions (large-scale B-mode foregrounds)
   ├─> Small-scale foregrounds (UNet input)
   
5. Construct synthetic UNet channel
   ├─> synthetic = scaled_cmb + b_small + unet_predicted_b_large
   
6. Apply ILC methods
   ├─> Vanilla ILC: 4 frequency channels
   ├─> Enhanced ILC: 4 frequency channels + UNet synthetic channel
   
7. Compare results
   ├─> MSE, Spatial Correlation, ILC Weights
```

### Key Functions

1. **`load_multifreq_data()`**: Loads B-mode patches for multiple frequencies
2. **`load_primary_cmb()`**: Loads primary CMB (frequency-independent)
3. **`load_unet_components()`**: Loads UNet predictions and small-scale foregrounds
4. **`load_model_and_generate_predictions()`**: Runs UNet model to generate predictions
5. **`apply_vanilla_ilc()`**: Applies vanilla ILC (4 frequencies only)
6. **`apply_enhanced_ilc()`**: Applies enhanced ILC (4 frequencies + UNet channel)
7. **`analyze_results()`**: Computes metrics (MSE, correlation, improvement ratio)
8. **`visualize_comparison()`**: Creates comparison plots

### Metrics

1. **Mean Squared Error (MSE)**:
   ```
   MSE = mean((ILC_reconstruction - primary_CMB)^2)
   ```

2. **Spatial Correlation**:
   ```
   Correlation = corrcoef(ILC_reconstruction.flatten(), primary_CMB.flatten())
   ```

3. **Improvement Ratio**:
   ```
   Ratio = vanilla_MSE / enhanced_MSE
   ```
   - Ratio > 1: Enhanced ILC is better
   - Ratio < 1: Vanilla ILC is better

4. **ILC Weights Analysis**:
   - How much weight does ILC assign to each frequency channel?
   - How much weight to the UNet synthetic channel?
   - High UNet weight → ILC trusts the UNet reconstruction

## Usage

### Basic Usage

```bash
python eval_ilc_enhancement.py \
    --model-path /path/to/model.pt \
    --output-dir /path/to/output
```

### With Options

```bash
python eval_ilc_enhancement.py \
    --model-path /scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/best_model_153600_19200_nonorm.pt \
    --output-dir /scratch/gpfs/JDUNKLEY/hshao/old_data/ilc_enhancement_results \
    --cmb-scale 0.001 \
    --n-test 500 \
    --sample-indices 0,10,20,30,40
```

### Arguments

- `--model-path`: Path to trained UNet model checkpoint (required)
- `--output-dir`: Output directory for results (required)
- `--cmb-scale`: Scaling factor for primary CMB (default: 0.001)
- `--normalize`: Use normalized model (flag)
- `--b-only`: Use B-only model (1 channel) instead of B+T+E (flag)
- `--n-test`: Number of test samples to process (default: all)
- `--sample-indices`: Comma-separated list of samples to visualize (default: first 5)

## Expected Results

### If UNet-Enhanced ILC Works Well:

1. **Lower MSE**: Enhanced ILC should have lower MSE than vanilla ILC
2. **Higher Correlation**: Enhanced ILC should have higher correlation with primary CMB
3. **Positive UNet Weight**: ILC assigns significant positive weight to UNet channel
4. **Most Patches Improved**: Majority of patches show improvement ratio > 1

### If UNet-Enhanced ILC Doesn't Help:

1. **Similar or Worse MSE**: No improvement or degradation
2. **Low/Negative UNet Weight**: ILC doesn't trust the UNet channel
3. **Few Patches Improved**: Improvement ratio ≈ 1 or < 1

### Potential Issues to Watch For:

1. **Overfitting**: UNet channel might be over-weighted if UNet predictions are too similar to training data
2. **Correlation Artifacts**: If UNet errors are correlated with CMB, could introduce bias
3. **Scale Mismatch**: CMB scaling factor might affect results (try different values: 0.0001, 0.001, 0.01)
4. **Weight Redistribution**: Enhanced ILC might just redistribute weights without improving reconstruction

## Output Files

1. **`ilc_comparison_results.npz`**: All numerical results
   - Vanilla/Enhanced ILC reconstructions
   - MSE, correlation, improvement ratios
   - ILC weights
   - Test indices

2. **`sample_*_comparison.png`**: Visual comparison for each sample
   - Primary CMB
   - Vanilla ILC reconstruction
   - Enhanced ILC reconstruction
   - Difference map

3. **`mse_comparison_scatter.png`**: MSE scatter plot
   - Hexbin plot of vanilla MSE vs enhanced MSE
   - Points below diagonal = improvement

4. **`correlation_comparison.png`**: Correlation comparison
   - Histograms of correlations
   - Scatter plot

5. **`ilc_weights_comparison.png`**: ILC weights boxplots
   - Vanilla ILC weights (4 frequencies)
   - Enhanced ILC weights (4 frequencies + UNet)

## Interpretation Guide

### Good Results:
- Mean improvement: > 10%
- Mean enhanced correlation: > 0.95
- UNet weight: 0.1 - 0.5 (balanced, not dominating)
- Patches improved: > 70%

### Moderate Results:
- Mean improvement: 1-10%
- Mean enhanced correlation: 0.90 - 0.95
- UNet weight: 0.05 - 0.1 or 0.5 - 0.8
- Patches improved: 50-70%

### Poor Results:
- Mean improvement: < 1% or negative
- Mean enhanced correlation: < 0.90
- UNet weight: < 0.05 (not trusted) or > 0.8 (over-reliance)
- Patches improved: < 50%

## Theoretical Considerations

### Why Enhanced ILC Might Work:

1. **Spatial Information**: UNet captures spatial foreground patterns that frequency-only ILC misses
2. **Learned Priors**: UNet learns foreground statistics from training data
3. **Non-linear Separation**: UNet provides non-linear component separation, ILC combines it linearly
4. **Complementary Information**: UNet + frequency diversity = better separation

### Why Enhanced ILC Might Not Work:

1. **UNet Errors**: If UNet errors are correlated with CMB, introduces bias
2. **Overfitting**: UNet might overfit to training data, doesn't generalize
3. **Redundancy**: UNet channel might be redundant with frequency information
4. **Noise Amplification**: Additional channel might amplify noise if UNet predictions are noisy

## Next Steps

1. **Run Evaluation**: Execute the script with your trained model
2. **Analyze Results**: Check MSE, correlation, and weight distributions
3. **Vary CMB Scale**: Try different scaling factors (0.0001, 0.001, 0.01, 0.1)
4. **Compare Models**: Try different UNet architectures or training configurations
5. **Cross-Spectrum Analysis**: Compute cross-power spectra between ILC reconstructions and primary CMB
6. **Statistical Significance**: Perform hypothesis tests to verify improvements are significant

## References

- ILC Method: Tegmark et al. (2003), "A high resolution foreground cleaned CMB map from WMAP"
- Multi-frequency Observations: Planck Collaboration (2018)
- Neural Networks for CMB: Caldeira et al. (2019), "DeepCMB: Lensing reconstruction of the CMB with deep neural networks"

