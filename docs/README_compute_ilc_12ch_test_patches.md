# `compute_ilc_12ch_test_patches.py`

Compute Internal Linear Combination (ILC) on the UNet **test patch split** and save reconstruction products for comparison with paper baselines and ML models.

Each variant uses a different set of foreground input channels (frequency × polarization). All variants share the same test indices, CMB draw, and ILC core (`apply_ilc_to_single_patch` from `apply_ilc_to_patches.py`).

---

## What it does

1. Loads **test_indices** from `test_results.npz` (default: n_test = 19,200 patches).
2. Loads **B-mode CMB** from `sim1-150_freq220_cmb_draw_b_all_scales.npy`, scaled by `--cmb-scale` (default `0.05`).
3. Loads **foreground patches** per channel from `new_DF_patches_B` (B modes) and `new_DF_patches_T_E` (T/E modes).
4. For each patch, runs ILC with constraint vector `a = ones(n_channels)` (weights sum to 1).
5. Saves ILC CMB reconstruction, residuals, weights, diagnostics, and optional figures.

**Physics note:** T and E channels are foreground-only inputs. The same B-mode CMB patch is added to **every** input channel inside `apply_ilc_to_single_patch`. ILC therefore preserves the CMB algebraically; worse MSE vs. fewer-channel runs reflects **foreground leakage** (`Σ wᵢ fᵢ`), not lost CMB signal.

---

## Input data (defaults)

| Role | Path |
|------|------|
| Test split | `/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/test_results.npz` |
| Split validation | `/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_B/singlefreq_normalization_stats.npz` |
| B foregrounds | `/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_B/` |
| T/E foregrounds | `/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_T_E/` |
| CMB draw | `{b-dir}/sim1-150_freq220_cmb_draw_b_all_scales.npy` |

Foreground file naming: `sim1-150_freq{FREQ}_{B|T|E}_patches.npy`.

Frequencies: **95, 145, 220, 270 GHz**.

---

## Channel variants

Exactly **one** mode flag per run (mutually exclusive). If none is set, the default is **12-channel B+T+E @ 4 frequencies**.

### Multi-frequency (4 bands)

| Flag | Channels | Channel order | Default output directory |
|------|----------|---------------|--------------------------|
| *(default)* | 12 | B@4freq, T@4freq, E@4freq | `/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_12ch_teb_test` |
| `--input-b-only` | 4 | B_95, B_145, B_220, B_270 | `.../ILC_products_4ch_b_test` |
| `--input-bt-only` | 8 | B@4freq, T@4freq | `.../ILC_products_8ch_bt_test` |
| `--input-be-only` | 8 | B@4freq, E@4freq | `.../ILC_products_8ch_be_test` |

12-channel order: `[B_95, B_145, B_220, B_270, T_95, …, T_270, E_95, …, E_270]`.

### Single-frequency (one band, default 220 GHz)

Use `--single-freq {95|145|220|270}` with one of:

| Flag | Channels | Example @ 220 GHz | Default output directory |
|------|----------|-------------------|--------------------------|
| `--single-freq-b-only` | 1 | B_220 | `.../ILC_products_1ch_b_220_test` |
| `--single-freq-bt-only` | 2 | B_220, T_220 | `.../ILC_products_2ch_bt_220_test` |
| `--single-freq-be-only` | 2 | B_220, E_220 | `.../ILC_products_2ch_be_220_test` |
| `--single-freq-bte-only` | 3 | B_220, T_220, E_220 | `.../ILC_products_3ch_bte_220_test` |

Output dir names include the frequency, e.g. `ILC_products_3ch_bte_145_test` for `--single-freq 145`.

All default output roots live under:

```
/scratch/gpfs/JDUNKLEY/hshao/old_data/
```

Override with `--output-dir`.

---

## Output files (per variant)

For a run with `file_tag` (e.g. `12ch`, `4ch_b`, `3ch_bte_220`), outputs go to that variant’s output directory:

| File | Description |
|------|-------------|
| `ILC_cmb_{tag}_test.npy` | ILC CMB reconstruction, shape `(n_test, 128, 128)` |
| `ILC_residuals_{tag}_test.npy` | `ILC_cmb − true_cmb_draw` |
| `ILC_fg_b_{tag}_test.npy` | B-channel foreground leakage (`w·f` on B inputs) |
| `ILC_fg_all_{tag}_test.npy` | Foreground leakage on all input channels |
| `ILC_weights_{tag}_test.npy` | Per-patch ILC weights, shape `(n_test, n_channels)` |
| `ILC_cmb_cross_spectrum_{tag}_test.npy` | Harmonic cross-spectra (unless `--skip-harmonic`) |
| `ILC_{tag}_test_metadata.npz` | Run config: indices, channels, cmb_scale, paths |
| `ILC_{tag}_test_summary.txt` | MSE, spatial corr, weight sums, variance, harmonic stats |
| `ILC_{tag}_test_weights_report.txt` | Global + sample per-channel weights |
| `ILC_{tag}_test_variance_stats.txt` | Per-patch spatial variance table |
| `ILC_{tag}_test_cross_spectrum_stats.txt` | Harmonic correlation summary |
| `ILC_{tag}_test_harmonic_summary.npz` | Ell-binned ρ(ℓ) aggregates |
| `figures/sample_*_dataset_*.png` | Sample patch visual comparisons |
| `figures/summary_diagnostics.png` | MSE / corr / weight-sum histograms |

Slurm runs also write logs via `tee`, e.g. `ilc_12ch_test.log` inside the output directory.

---

## Post-hoc diagnostics (no ILC recompute)

Point `--output-dir` at an existing run directory:

| Flag | Action |
|------|--------|
| `--figures-only` | Regenerate PNG figures from saved `.npy` |
| `--weights-only` | Print/save weight diagnostics |
| `--variance-only` | Recompute spatial variance stats |
| `--harmonic-only` | Recompute harmonic correlation |
| `--compare-variance DIR [DIR ...]` | Cross-run variance comparison table |

Example:

```bash
python compute_ilc_12ch_test_patches.py --compare-variance \
  /scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_4ch_b_test \
  /scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_12ch_teb_test \
  /scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_3ch_bte_220_test
```

---

## How to run

**Requires a compute node** with `matplotlib` (login node typically lacks it). Request ~80G RAM for 12ch / full test set.

### Slurm batch scripts (same directory)

| Script | Variant(s) |
|--------|------------|
| `run_ilc_12ch_test.sbatch` | 12ch B+T+E (default) |
| `run_ilc_4ch_b_test.sbatch` | 4ch B-only |
| `run_ilc_8ch_bt_test.sbatch` | 8ch B+T |
| `run_ilc_8ch_be_test.sbatch` | 8ch B+E |
| `run_ilc_3ch_bte_220_test.sbatch` | 3ch B+T+E @ 220 GHz |
| `run_ilc_220ghz_all_test.sbatch` | All four single-freq 220 GHz variants |

```bash
cd /scratch/gpfs/JDUNKLEY/hshao/ILC_ML/toy_model/toy_model/DustFilaments/new_patches/new_patches
sbatch run_ilc_12ch_test.sbatch
```

### Interactive example

```bash
python compute_ilc_12ch_test_patches.py \
  --single-freq-bte-only \
  --single-freq 220 \
  --cmb-scale 1.0 \
  --output-dir /scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_3ch_bte_220_test
```

Pipe to a log file if you want a saved transcript:

```bash
python compute_ilc_12ch_test_patches.py --input-b-only --cmb-scale 1.0 \
  2>&1 | tee /scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_4ch_b_test/run.log
```

---

## Key options

| Option | Default | Notes |
|--------|---------|-------|
| `--cmb-scale` | `0.05` | Matches UNet training scale; use `1.0` for unscaled paper-style comparison |
| `--test-results-npz` | see Input data | Defines which patches are “test” |
| `--b-dir`, `--te-dir` | see Input data | Foreground patch roots |
| `--skip-harmonic` | off | Skip cross-spectrum computation (faster) |
| `--ell-max` | `200` | Max ℓ for harmonic correlation |
| `--sample-indices` | first 5 | Array indices for PNG samples |

---

## Relation to paper / eval pipeline

Paper table values in `multifreq-interscale-paper/main.tex` come from the separate **`eval_test_results.py`** pipeline under paths like:

```
/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_interscale_unet_1024/
```

This script produces **parallel ILC baseline products** under `ILC_products_*_test` for direct comparison of channel-count and polarization choices (4 vs 8 vs 12 vs single-freq @ 220).

When comparing to `test_results.npz` fields such as `observed_b_all_scales`, ensure `--cmb-scale` matches how that npz was built (mismatch produces expected validation warnings, not necessarily a bug).

---

## Memory

Approximate peak RAM for n_test = 19,200:

- **12ch:** ~37 GiB  
- **8ch / 4ch / single-freq:** proportionally less  

Use `--mem=80G` in Slurm for headroom.
