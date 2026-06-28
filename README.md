# Code release: signal-preserving CMB foreground modeling (ICML 2026)

Signal-preserving U-Net training and evaluation for single-frequency and multi-frequency CMB foreground removal (inter-scale ML + ILC hybrid). Companion to the paper in [`multifreq-interscale-paper`](https://github.com/HelenShao/multifreq-interscale-paper).

**→ Start here for paper results: [`REPRODUCTION.md`](REPRODUCTION.md)**

This repo does not contain `DATA/`, `checkpoints/`, or large `.npy` arrays. Set `ILC_ML_DATA_ROOT` to match `paths_config.py`, or use a local `DATA/` directory next to these scripts. For dependencies, see [`ENVIRONMENT.md`](ENVIRONMENT.md).

## What's in this repo

| Category | Files |
|----------|-------|
| **Core** | `architecture.py`, `paths_config.py`, `plot_params.py`, `cmb_spectrum_2d.py`, `apply_ilc_core.py` |
| **Data prep** | `compute_ilc_products.py`, `generate_singlefreq_normalization_stats.py`, `generate_interscale_normalization_stats.py`, `build_eval_test_data.py` |
| **Training** | `train_unet_singlefreq.py`, `train_unet_interscale.py` |
| **Evaluation** | `eval_test_results_singlefreq.py`, `eval_test_results.py`, `eval_ilc_enhancement.py` |
| **ILC baselines** | `compute_ilc_12ch_test_patches.py` + `slurm/run_ilc_*.sbatch` |
| **Figures** | `reproduce_visualizations.py`, `compare_models.py`, `plot_signal_preservation_validation.py`, … |
| **Docs** | `docs/PAPER_OUTLINE.md`, `docs/EVAL_OUTPUT_FILES_DOCUMENTATION.md`, … |

Run scripts from **this directory** (or ensure it is on `PYTHONPATH`) so local imports resolve.

## Data downloads

Obtain **data** and **pretrained `.pt` weights** from [ICML_AI4PHYSICS data and models](https://drive.google.com/drive/folders/1fGdGXZSukeqtr_QxOT10rJRp5Df2Dlrf?usp=drive_link).

| File | Size (approx.) | Download |
|------|----------------|----------|
| Evaluation Data (`teb_nonorm.npz`, `full_16ch_nonorm.npz`) | ~55 GB | `ICML2026_AI4PHYSICS_eval_exports.zip` |
| Checkpoints (`singlefreq/`, `hybrid/`) | ~2.8 GB | `ICML2026_AI4PHYSICS_checkpoints.zip` |

**ZIP checksums (SHA256)**
- `ICML2026_AI4PHYSICS_eval_exports.zip`: `5c5fe2f9ef00210b99dcedb41f59834c82293de0fbc5b2bdfff5bdff3e8c1708`
- `ICML2026_AI4PHYSICS_checkpoints.zip`: `149bc39201289471e0e91ffa65a2b097e8f445bd178a179edd87f2fbe4950d9a`

### Quick eval (from downloaded ZIPs)

```bash
unzip ICML2026_AI4PHYSICS_eval_exports.zip -d eval_exports
unzip ICML2026_AI4PHYSICS_checkpoints.zip

python eval_test_results_singlefreq.py --no-viz \
  --test-data-npz eval_exports/singlefreq/teb_nonorm.npz \
  --model-path checkpoints/singlefreq/te_plus_e_plus_bs_nonorm.pt

python eval_test_results.py --no-viz --add-e-t \
  --test-data-npz eval_exports/hybrid/full_16ch_nonorm.npz \
  --model-path checkpoints/hybrid/ilc_fg_plus_bs_plus_te_fourfreq_nonorm.pt
```

## Data paths

All paths are centralized in `paths_config.py`. Override with environment variables:

| Variable | Purpose |
|----------|---------|
| `ILC_ML_DATA_ROOT` | Root data directory |
| `ILC_ML_OUTPUT_SINGLEFREQ` | Single-freq eval output root |
| `ILC_ML_OUTPUT_INTERSCALE` | Hybrid eval output root |
| `ILC_ML_ILC_TEST_ROOT` | ILC baseline products (`ILC_products_*_test/`) |

## Training and evaluation

- **Training:** `train_unet_*.py` (flags: `--b-only`, `--ilc-only`, `--add-e-t`, `--no-normalize`, etc.)
- **Evaluation:** `eval_test_results*.py` with `--model-path` to a saved `best_model_*.pt` or `--test-data-npz` for exported bundles

After evaluation writes `first_sample_data.npz`:

```bash
python reproduce_visualizations.py --data-path /path/to/first_sample_data.npz --output-dir /path/to/plots
```

## Installation

See [`ENVIRONMENT.md`](ENVIRONMENT.md).

## License

See repository license (if applicable). DustFilaments simulations: cite Hervías-Caimapo et al. 2022.
