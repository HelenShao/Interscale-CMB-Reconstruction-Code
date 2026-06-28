# Paper reproduction guide

Reproduce results in [`multifreq-interscale-paper`](https://github.com/HelenShao/multifreq-interscale-paper) (LaTeX + figures) using this code release.

**Related docs in `docs/`:**
- `PAPER_OUTLINE.md` — section-by-section results map and extracted numbers
- `EVAL_OUTPUT_FILES_DOCUMENTATION.md` — `eval_test_results_singlefreq.py` outputs
- `README_compute_ilc_12ch_test_patches.md` — ILC baseline on the UNet test split
- `ILC_ENHANCEMENT_README.md` — enhanced ILC (Appendix)

## Data layout

Set `ILC_ML_DATA_ROOT` to a directory containing:

```
$ILC_ML_DATA_ROOT/
├── new_DF_patches_B/          # B-mode patches + CMB draw + singlefreq_normalization_stats.npz
├── new_DF_patches_T_E/        # T/E patches @ 95,145,220,270 GHz
├── ILC_products/              # from compute_ilc_products.py
├── singlefreq_unet_1024_0.05_scaled/   # single-freq eval outputs
└── ILC_interscale_unet_1024/           # hybrid eval outputs
```

**Pretrained weights** (~2.8 GB): [Google Drive checkpoints](https://drive.google.com/drive/folders/1fGdGXZSukeqtr_QxOT10rJRp5Df2Dlrf?usp=drive_link) → `checkpoints/`.

**Test-only bundle:** run `build_eval_test_data.py` to export a small `.npz` for evaluation without full patch arrays.

## Reproduction order

### Step 0 — Environment

See `ENVIRONMENT.md`. Run all scripts from **this directory**:

```bash
export ILC_ML_DATA_ROOT=/path/to/your/data
cd Interscale-CMB-Reconstruction-Code
```

### Step 1 — ILC training products (multi-freq hybrid inputs)

```bash
python compute_ilc_products.py
python generate_interscale_normalization_stats.py          # 8ch stats
python generate_interscale_normalization_stats.py --include-e-t   # 16ch stats
```

### Step 2 — Single-frequency stats

```bash
python generate_singlefreq_normalization_stats.py
```

### Step 3 — Train U-Nets (or use pretrained checkpoints)

| Paper section | Script | Key flags |
|---------------|--------|-----------|
| 7.1 B-only | `train_unet_singlefreq.py` | `--b-only --no-normalize` |
| 7.1 T+E+B | `train_unet_singlefreq.py` | `--no-normalize` (default T+E+B) |
| 7.1 T-only, E-only, TE | `train_unet_singlefreq.py` | `--t-only`, `--e-only`, `--te-only` |
| 7.2 ILC-only (4ch) | `train_unet_interscale.py` | `--ilc-only --no-normalize` |
| 7.2 ILC+B_S (8ch) | `train_unet_interscale.py` | `--no-normalize` |
| 7.2 ILC+B_S+T+E (16ch) | `train_unet_interscale.py` | `--add-e-t --no-normalize` |

### Step 4 — Evaluate U-Nets (paper tables)

**Single-frequency** (Section 7.1):

```bash
python eval_test_results_singlefreq.py --no-viz \
  --model-path checkpoints/singlefreq/... \
  --output-dir $ILC_ML_DATA_ROOT/singlefreq_unet_1024_0.05_scaled/unnormalized_teb
# Channel variants: --b-only, --t-only, --e-only, --te-only
```

**Multi-frequency hybrid** (Section 7.2, Table 2):

```bash
python eval_test_results.py --no-viz --ilc-only --model-path checkpoints/hybrid/...
python eval_test_results.py --no-viz --model-path checkpoints/hybrid/...          # 8ch
python eval_test_results.py --no-viz --add-e-t --model-path checkpoints/hybrid/... # 16ch
python eval_test_results.py --compare-all   # side-by-side comparison figures
```

Outputs: `test_results.npz`, `*_correlation_stats.txt`, `*_cross_spectrum_stats.txt`, `*_mse_stats.txt`.  
See `docs/EVAL_OUTPUT_FILES_DOCUMENTATION.md`.

### Step 5 — ILC baselines on test split

Parallel ILC runs (same `test_indices` as UNet eval):

```bash
python compute_ilc_12ch_test_patches.py                    # default 12ch
python compute_ilc_12ch_test_patches.py --input-b-only     # 4ch B
python compute_ilc_12ch_test_patches.py --single-freq-bte-only --single-freq 220
# Or: sbatch slurm/run_ilc_12ch_test.sbatch
```

See `docs/README_compute_ilc_12ch_test_patches.md`.

### Step 6 — Enhanced ILC (Appendix A.1)

```bash
python eval_ilc_enhancement.py \
  --model-path checkpoints/singlefreq/... \
  --output-dir $ILC_ML_DATA_ROOT/.../ilc_enhancement_results \
  --cmb-scale 0.01
```

See `docs/ILC_ENHANCEMENT_README.md`.

### Step 7 — Figures

| Figure / analysis | Script |
|-------------------|--------|
| Sample reconstructions | `reproduce_visualizations.py` |
| Multi-freq quick plots | `visualize_multifreq_results.py` |
| Single-freq cross-spectrum comparison | `compare_models.py`, `reproduce_v7_plot.py` |
| Signal preservation validation | `plot_signal_preservation_validation.py` |
| Multi-config comparison | `eval_test_results.py --compare-all` |

## Paper ↔ output mapping (quick reference)

| Paper result | Primary output file |
|--------------|---------------------|
| Table 2 (multi-freq) | `ILC_interscale_unet_1024/unnormalized_*/test_results.npz` + `*_stats.txt` |
| Section 7.1 B-only ρ = 0.46 | `singlefreq_.../normalized_b_only/*_correlation_stats.txt` |
| Section 7.1 T+E+B ρ = 0.76 | `singlefreq_.../unnormalized_teb/*_correlation_stats.txt` |
| ILC baseline MSE | `ILC_products_*_test/ILC_*_test_summary.txt` |

Full paths and extracted numbers: `docs/PAPER_OUTLINE.md`.

## Slurm

Batch scripts in `slurm/` use `ILC_ML_DATA_ROOT` and paths relative to this repo. Example:

```bash
export ILC_ML_DATA_ROOT=/scratch/gpfs/JDUNKLEY/hshao/old_data
sbatch slurm/run_ilc_12ch_test.sbatch
```

**Tip:** pipe long runs through `tee run.log` if you want a saved transcript.

## LaTeX paper

Source and committed figures: [multifreq-interscale-paper](https://github.com/HelenShao/multifreq-interscale-paper).
