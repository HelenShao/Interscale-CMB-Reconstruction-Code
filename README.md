# Code release: signal-preserving CMB foreground modeling (ICML 2026)

This folder contains the scripts used to train U-Nets, evaluate test patches, and reproduce figures. **This repo does not contain subdirs `DATA/`, `docs/`, and `checkpoints/` due to their large sizes.** You can obtain **data** and **pretrained `.pt` weights** from: [ICML_AI4PHYSICS data and models](https://drive.google.com/drive/folders/1fGdGXZSukeqtr_QxOT10rJRp5Df2Dlrf?usp=drive_link). Download them into `code_release/checkpoints/`, preserving the `singlefreq/` and `hybrid/` layout used in the `--model-path` examples below. Locally, set `ILC_ML_DATA_ROOT` to a tree matching `paths_config.py`, or keep a `DATA/` directory next to these scripts. For dependencies, see `ENVIRONMENT.md`.

## Layout

| File | Role |
|------|------|
| `paths_config.py` | Central data/output paths (override with `ILC_ML_*` env vars). |
| `architecture.py` | UNet architecture |
| `apply_ilc_core.py` | Patch-level ILC reference implementation (not used by training/eval drivers). |
| `cmb_spectrum_2d.py` | `calculate_2d_spectrum` for harmonic correlations in eval (NumPy). |
| `plot_params.py` | Matplotlib style defaults|
| `train_unet_singlefreq.py` | Single-frequency inter-scale training. |
| `train_unet_interscale.py` | Hybrid multi-frequency + inter-scale training. |
| `eval_test_results_singlefreq.py` | Single-frequency test metrics + plots. |
| `eval_test_results.py` | Hybrid model test metrics + plots. |
| `reproduce_visualizations.py` | Figures from saved `first_sample_data.npz`. |
| `visualize_multifreq_results.py` | Quick plots from `test_results.npz`. |
| `checkpoints/` | **Not in git:** download `.pt` files from the [Drive link above](https://drive.google.com/drive/folders/1fGdGXZSukeqtr_QxOT10rJRp5Df2Dlrf?usp=drive_link) into this path. |

Run scripts from **this directory** (or ensure it is on `PYTHONPATH`) so local imports resolve.

## Installation

See `ENVIRONMENT.md` for dependencies.

## Training and evaluation steps

1. Obtain DustFilaments foreground patches, CMB draws, **`ILC_products/`**, and normalization `.npz` files
   ## Data downloads

| File | Size (approx.) | Download |
|------|----------------|----------|
| Evaluation Data (`teb_nonorm.npz`, `full_16ch_nonorm.npz`) | ~55 GB | `ICML2026_AI4PHYSICS_eval_exports.zip` |
| Checkpoints (`singlefreq/`, `hybrid/`) | ~2.8 GB | `ICML2026_AI4PHYSICS_checkpoints.zip` |

**ZIP Files**
- `ICML2026_AI4PHYSICS_eval_exports.zip`: `5c5fe2f9ef00210b99dcedb41f59834c82293de0fbc5b2bdfff5bdff3e8c1708`
- `ICML2026_AI4PHYSICS_checkpoints.zip`: `149bc39201289471e0e91ffa65a2b097e8f445bd178a179edd87f2fbe4950d9a`

### Setup

```bash
unzip ICML2026_AI4PHYSICS_eval_exports.zip -d eval_exports
unzip ICML2026_AI4PHYSICS_checkpoints.zip

python eval_test_results_singlefreq.py --no-viz \
  --test-data-npz eval_exports/singlefreq/teb_nonorm.npz \
  --model-path checkpoints/singlefreq/te_plus_e_plus_bs_nonorm.pt

python eval_test_results.py --no-viz --add-e-t \
  --test-data-npz eval_exports/hybrid/full_16ch_nonorm.npz \
  --model-path checkpoints/hybrid/ilc_fg_plus_bs_plus_te_fourfreq_nonorm.pt
3. **Train** (examples):  
   `python train_unet_singlefreq.py --no-normalize`  
   `python train_unet_interscale.py --no-normalize`  
   `python train_unet_interscale.py --add-e-t --no-normalize`
4. **Evaluate**:
   **From exported `.npz`** : pass `--test-data-npz` with the path to the evaluation bundle, the same channel options as the checkpoint, and `--model-path`:
   ```bash
   python eval_test_results_singlefreq.py --no-viz --test-data-npz path/to/teb_nonorm.npz --model-path checkpoints/singlefreq/te_plus_e_plus_bs_nonorm.pt
   python eval_test_results.py --no-viz --add-e-t --test-data-npz path/to/full_16ch_nonorm.npz --model-path checkpoints/hybrid/…_et_….pt
   ```
   (Use `--help` on each script for channel flags; normalized exports require `--normalize` to match the saved data.)

## Running the code

- **Training:** `train_unet_*.py`scripts  (channel flags: `--b-only`, `--ilc-only`, `--add-e-t`, `--no-normalize`, etc.).
- **Evaluation:** `eval_test_results*.py` flags and `--model-path` to the saved `best_model_*.pt`.

**Pretrained checkpoints:** ~2.8 GB total (`singlefreq/` + `hybrid/`). Not stored in this repository—download from [this shared folder](https://drive.google.com/drive/folders/1fGdGXZSukeqtr_QxOT10rJRp5Df2Dlrf?usp=drive_link) and unpack under `checkpoints/` (see instructions above).

## Reproducing plots

After evaluation writes `first_sample_data.npz`:

```bash
python reproduce_visualizations.py --data-path /path/to/first_sample_data.npz --output-dir /path/to/plots
```
