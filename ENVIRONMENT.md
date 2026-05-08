# Python environment

Training, evaluation, and plotting scripts rely on the packages below.

## Required

| Package | Role in this repo |
|--------|-------------------|
| **Python** | 3.9+ recommended (3.10+ typical on clusters). |
| **NumPy** | Arrays, `.npz` stats, ILC / patch I/O in almost every script. |
| **PyTorch** (`torch`) | `train_unet_*.py`, `eval_test_results*.py`, `architecture.py`. |
| **torchvision** | Used in `architecture.py` (`torchvision.transforms.functional`). Install alongside PyTorch. |
| **matplotlib** | Plotting in training, evaluation, `reproduce_visualizations.py`, `visualize_multifreq_results.py`. |
| **seaborn** | Required by `plot_params.py` (style defaults; imported by eval and reproduce scripts). |
| **tqdm** | Progress bars in training and evaluation loaders. |
| **healpy** | `eval_test_results.py` and `eval_test_results_singlefreq.py` use `healpy` (e.g. `nside2resol`) for patch geometry / spectra metadata. |

## Optional

- **CUDA**-enabled PyTorch: strongly recommended for training and faster evaluation; CPU-only works but is slow for large U-Nets.

## Install examples

**pip** (adjust the `torch` index URL for your CUDA version; see [pytorch.org](https://pytorch.org)):

```bash
pip install numpy matplotlib seaborn tqdm healpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**conda** (channel names and CUDA builds vary by site):

```bash
conda install numpy matplotlib seaborn tqdm healpy pytorch torchvision -c pytorch -c conda-forge
```
