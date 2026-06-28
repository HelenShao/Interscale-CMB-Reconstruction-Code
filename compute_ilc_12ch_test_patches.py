#!/usr/bin/env python3
"""
Compute multi-channel ILC on test patches from test_results.npz.

Default (12 channels): 4 freq x B/T/E, channel order by polarization:
  [B_95, B_145, B_220, B_270, T_95, ..., T_270, E_95, ..., E_270]

Variants (mutually exclusive flags):
  --input-b-only   : 4 channels, all-scale B foregrounds only
  --input-bt-only  : 8 channels, B + T foregrounds (4 freq each)
  --input-be-only  : 8 channels, B + E foregrounds (4 freq each)

Single-frequency variants (--single-freq, default 220 GHz; no 95/145/270):
  --single-freq-b-only   : 1 channel, B @ single freq
  --single-freq-bt-only  : 2 channels, B + T @ single freq
  --single-freq-be-only  : 2 channels, B + E @ single freq
  --single-freq-bte-only : 3 channels, B + T + E @ single freq

Loads native CMB from sim1-150_freq220_cmb_draw_b_all_scales.npy and scales by
cmb_scale (default 0.05, matching UNet training / realistic fg-dominated regime).
apply_ilc_to_single_patch adds the same B-mode CMB to all input channels.

Physics note: T/E channels are foreground-only inputs; B-mode CMB is added to all
channels inside apply_ilc_to_single_patch — an extension beyond 4-freq B-only ILC.

Memory: ~37 GiB peak load for n_test=19200 (12ch); less for 4/8ch variants.
Run on compute node with --mem=80G (see run_ilc_*_test.sbatch).
"""

import argparse
import gc
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Local code-release imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from apply_ilc_core import apply_ilc_to_single_patch

try:
    from cmb_spectrum_2d import calculate_2d_spectrum
except ImportError as e:
    calculate_2d_spectrum = None
    print(f"Warning: calculate_2d_spectrum unavailable ({e}); harmonic correlation skipped")

import paths_config

FREQUENCIES = [95, 145, 220, 270]
DEFAULT_SINGLE_FREQ = 220
CMB_FILE = "sim1-150_freq220_cmb_draw_b_all_scales.npy"
DEFAULT_STATS_FILE = paths_config.STATS_FILE_SINGLEFREQ
DEFAULT_OUTPUT_ROOT = paths_config.ILC_TEST_OUTPUT_ROOT

# Flat-sky cross-spectrum parameters (match eval_test_results_singlefreq / apply_ilc)
PATCH_SIZE = 128
PIX_SIZE = 3  # arcmin, ~NSIDE=1024 resolution
DELTA_ELL = 50
DEFAULT_ELL_MAX = 200  # matches eval_test_results_singlefreq ELL_CUTOFF

CHANNEL_NAMES_12CH = tuple(
    [f"B_{f}" for f in FREQUENCIES]
    + [f"T_{f}" for f in FREQUENCIES]
    + [f"E_{f}" for f in FREQUENCIES]
)
CHANNEL_NAMES_4CH_B = tuple(f"B_{f}" for f in FREQUENCIES)
CHANNEL_NAMES_8CH_BT = tuple(
    [f"B_{f}" for f in FREQUENCIES] + [f"T_{f}" for f in FREQUENCIES]
)
CHANNEL_NAMES_8CH_BE = tuple(
    [f"B_{f}" for f in FREQUENCIES] + [f"E_{f}" for f in FREQUENCIES]
)


def channel_names_single_freq(freq, pols):
    """Build channel names for one frequency, e.g. pols=('B','T','E') -> B_220, T_220, E_220."""
    return tuple(f"{pol}_{freq}" for pol in pols)


def b_channel_indices_from_names(channel_names):
    """Indices of B-mode channels in channel_names (for ilc_fg_b extraction)."""
    return [i for i, name in enumerate(channel_names) if name.startswith("B_")]


def b220_channel_index(channel_names):
    """Index of B_220 in channel_names, or None if absent."""
    target = f"B_{DEFAULT_SINGLE_FREQ}"
    for i, name in enumerate(channel_names):
        if name == target:
            return i
    # Multi-freq: B_220 is index 2 in standard B-first ordering
    for i, name in enumerate(channel_names):
        if name == "B_220":
            return i
    return None


def find_metadata_path(output_dir):
    """Locate ILC_*_test_metadata.npz in output_dir."""
    import glob

    candidates = sorted(glob.glob(os.path.join(output_dir, "ILC_*_test_metadata.npz")))
    if not candidates:
        raise FileNotFoundError(f"Missing metadata in {output_dir}")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple metadata files in {output_dir}: "
            + ", ".join(os.path.basename(p) for p in candidates)
        )
    return candidates[0]


def resolve_channel_config(args):
    """Return (channel_names, n_channels, file_tag, channel_mode, default_output_dir, title, b_ch_idx)."""
    freq = int(args.single_freq)

    if args.single_freq_b_only:
        names = channel_names_single_freq(freq, ("B",))
        tag = f"1ch_b_{freq}"
        return (
            names, 1, tag, f"1ch_b_{freq}GHz",
            f"{DEFAULT_OUTPUT_ROOT}/ILC_products_1ch_b_{freq}_test",
            f"1-channel B-only ILC @ {freq} GHz",
            b_channel_indices_from_names(names),
        )
    if args.single_freq_bt_only:
        names = channel_names_single_freq(freq, ("B", "T"))
        tag = f"2ch_bt_{freq}"
        return (
            names, 2, tag, f"2ch_bt_{freq}GHz",
            f"{DEFAULT_OUTPUT_ROOT}/ILC_products_2ch_bt_{freq}_test",
            f"2-channel B+T ILC @ {freq} GHz",
            b_channel_indices_from_names(names),
        )
    if args.single_freq_be_only:
        names = channel_names_single_freq(freq, ("B", "E"))
        tag = f"2ch_be_{freq}"
        return (
            names, 2, tag, f"2ch_be_{freq}GHz",
            f"{DEFAULT_OUTPUT_ROOT}/ILC_products_2ch_be_{freq}_test",
            f"2-channel B+E ILC @ {freq} GHz",
            b_channel_indices_from_names(names),
        )
    if args.single_freq_bte_only:
        names = channel_names_single_freq(freq, ("B", "T", "E"))
        tag = f"3ch_bte_{freq}"
        return (
            names, 3, tag, f"3ch_bte_{freq}GHz",
            f"{DEFAULT_OUTPUT_ROOT}/ILC_products_3ch_bte_{freq}_test",
            f"3-channel B+T+E ILC @ {freq} GHz",
            b_channel_indices_from_names(names),
        )
    if args.input_b_only:
        names = CHANNEL_NAMES_4CH_B
        return (
            names, 4, "4ch_b", "4ch_b_only",
            f"{DEFAULT_OUTPUT_ROOT}/ILC_products_4ch_b_test",
            "4-channel B-only ILC",
            b_channel_indices_from_names(names),
        )
    if args.input_bt_only:
        names = CHANNEL_NAMES_8CH_BT
        return (
            names, 8, "8ch_bt", "8ch_bt_by_pol",
            f"{DEFAULT_OUTPUT_ROOT}/ILC_products_8ch_bt_test",
            "8-channel B+T ILC",
            b_channel_indices_from_names(names),
        )
    if args.input_be_only:
        names = CHANNEL_NAMES_8CH_BE
        return (
            names, 8, "8ch_be", "8ch_be_by_pol",
            f"{DEFAULT_OUTPUT_ROOT}/ILC_products_8ch_be_test",
            "8-channel B+E ILC",
            b_channel_indices_from_names(names),
        )
    names = CHANNEL_NAMES_12CH
    return (
        names, 12, "12ch", "12ch_BTE_by_pol",
        f"{DEFAULT_OUTPUT_ROOT}/ILC_products_12ch_teb_test",
        "12-channel ILC",
        b_channel_indices_from_names(names),
    )


def output_stem(file_tag):
    return f"ILC_{file_tag}_test"


def fg_path(channel_name, b_dir, te_dir):
    pol, freq_str = channel_name.split("_")
    freq = int(freq_str)
    if pol == "B":
        return os.path.join(b_dir, f"sim1-150_freq{freq}_B_patches.npy")
    return os.path.join(te_dir, f"sim1-150_freq{freq}_{pol}_patches.npy")


def load_test_subset(path, test_indices):
    """Load full .npy into RAM, extract test rows, delete full array."""
    print(f"  Loading: {path}")
    full = np.load(path)
    assert full.shape[0] > test_indices.max(), (
        f"Index {test_indices.max()} out of range for {path} (n={full.shape[0]})"
    )
    subset = full[test_indices].astype(np.float32, copy=True)
    del full
    gc.collect()
    return subset


def spatial_correlation(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    if a_flat.std() == 0 or b_flat.std() == 0:
        return np.nan
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def compute_normalized_cross_spectrum(
    map1, map2, delta_ell=DELTA_ELL, ell_max=DEFAULT_ELL_MAX, pix_size=PIX_SIZE, N=PATCH_SIZE
):
    """Normalized harmonic correlation rho(l) = C_12 / sqrt(C_11 * C_22)."""
    if calculate_2d_spectrum is None:
        raise RuntimeError("calculate_2d_spectrum not available")

    ell_array, cl_cross = calculate_2d_spectrum(
        map1, map2,
        delta_ell=delta_ell, ell_max=ell_max,
        pix_size=pix_size, N=N, lib_mode="NumPy",
    )
    _, cl_map1_map1 = calculate_2d_spectrum(
        map1, map1,
        delta_ell=delta_ell, ell_max=ell_max,
        pix_size=pix_size, N=N, lib_mode="NumPy",
    )
    _, cl_map2_map2 = calculate_2d_spectrum(
        map2, map2,
        delta_ell=delta_ell, ell_max=ell_max,
        pix_size=pix_size, N=N, lib_mode="NumPy",
    )
    normalized_cross_ps = cl_cross / np.sqrt(cl_map1_map1 * cl_map2_map2)
    return ell_array, normalized_cross_ps


def compute_harmonic_correlation(ilc_cmb, cmb_draw, ell_max=DEFAULT_ELL_MAX):
    """
    Compute normalized cross-spectrum (harmonic correlation) between ILC CMB and truth.

    Returns per-patch rho(l), test-set mean/std/percentiles, and scalar summaries
    matching eval_test_results_singlefreq cross_spectrum_stats.txt conventions.
    """
    if calculate_2d_spectrum is None:
        return None

    n_test = len(cmb_draw)
    cross_list = []
    ell_array_ref = None
    estimated_nbins = max(1, int(ell_max / DELTA_ELL))

    print(
        f"\nComputing harmonic correlation (ILC CMB vs true CMB) for {n_test} patches..."
    )
    print(
        f"  delta_ell={DELTA_ELL}, ell_max={ell_max}, pix_size={PIX_SIZE}, N={PATCH_SIZE}"
    )

    for i in tqdm(range(n_test), desc="Harmonic corr"):
        try:
            ell_array, rho = compute_normalized_cross_spectrum(
                cmb_draw[i], ilc_cmb[i], ell_max=ell_max
            )
            cross_list.append(rho)
            if ell_array_ref is None:
                ell_array_ref = ell_array.copy()
        except Exception as exc:
            print(f"    Warning: patch {i}: {exc}")
            if ell_array_ref is not None:
                cross_list.append(np.full_like(ell_array_ref, np.nan))
            else:
                cross_list.append(np.full(estimated_nbins, np.nan))

    cross_array = np.array(cross_list)
    mean_rho = np.nanmean(cross_array, axis=0)
    std_rho = np.nanstd(cross_array, axis=0)
    p16 = np.nanpercentile(cross_array, 16, axis=0)
    p84 = np.nanpercentile(cross_array, 84, axis=0)

    overall_mean = float(np.nanmean(cross_array))
    overall_std = float(np.nanstd(cross_array))
    mean_lower_16 = float(np.nanmean(p16))
    mean_upper_84 = float(np.nanmean(p84))

    return {
        "ell_array": ell_array_ref,
        "cross_spectra": cross_array,
        "mean_rho": mean_rho,
        "std_rho": std_rho,
        "percentile_16": p16,
        "percentile_84": p84,
        "harmonic_corr_mean": overall_mean,
        "harmonic_corr_std": overall_std,
        "mean_lower_16th_across_ell": mean_lower_16,
        "mean_upper_84th_across_ell": mean_upper_84,
        "n_ell_bins": int(len(ell_array_ref)) if ell_array_ref is not None else 0,
    }


def save_cross_spectrum_stats(output_dir, file_tag, harmonic, channel_mode, n_test):
    """Write cross_spectrum_stats.txt compatible with compare_models.py parser."""
    if harmonic is None:
        return

    stats_path = os.path.join(
        output_dir, f"{output_stem(file_tag)}_cross_spectrum_stats.txt"
    )
    ell_array = harmonic["ell_array"]
    mean_rho = harmonic["mean_rho"]
    std_rho = harmonic["std_rho"]
    p16 = harmonic["percentile_16"]
    p84 = harmonic["percentile_84"]
    cross_array = harmonic["cross_spectra"]

    with open(stats_path, "w") as f:
        f.write("ILC CMB vs True CMB Cross-Power Spectrum Statistics\n")
        f.write(f"Channel mode: {channel_mode}\n")
        f.write(f"Number of patches: {n_test}\n")
        f.write(f"Ell bins: {len(ell_array)}\n")
        f.write(f"\n{'=' * 80}\n\n")

        f.write("Ell (multipole) array:\n  [")
        for i, ell in enumerate(ell_array):
            sep = ", " if i < len(ell_array) - 1 else ""
            f.write(f"{ell:.1f}{sep}")
            if (i + 1) % 8 == 0 and i < len(ell_array) - 1:
                f.write("\n   ")
        f.write("]\n")
        f.write(f"\n{'=' * 80}\n\n")

        f.write("Mean cross-spectrum (per ell bin, averaged across patches):\n")
        f.write(f"  Mean across ell: {np.nanmean(mean_rho):.6e}\n")
        f.write(f"  Std across ell: {np.nanstd(mean_rho):.6e}\n\n")
        f.write("Per-ell-bin statistics (mean ± std across patches):\n")
        for i, ell in enumerate(ell_array):
            if not np.isnan(mean_rho[i]):
                f.write(
                    f"  ell={ell:6.1f}: {mean_rho[i]:.6e} ± {std_rho[i]:.6e}\n"
                )

        f.write("\n\n2-tailed percentiles (16th and 84th, 1-sigma equivalent):\n")
        f.write(
            f"  Mean lower (16th percentile, across ell): "
            f"{harmonic['mean_lower_16th_across_ell']:.6e}\n"
        )
        f.write(
            f"  Mean upper (84th percentile, across ell): "
            f"{harmonic['mean_upper_84th_across_ell']:.6e}\n\n"
        )
        f.write("Per-ell-bin 2-tailed percentiles (16th-84th):\n")
        for i, ell in enumerate(ell_array):
            if not np.isnan(p16[i]) and not np.isnan(p84[i]):
                f.write(f"  ell={ell:6.1f}: [{p16[i]:.6e}, {p84[i]:.6e}]\n")

        f.write("\n\nOverall statistics (across all patches and ell bins):\n")
        f.write(f"  Mean cross-spectrum: {harmonic['harmonic_corr_mean']:.6e}\n")
        f.write(f"  Std cross-spectrum: {harmonic['harmonic_corr_std']:.6e}\n")

        f.write(f"\n\n{'=' * 80}\n")
        f.write(
            "Data for plotting (ell, mean_cross_ps, std_cross_ps, "
            "lower_percentile, upper_percentile):\n"
        )
        f.write("Format: ell  mean_cross_ps  std_cross_ps  lower_16th  upper_84th\n")
        f.write(f"{'=' * 80}\n")
        for i, ell in enumerate(ell_array):
            if not np.isnan(mean_rho[i]):
                f.write(
                    f"{ell:6.1f}  {mean_rho[i]:.6e}  {std_rho[i]:.6e}  "
                    f"{p16[i]:.6e}  {p84[i]:.6e}\n"
                )

    print(f"  Saved cross-spectrum stats: {stats_path}")
    print(
        f"  Mean harmonic correlation (all ell, all patches): "
        f"{harmonic['harmonic_corr_mean']:.4f} ± {harmonic['harmonic_corr_std']:.4f}"
    )
    print(
        f"  Mean 16th/84th percentile (across ell): "
        f"{harmonic['mean_lower_16th_across_ell']:.4f} / "
        f"{harmonic['mean_upper_84th_across_ell']:.4f}"
    )
    return stats_path


def validate_inputs(test_indices, n_test, fg_stack, cmb_draw, observed_b_220_ref, channel_names):
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    assert test_indices.max() < 384000
    assert len(np.unique(test_indices)) == n_test
    print(f"  Index sanity: OK (n={n_test}, max idx={test_indices.max()})")

    b220_idx = b220_channel_index(channel_names)
    if b220_idx is None:
        print("  220 GHz B channel not in inputs — skipped observed_b cross-check")
        return

    b_fg_220 = fg_stack[:, b220_idx]
    observed_b_220 = cmb_draw + b_fg_220
    if observed_b_220_ref is not None:
        try:
            np.testing.assert_allclose(
                observed_b_220_ref, observed_b_220, rtol=1e-5, atol=1e-5
            )
            print("  220 GHz observed_b consistency with npz: OK")
        except AssertionError:
            print(
                "  WARNING: observed_b_all_scales in npz does not match "
                "cmb_scale-scaled cmb + B_fg_220 (npz may use different CMB scale)."
            )
            print(f"    max abs diff: {np.max(np.abs(observed_b_220_ref - observed_b_220)):.6e}")
    else:
        print("  observed_b_all_scales not in npz — skipped cross-check")


def run_ilc(fg_stack, cmb_draw, n_test, H, W, n_channels, b_channel_indices):
    print("\n" + "=" * 80)
    print(f"APPLYING {n_channels}-CHANNEL ILC TO {n_test} TEST PATCHES")
    print("=" * 80)

    n_b_out = len(b_channel_indices)
    ilc_cmb = np.zeros((n_test, H, W), dtype=np.float32)
    ilc_residuals = np.zeros((n_test, H, W), dtype=np.float32)
    ilc_fg_all = np.zeros((n_test, n_channels, H, W), dtype=np.float32)
    ilc_fg_b = np.zeros((n_test, n_b_out, H, W), dtype=np.float32)
    ilc_weights = np.zeros((n_test, n_channels), dtype=np.float32)

    singular_fallback_indices = []
    weight_sums = np.zeros(n_test, dtype=np.float64)

    t0 = time.time()
    for i in tqdm(range(n_test), desc="ILC"):
        freq_patches = fg_stack[i]
        cmb_patch = cmb_draw[i]

        ilc_cmb_i, ilc_fg_i, weights, ilc_res_i = apply_ilc_to_single_patch(
            freq_patches, cmb_patch, get_residuals=True
        )

        weight_sums[i] = weights.sum()
        if not np.isclose(weights.sum(), 1.0, atol=1e-5):
            if np.allclose(weights, np.ones(n_channels) / n_channels):
                singular_fallback_indices.append(i)
            else:
                raise AssertionError(
                    f"patch {i}: weights sum={weights.sum()}, expected 1.0"
                )

        ilc_cmb[i] = ilc_cmb_i
        ilc_residuals[i] = ilc_res_i
        ilc_fg_all[i] = ilc_fg_i
        ilc_fg_b[i] = ilc_fg_i[b_channel_indices]
        ilc_weights[i] = weights

    elapsed = time.time() - t0
    print(f"\n  ILC complete in {elapsed:.1f}s ({elapsed / n_test * 1000:.2f} ms/patch)")

    print("\n  Weight sum statistics:")
    print(f"    mean: {weight_sums.mean():.8f} +/- {weight_sums.std():.8f}")
    print(f"    range: [{weight_sums.min():.8f}, {weight_sums.max():.8f}]")
    if singular_fallback_indices:
        print(
            f"    WARNING: {len(singular_fallback_indices)} patches used "
            "equal-weight fallback (singular covariance)"
        )

    return {
        "ilc_cmb": ilc_cmb,
        "ilc_residuals": ilc_residuals,
        "ilc_fg_all": ilc_fg_all,
        "ilc_fg_b": ilc_fg_b,
        "ilc_weights": ilc_weights,
        "weight_sums": weight_sums,
        "singular_fallback_indices": np.array(singular_fallback_indices, dtype=np.int64),
        "elapsed_s": elapsed,
    }


def compute_diagnostics(ilc_cmb, cmb_draw, n_test):
    mse = np.mean((ilc_cmb - cmb_draw) ** 2, axis=(1, 2))
    corrs = np.array(
        [spatial_correlation(ilc_cmb[i], cmb_draw[i]) for i in range(n_test)]
    )
    valid = corrs[~np.isnan(corrs)]

    true_cmb_var = np.var(cmb_draw, axis=(1, 2))
    ilc_cmb_var = np.var(ilc_cmb, axis=(1, 2))
    residual_var = np.var(ilc_cmb - cmb_draw, axis=(1, 2))
    var_ratio = ilc_cmb_var / np.maximum(true_cmb_var, 1e-30)

    return {
        "mse_mean": float(mse.mean()),
        "mse_std": float(mse.std()),
        "corr_mean": float(valid.mean()) if len(valid) else np.nan,
        "corr_std": float(valid.std()) if len(valid) else np.nan,
        "corr_median": float(np.median(valid)) if len(valid) else np.nan,
        "n_corr_gt_0.9": int(np.sum(valid > 0.9)),
        "mse_per_patch": mse,
        "corr_per_patch": corrs,
        "true_cmb_var_per_patch": true_cmb_var,
        "ilc_cmb_var_per_patch": ilc_cmb_var,
        "residual_var_per_patch": residual_var,
        "var_ratio_per_patch": var_ratio,
        "true_cmb_var_mean": float(true_cmb_var.mean()),
        "true_cmb_var_std": float(true_cmb_var.std()),
        "ilc_cmb_var_mean": float(ilc_cmb_var.mean()),
        "ilc_cmb_var_std": float(ilc_cmb_var.std()),
        "residual_var_mean": float(residual_var.mean()),
        "residual_var_std": float(residual_var.std()),
        "var_ratio_mean": float(var_ratio.mean()),
        "var_ratio_std": float(var_ratio.std()),
    }


def format_variance_summary_lines(diagnostics):
    """Lines for summary / variance report (spatial variance per patch)."""
    return [
        "Spatial variance per patch (var over 128x128 pixels):",
        f"  true CMB var mean:  {diagnostics['true_cmb_var_mean']:.6e}",
        f"  true CMB var std:   {diagnostics['true_cmb_var_std']:.6e}",
        f"  ILC CMB var mean:   {diagnostics['ilc_cmb_var_mean']:.6e}",
        f"  ILC CMB var std:    {diagnostics['ilc_cmb_var_std']:.6e}",
        f"  residual var mean:  {diagnostics['residual_var_mean']:.6e}  "
        "(= var(ILC_cmb - true CMB), foreground leakage)",
        f"  residual var std:   {diagnostics['residual_var_std']:.6e}",
        f"  ILC/true var ratio mean: {diagnostics['var_ratio_mean']:.6f}",
        f"  ILC/true var ratio std:  {diagnostics['var_ratio_std']:.6f}",
    ]


def save_variance_stats(output_dir, file_tag, diagnostics, n_test, channel_mode):
    """Save per-patch variance table and print summary."""
    stats_path = os.path.join(output_dir, f"{output_stem(file_tag)}_variance_stats.txt")
    with open(stats_path, "w") as f:
        f.write("# Spatial variance per test patch\n")
        f.write("# channel_mode: {}\n".format(channel_mode))
        f.write("# Format: patch_idx, true_cmb_var, ilc_cmb_var, residual_var, "
                "var_ratio_ilc_over_true\n")
        f.write(f"# Number of patches: {n_test}\n")
        f.write("#\n")
        for i in range(n_test):
            f.write(
                f"{i}  {diagnostics['true_cmb_var_per_patch'][i]:.6e}  "
                f"{diagnostics['ilc_cmb_var_per_patch'][i]:.6e}  "
                f"{diagnostics['residual_var_per_patch'][i]:.6e}  "
                f"{diagnostics['var_ratio_per_patch'][i]:.6f}\n"
            )
        f.write("#\n")
        for line in format_variance_summary_lines(diagnostics):
            f.write(f"# {line}\n")

    print("\n" + "=" * 80)
    print("ILC CMB VARIANCE DIAGNOSTICS")
    print("=" * 80)
    for line in format_variance_summary_lines(diagnostics):
        print(f"  {line}")
    print(f"\nSaved variance stats: {stats_path}")
    return stats_path


def _load_metadata_and_cmb_for_output_dir(output_dir):
    meta_path = find_metadata_path(output_dir)
    meta = np.load(meta_path, allow_pickle=True)
    test_indices = meta["test_indices"]
    file_tag = str(meta["file_tag"]) if "file_tag" in meta else "12ch"
    channel_mode = str(meta["channel_mode"])
    n_test = len(test_indices)
    cmb_scale = float(meta["cmb_scale"])
    cmb_path = str(meta["cmb_source"])

    ilc_path = os.path.join(output_dir, f"ILC_cmb_{file_tag}_test.npy")
    if not os.path.isfile(ilc_path):
        raise FileNotFoundError(f"Missing ILC CMB: {ilc_path}")

    ilc_cmb_mm = np.load(ilc_path, mmap_mode="r")
    cmb_full = np.load(cmb_path, mmap_mode="r")
    n_test = ilc_cmb_mm.shape[0]
    ilc_cmb = np.zeros((n_test, ilc_cmb_mm.shape[1], ilc_cmb_mm.shape[2]), dtype=np.float32)
    cmb_draw = np.zeros_like(ilc_cmb)
    chunk = 512
    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        ilc_cmb[start:end] = np.array(ilc_cmb_mm[start:end], dtype=np.float32)
        ds_rows = test_indices[start:end]
        cmb_draw[start:end] = np.array(cmb_full[ds_rows], dtype=np.float32) * cmb_scale

    return {
        "output_dir": output_dir,
        "file_tag": file_tag,
        "channel_mode": channel_mode,
        "n_test": n_test,
        "ilc_cmb": ilc_cmb,
        "cmb_draw": cmb_draw,
    }


def regenerate_variance_diagnostics(output_dir):
    """Compute variance stats from saved ILC_cmb + true CMB."""
    loaded = _load_metadata_and_cmb_for_output_dir(output_dir)
    diagnostics = compute_diagnostics(loaded["ilc_cmb"], loaded["cmb_draw"], loaded["n_test"])
    save_variance_stats(
        output_dir,
        loaded["file_tag"],
        diagnostics,
        loaded["n_test"],
        loaded["channel_mode"],
    )
    return diagnostics, loaded


def compare_variance_across_dirs(output_dirs):
    """Print comparison table of ILC CMB variance across multiple runs."""
    rows = []
    for output_dir in output_dirs:
        loaded = _load_metadata_and_cmb_for_output_dir(output_dir)
        diagnostics = compute_diagnostics(
            loaded["ilc_cmb"], loaded["cmb_draw"], loaded["n_test"]
        )
        save_variance_stats(
            output_dir,
            loaded["file_tag"],
            diagnostics,
            loaded["n_test"],
            loaded["channel_mode"],
        )
        rows.append((loaded["channel_mode"], loaded["n_test"], diagnostics))

    print("\n" + "=" * 80)
    print("CROSS-RUN ILC CMB VARIANCE COMPARISON")
    print("=" * 80)
    header = (
        f"{'channel_mode':<20} {'n_test':>8} "
        f"{'true_var':>12} {'ilc_var':>12} {'resid_var':>12} {'ilc/true':>10}"
    )
    print(header)
    print("-" * len(header))
    for channel_mode, n_test, diag in rows:
        print(
            f"{channel_mode:<20} {n_test:8d} "
            f"{diag['true_cmb_var_mean']:12.6e} "
            f"{diag['ilc_cmb_var_mean']:12.6e} "
            f"{diag['residual_var_mean']:12.6e} "
            f"{diag['var_ratio_mean']:10.6f}"
        )
    print(
        "\nNote: true CMB variance should match across runs (same test set & cmb_scale). "
        "Higher ILC CMB var / residual var => more foreground leakage (sum_i w_i f_i)."
    )


def _save_one_sample_figure(
    figures_dir,
    arr_idx,
    ds_idx,
    true,
    recon,
    resid,
    fg_patches,
    fg_titles,
    mse_val,
    corr_val,
    weight_sum,
    args,
    title,
):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    panels_row0 = [
        (true, f"True CMB (cmb_scale={args.cmb_scale})"),
        (recon, "ILC CMB reconstruction"),
        (resid, "ILC residuals (recon - true)"),
    ]
    vmax_cmb = max(max(np.abs(p[0]).max() for p in panels_row0[:2]), 1e-12)
    for ax, (data, panel_title) in zip(axes[0, :3], panels_row0):
        if "residual" in panel_title.lower():
            vmax = max(np.abs(data).max(), 1e-12)
        else:
            vmax = vmax_cmb
        im = ax.imshow(data, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_title(panel_title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    axes[0, 3].axis("off")
    stats_text = (
        f"dataset index: {ds_idx}\n"
        f"array index: {arr_idx}\n"
        f"MSE: {mse_val:.6e}\n"
        f"spatial corr: {corr_val:.4f}\n"
        f"weight sum: {weight_sum:.6f}"
    )
    axes[0, 3].text(
        0.05, 0.5, stats_text, transform=axes[0, 3].transAxes,
        fontsize=11, verticalalignment="center", family="monospace",
    )

    vmax_fg = max(max(np.abs(fg).max() for fg in fg_patches), 1e-12)
    for j in range(4):
        ax = axes[1, j]
        if j < len(fg_patches):
            fg = fg_patches[j]
            im = ax.imshow(fg, cmap="RdBu_r", origin="lower", vmin=-vmax_fg, vmax=vmax_fg)
            ax.set_title(fg_titles[j])
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.axis("off")

    fig.suptitle(f"{title} — test patch array_idx={arr_idx}, dataset_idx={ds_idx}")
    fig.tight_layout()
    out_path = os.path.join(figures_dir, f"sample_{arr_idx}_dataset_{ds_idx}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def save_sample_figures(
    figures_dir,
    sample_array_indices,
    test_indices,
    cmb_draw,
    results,
    diagnostics,
    args,
    title,
    channel_names,
    b_channel_indices,
):
    """Save per-patch PNG comparisons under figures_dir."""
    os.makedirs(figures_dir, exist_ok=True)

    ilc_cmb = results["ilc_cmb"]
    ilc_residuals = results["ilc_residuals"]
    ilc_fg_b = results["ilc_fg_b"]
    mse = diagnostics["mse_per_patch"]
    corrs = diagnostics["corr_per_patch"]
    n_b = ilc_fg_b.shape[1]
    fg_titles = [
        f"ILC {channel_names[b_channel_indices[ch]]} fg leakage"
        for ch in range(n_b)
    ]

    for arr_idx in sample_array_indices:
        ds_idx = int(test_indices[arr_idx])
        _save_one_sample_figure(
            figures_dir,
            arr_idx,
            ds_idx,
            cmb_draw[arr_idx],
            ilc_cmb[arr_idx],
            ilc_residuals[arr_idx],
            [ilc_fg_b[arr_idx, ch] for ch in range(n_b)],
            fg_titles,
            float(mse[arr_idx]),
            float(corrs[arr_idx]),
            float(results["ilc_weights"][arr_idx].sum()),
            args,
            title,
        )


def save_summary_figures(figures_dir, diagnostics, results, n_test):
    """Global diagnostic PNGs (MSE / correlation / weight sums)."""
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(diagnostics["mse_per_patch"], bins=50, edgecolor="black", alpha=0.75)
    axes[0].set_xlabel("MSE (ILC CMB vs true CMB)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"MSE distribution (n={n_test})")

    valid = diagnostics["corr_per_patch"][~np.isnan(diagnostics["corr_per_patch"])]
    axes[1].hist(valid, bins=50, edgecolor="black", alpha=0.75)
    axes[1].set_xlabel("Spatial correlation")
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        f"Corr distribution (median={np.median(valid):.3f})"
    )

    axes[2].hist(results["weight_sums"], bins=50, edgecolor="black", alpha=0.75)
    axes[2].axvline(1.0, color="red", linestyle="--", label="target=1.0")
    axes[2].set_xlabel("sum(ILC weights)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Weight sum distribution")
    axes[2].legend()

    fig.tight_layout()
    summary_png = os.path.join(figures_dir, "summary_diagnostics.png")
    fig.savefig(summary_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {summary_png}")


def parse_sample_indices(spec, n_test):
    if spec is None or str(spec).strip() == "":
        return list(range(min(5, n_test)))
    indices = [int(x.strip()) for x in str(spec).split(",")]
    for idx in indices:
        if idx < 0 or idx >= n_test:
            raise ValueError(f"sample index {idx} out of range [0, {n_test})")
    return indices


def resolve_weight_sample_indices(args, n_test):
    spec = args.weight_sample_indices
    if spec is None or str(spec).strip() == "":
        spec = args.sample_indices
    return parse_sample_indices(spec, n_test)


def polarization_group_indices(channel_names):
    """Map polarization label -> list of channel indices."""
    groups = {"B": [], "T": [], "E": []}
    for idx, name in enumerate(channel_names):
        pol = name.split("_")[0]
        if pol in groups:
            groups[pol].append(idx)
    return groups


def summarize_weights_by_polarization(mean_weights, channel_names):
    """Return dict of polarization -> summed mean weight across its channels."""
    groups = polarization_group_indices(channel_names)
    return {
        pol: float(np.sum(mean_weights[indices])) if indices else 0.0
        for pol, indices in groups.items()
    }


def format_weight_lines(
    arr_idx,
    ds_idx,
    weights,
    channel_names,
    mse_val=None,
    corr_val=None,
):
    """Format per-channel ILC weights for one patch."""
    lines = [
        f"  array_idx={arr_idx}, dataset_idx={ds_idx}",
    ]
    if mse_val is not None:
        lines.append(f"  MSE(CMB): {mse_val:.6e}")
    if corr_val is not None:
        lines.append(f"  spatial corr(CMB): {corr_val:.6f}")
    lines.append(f"  sum(weights): {float(np.sum(weights)):.8f}")
    lines.append("  channel weights:")
    for ch, name in enumerate(channel_names):
        lines.append(f"    [{ch:2d}] {name:8s}  {weights[ch]:+.8f}")
    pol_sums = summarize_weights_by_polarization(weights, channel_names)
    if any(pol_sums[p] != 0.0 for p in ("T", "E")):
        lines.append(
            "  polarization sums: "
            + ", ".join(f"{pol}={pol_sums[pol]:+.6f}" for pol in ("B", "T", "E"))
        )
    return lines


def build_weight_diagnostics_report(
    ilc_weights,
    channel_names,
    test_indices,
    sample_array_indices,
    diagnostics=None,
    channel_mode=None,
):
    """Build stdout + file report for global and sample ILC weights."""
    n_test, n_channels = ilc_weights.shape
    mean_w = ilc_weights.mean(axis=0)
    std_w = ilc_weights.std(axis=0)
    pol_mean = summarize_weights_by_polarization(mean_w, channel_names)

    lines = [
        "ILC WEIGHT DIAGNOSTICS",
        "=" * 60,
        f"channel_mode: {channel_mode}",
        f"n_test: {n_test}, n_channels: {n_channels}",
        f"channels: {', '.join(channel_names)}",
        "",
        "Global mean weight per channel (across all test patches):",
    ]
    for ch, name in enumerate(channel_names):
        lines.append(
            f"  [{ch:2d}] {name:8s}  mean={mean_w[ch]:+.8f}  std={std_w[ch]:.8f}"
        )
    lines.extend([
        "",
        "Global mean weight summed by polarization:",
        f"  B: {pol_mean['B']:+.8f}",
        f"  T: {pol_mean['T']:+.8f}",
        f"  E: {pol_mean['E']:+.8f}",
        "",
        f"Sample patches (array indices: {sample_array_indices}):",
        "-" * 60,
    ])

    for arr_idx in sample_array_indices:
        ds_idx = int(test_indices[arr_idx])
        mse_val = None
        corr_val = None
        if diagnostics is not None:
            mse_val = float(diagnostics["mse_per_patch"][arr_idx])
            corr_val = float(diagnostics["corr_per_patch"][arr_idx])
        lines.extend(
            format_weight_lines(
                arr_idx,
                ds_idx,
                ilc_weights[arr_idx],
                channel_names,
                mse_val=mse_val,
                corr_val=corr_val,
            )
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def print_and_save_weight_diagnostics(
    output_dir,
    file_tag,
    ilc_weights,
    channel_names,
    test_indices,
    sample_array_indices,
    diagnostics=None,
    channel_mode=None,
):
    """Print sample ILC weights and save full weight report to output_dir."""
    report = build_weight_diagnostics_report(
        ilc_weights,
        channel_names,
        test_indices,
        sample_array_indices,
        diagnostics=diagnostics,
        channel_mode=channel_mode,
    )
    print("\n" + "=" * 80)
    print("ILC WEIGHT SAMPLES")
    print("=" * 80)
    print(report, end="")

    report_path = os.path.join(output_dir, f"{output_stem(file_tag)}_weights_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved weight report: {report_path}")
    return report_path


def regenerate_weight_diagnostics(output_dir, args):
    """Print/save weight diagnostics from existing ILC_weights_*.npy outputs."""
    meta_path = find_metadata_path(output_dir)
    meta = np.load(meta_path, allow_pickle=True)
    test_indices = meta["test_indices"]
    channel_names = [str(x) for x in meta["channel_names"]]
    file_tag = str(meta["file_tag"]) if "file_tag" in meta else "12ch"
    channel_mode = str(meta["channel_mode"])
    n_test = len(test_indices)

    weights_path = os.path.join(output_dir, f"ILC_weights_{file_tag}_test.npy")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    ilc_weights = np.load(weights_path)
    sample_indices = resolve_weight_sample_indices(args, n_test)

    diagnostics = None
    cmb_path = os.path.join(output_dir, f"ILC_cmb_{file_tag}_test.npy")
    if os.path.isfile(cmb_path):
        ilc_cmb = np.load(cmb_path, mmap_mode="r")
        cmb_full = np.load(str(meta["cmb_source"]), mmap_mode="r")
        cmb_scale = float(meta["cmb_scale"])
        mse = np.full(n_test, np.nan, dtype=np.float64)
        corrs = np.full(n_test, np.nan, dtype=np.float64)
        for arr_idx in sample_indices:
            ds_idx = int(test_indices[arr_idx])
            true = np.array(cmb_full[ds_idx], dtype=np.float32) * cmb_scale
            recon = np.array(ilc_cmb[arr_idx], dtype=np.float32)
            mse[arr_idx] = float(np.mean((recon - true) ** 2))
            corrs[arr_idx] = spatial_correlation(recon, true)
        diagnostics = {
            "mse_per_patch": mse,
            "corr_per_patch": corrs,
        }

    print_and_save_weight_diagnostics(
        output_dir,
        file_tag,
        ilc_weights,
        channel_names,
        test_indices,
        sample_indices,
        diagnostics=diagnostics,
        channel_mode=channel_mode,
    )


def regenerate_harmonic(output_dir, ell_max):
    """Recompute harmonic correlation from saved ILC CMB + true CMB (mmap)."""
    meta_path = find_metadata_path(output_dir)
    meta = np.load(meta_path, allow_pickle=True)
    test_indices = meta["test_indices"]
    cmb_scale = float(meta["cmb_scale"])
    cmb_path = str(meta["cmb_source"])
    file_tag = str(meta["file_tag"]) if "file_tag" in meta else "12ch"
    channel_mode = str(meta["channel_mode"])
    n_test = len(test_indices)

    ilc_cmb_mm = np.load(
        os.path.join(output_dir, f"ILC_cmb_{file_tag}_test.npy"), mmap_mode="r"
    )
    cmb_full = np.load(cmb_path, mmap_mode="r")

    ilc_cmb = np.zeros((n_test, ilc_cmb_mm.shape[1], ilc_cmb_mm.shape[2]), dtype=np.float32)
    cmb_draw = np.zeros_like(ilc_cmb)
    chunk = 512
    print(f"Loading ILC CMB + true CMB in chunks of {chunk}...")
    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        ilc_cmb[start:end] = np.array(ilc_cmb_mm[start:end], dtype=np.float32)
        ds_rows = test_indices[start:end]
        cmb_draw[start:end] = np.array(cmb_full[ds_rows], dtype=np.float32) * cmb_scale

    harmonic = compute_harmonic_correlation(ilc_cmb, cmb_draw, ell_max=ell_max)
    if harmonic is not None:
        np.save(
            os.path.join(output_dir, f"ILC_cmb_cross_spectrum_{file_tag}_test.npy"),
            harmonic["cross_spectra"],
        )
        save_cross_spectrum_stats(output_dir, file_tag, harmonic, channel_mode, n_test)
    return harmonic


def regenerate_figures(output_dir, args):
    """Regenerate sample PNGs from saved ILC outputs (mmap, low memory)."""
    meta_path = find_metadata_path(output_dir)
    meta = np.load(meta_path, allow_pickle=True)
    test_indices = meta["test_indices"]
    cmb_scale = float(meta["cmb_scale"])
    cmb_path = str(meta["cmb_source"])
    file_tag = str(meta["file_tag"]) if "file_tag" in meta else "12ch"
    title = str(meta["ilc_title"]) if "ilc_title" in meta else "12-channel ILC"
    n_test = len(test_indices)
    stem = output_stem(file_tag)

    args.cmb_scale = cmb_scale
    figures_dir = os.path.join(output_dir, "figures")
    sample_indices = parse_sample_indices(args.sample_indices, n_test)

    print(f"Regenerating sample figures from: {output_dir}")
    print(f"  Channel mode: {meta['channel_mode']}")
    print(f"  Sample array indices: {sample_indices}")

    ilc_cmb = np.load(os.path.join(output_dir, f"ILC_cmb_{file_tag}_test.npy"), mmap_mode="r")
    ilc_residuals = np.load(
        os.path.join(output_dir, f"ILC_residuals_{file_tag}_test.npy"), mmap_mode="r"
    )
    ilc_fg_b = np.load(
        os.path.join(output_dir, f"ILC_fg_b_{file_tag}_test.npy"), mmap_mode="r"
    )
    ilc_weights = np.load(
        os.path.join(output_dir, f"ILC_weights_{file_tag}_test.npy"), mmap_mode="r"
    )
    cmb_full = np.load(cmb_path, mmap_mode="r")

    channel_names = [str(x) for x in meta["channel_names"]]
    if "b_channel_indices" in meta:
        b_channel_indices = [int(x) for x in meta["b_channel_indices"]]
    else:
        b_channel_indices = b_channel_indices_from_names(channel_names)

    os.makedirs(figures_dir, exist_ok=True)
    n_b = ilc_fg_b.shape[1]
    fg_titles = [
        f"ILC {channel_names[b_channel_indices[ch]]} fg leakage"
        for ch in range(n_b)
    ]
    for arr_idx in sample_indices:
        ds_idx = int(test_indices[arr_idx])
        true = np.array(cmb_full[ds_idx], dtype=np.float32) * cmb_scale
        recon = np.array(ilc_cmb[arr_idx], dtype=np.float32)
        resid = np.array(ilc_residuals[arr_idx], dtype=np.float32)
        fg_patches = [
            np.array(ilc_fg_b[arr_idx, ch], dtype=np.float32) for ch in range(n_b)
        ]
        mse_val = float(np.mean((recon - true) ** 2))
        corr_val = spatial_correlation(recon, true)
        weight_sum = float(ilc_weights[arr_idx].sum())
        _save_one_sample_figure(
            figures_dir,
            arr_idx,
            ds_idx,
            true,
            recon,
            resid,
            fg_patches,
            fg_titles,
            mse_val,
            corr_val,
            weight_sum,
            args,
            title,
        )

    print(f"\nDone. Figures saved to: {figures_dir}")


def save_outputs(
    output_dir,
    results,
    diagnostics,
    test_indices,
    args,
    cmb_path,
    cmb_draw,
    channel_names,
    n_channels,
    file_tag,
    channel_mode,
    title,
    b_channel_indices,
    harmonic=None,
):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f"ILC_cmb_{file_tag}_test.npy"), results["ilc_cmb"])
    np.save(
        os.path.join(output_dir, f"ILC_residuals_{file_tag}_test.npy"),
        results["ilc_residuals"],
    )
    np.save(os.path.join(output_dir, f"ILC_fg_b_{file_tag}_test.npy"), results["ilc_fg_b"])
    np.save(
        os.path.join(output_dir, f"ILC_fg_all_{file_tag}_test.npy"), results["ilc_fg_all"]
    )
    np.save(
        os.path.join(output_dir, f"ILC_weights_{file_tag}_test.npy"), results["ilc_weights"]
    )
    if harmonic is not None:
        np.save(
            os.path.join(output_dir, f"ILC_cmb_cross_spectrum_{file_tag}_test.npy"),
            harmonic["cross_spectra"],
        )
        if harmonic["ell_array"] is not None:
            np.savez(
                os.path.join(output_dir, f"{output_stem(file_tag)}_harmonic_summary.npz"),
                ell_array=harmonic["ell_array"],
                mean_rho=harmonic["mean_rho"],
                std_rho=harmonic["std_rho"],
                percentile_16=harmonic["percentile_16"],
                percentile_84=harmonic["percentile_84"],
                harmonic_corr_mean=np.float64(harmonic["harmonic_corr_mean"]),
                harmonic_corr_std=np.float64(harmonic["harmonic_corr_std"]),
                mean_lower_16th_across_ell=np.float64(harmonic["mean_lower_16th_across_ell"]),
                mean_upper_84th_across_ell=np.float64(harmonic["mean_upper_84th_across_ell"]),
            )

    np.savez(
        os.path.join(output_dir, f"{output_stem(file_tag)}_metadata.npz"),
        test_indices=test_indices,
        channel_names=np.array(channel_names),
        frequencies=np.array(FREQUENCIES),
        cmb_scale=np.float64(args.cmb_scale),
        cmb_source=np.array(cmb_path),
        test_results_npz=np.array(args.test_results_npz),
        n_test=np.int32(len(test_indices)),
        n_channels=np.int32(n_channels),
        channel_mode=np.array(channel_mode),
        file_tag=np.array(file_tag),
        ilc_title=np.array(title),
        b_channel_indices=np.array(b_channel_indices, dtype=np.int32),
    )

    n_singular = len(results["singular_fallback_indices"])
    summary_lines = [
        f"{n_channels}-CHANNEL ILC TEST PATCHES SUMMARY",
        "=" * 60,
        f"test_results_npz: {args.test_results_npz}",
        f"n_test: {len(test_indices)}",
        f"n_channels: {n_channels}",
        f"cmb_source: {cmb_path} (cmb_scale={args.cmb_scale})",
        f"channel_mode: {channel_mode}",
        f"channels: {', '.join(channel_names)}",
        "",
        "Weight sums (should be 1.0):",
        f"  mean: {results['weight_sums'].mean():.8f}",
        f"  std:  {results['weight_sums'].std():.8f}",
        f"  min:  {results['weight_sums'].min():.8f}",
        f"  max:  {results['weight_sums'].max():.8f}",
        f"  singular-covariance fallbacks: {n_singular}",
        "",
        "ILC CMB vs true cmb_draw (same cmb_scale applied to truth):",
        f"  MSE mean: {diagnostics['mse_mean']:.6e}",
        f"  MSE std:  {diagnostics['mse_std']:.6e}",
        f"  Spatial corr mean:   {diagnostics['corr_mean']:.4f}",
        f"  Spatial corr std:    {diagnostics['corr_std']:.4f}",
        f"  Spatial corr median: {diagnostics['corr_median']:.4f}",
        f"  Patches with corr > 0.9: "
        f"{diagnostics['n_corr_gt_0.9']} / {len(test_indices)}",
        "",
    ]
    summary_lines.extend(format_variance_summary_lines(diagnostics))
    if harmonic is not None:
        summary_lines.extend([
            "",
            "Harmonic correlation (ILC CMB vs true CMB, normalized cross-spectrum):",
            f"  Mean rho (all ell, all patches): {harmonic['harmonic_corr_mean']:.6f}",
            f"  Std rho (all ell, all patches):  {harmonic['harmonic_corr_std']:.6f}",
            f"  Mean 16th percentile (across ell): {harmonic['mean_lower_16th_across_ell']:.6f}",
            f"  Mean 84th percentile (across ell): {harmonic['mean_upper_84th_across_ell']:.6f}",
            f"  n_ell_bins: {harmonic['n_ell_bins']}",
        ])
    summary_lines.extend([
        "",
        f"Runtime: {results['elapsed_s']:.1f}s",
    ])
    summary_path = os.path.join(output_dir, f"{output_stem(file_tag)}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    if harmonic is not None:
        save_cross_spectrum_stats(
            output_dir, file_tag, harmonic, channel_mode, len(test_indices)
        )

    save_variance_stats(
        output_dir, file_tag, diagnostics, len(test_indices), channel_mode
    )

    figures_dir = os.path.join(output_dir, "figures")
    sample_indices = parse_sample_indices(args.sample_indices, len(test_indices))
    print(f"\nSaving PNG figures to: {figures_dir}")
    print(f"  Sample array indices: {sample_indices}")
    save_sample_figures(
        figures_dir,
        sample_indices,
        test_indices,
        cmb_draw,
        results,
        diagnostics,
        args,
        title,
        channel_names,
        b_channel_indices,
    )
    save_summary_figures(figures_dir, diagnostics, results, len(test_indices))

    weight_sample_indices = resolve_weight_sample_indices(args, len(test_indices))
    print_and_save_weight_diagnostics(
        output_dir,
        file_tag,
        results["ilc_weights"],
        channel_names,
        test_indices,
        weight_sample_indices,
        diagnostics=diagnostics,
        channel_mode=channel_mode,
    )

    print(f"\nSaved outputs to: {output_dir}")
    print(f"Saved figures to: {figures_dir}")
    for name in [
        f"ILC_cmb_{file_tag}_test.npy",
        f"ILC_residuals_{file_tag}_test.npy",
        f"ILC_fg_b_{file_tag}_test.npy",
        f"ILC_fg_all_{file_tag}_test.npy",
        f"ILC_weights_{file_tag}_test.npy",
        f"{output_stem(file_tag)}_metadata.npz",
        f"{output_stem(file_tag)}_summary.txt",
        f"{output_stem(file_tag)}_weights_report.txt",
        f"{output_stem(file_tag)}_variance_stats.txt",
    ]:
        print(f"  {name}")
    if harmonic is not None:
        print(f"  ILC_cmb_cross_spectrum_{file_tag}_test.npy")
        print(f"  {output_stem(file_tag)}_cross_spectrum_stats.txt")
        print(f"  {output_stem(file_tag)}_harmonic_summary.npz")
    print("  figures/sample_*_dataset_*.png")
    print("  figures/summary_diagnostics.png")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-channel ILC on test patches (B/T/E @ 4 frequencies)"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--input-b-only",
        action="store_true",
        help="4 channels: all-scale B foregrounds only (exclude T, E)",
    )
    mode_group.add_argument(
        "--input-bt-only",
        action="store_true",
        help="8 channels: B + T foregrounds (4 freq each, exclude E)",
    )
    mode_group.add_argument(
        "--input-be-only",
        action="store_true",
        help="8 channels: B + E foregrounds (4 freq each, exclude T)",
    )
    mode_group.add_argument(
        "--single-freq-b-only",
        action="store_true",
        help="1 channel: B foreground at --single-freq only (default 220 GHz)",
    )
    mode_group.add_argument(
        "--single-freq-bt-only",
        action="store_true",
        help="2 channels: B + T at --single-freq only",
    )
    mode_group.add_argument(
        "--single-freq-be-only",
        action="store_true",
        help="2 channels: B + E at --single-freq only",
    )
    mode_group.add_argument(
        "--single-freq-bte-only",
        action="store_true",
        help="3 channels: B + T + E at --single-freq only",
    )
    parser.add_argument(
        "--single-freq",
        type=int,
        default=DEFAULT_SINGLE_FREQ,
        help=f"Frequency (GHz) for single-freq modes (default {DEFAULT_SINGLE_FREQ})",
    )
    parser.add_argument(
        "--test-results-npz",
        default=(
            "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/"
            "unnormalized_teb/test_results.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default depends on channel mode)",
    )
    parser.add_argument(
        "--b-dir",
        default="/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_B",
    )
    parser.add_argument(
        "--te-dir",
        default="/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_T_E",
    )
    parser.add_argument(
        "--stats-file",
        default=DEFAULT_STATS_FILE,
        help="Normalization stats npz (validates test_indices against canonical split)",
    )
    parser.add_argument(
        "--cmb-scale",
        type=float,
        default=0.05,
        help="Scale factor applied to native cmb_draw (default 0.05, matches UNet training)",
    )
    parser.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated array indices into test set for PNG plots (default: first 5)",
    )
    parser.add_argument(
        "--weight-sample-indices",
        default=None,
        help=(
            "Comma-separated array indices for ILC weight printing "
            "(default: same as --sample-indices, or first 5)"
        ),
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate PNG figures from saved .npy outputs (skip ILC recompute)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Print/save ILC weight diagnostics from saved outputs (skip ILC recompute)",
    )
    parser.add_argument(
        "--variance-only",
        action="store_true",
        help="Compute ILC CMB spatial variance stats from saved outputs",
    )
    parser.add_argument(
        "--compare-variance",
        nargs="+",
        metavar="OUTPUT_DIR",
        default=None,
        help="Compare ILC CMB variance across multiple saved output directories",
    )
    parser.add_argument(
        "--harmonic-only",
        action="store_true",
        help="Recompute harmonic correlation from saved ILC CMB outputs",
    )
    parser.add_argument(
        "--skip-harmonic",
        action="store_true",
        help="Skip harmonic correlation computation during full ILC run",
    )
    parser.add_argument(
        "--ell-max",
        type=int,
        default=DEFAULT_ELL_MAX,
        help=f"Max ell for harmonic correlation (default {DEFAULT_ELL_MAX})",
    )
    args = parser.parse_args()

    (
        channel_names,
        n_channels,
        file_tag,
        channel_mode,
        default_output_dir,
        title,
        b_channel_indices,
    ) = resolve_channel_config(args)
    if args.output_dir is None:
        args.output_dir = default_output_dir

    single_freq_modes = (
        args.single_freq_b_only
        or args.single_freq_bt_only
        or args.single_freq_be_only
        or args.single_freq_bte_only
    )
    if single_freq_modes and args.single_freq not in FREQUENCIES:
        parser.error(
            f"--single-freq {args.single_freq} not in available frequencies {FREQUENCIES}"
        )

    if args.compare_variance:
        compare_variance_across_dirs(args.compare_variance)
        return

    if args.figures_only:
        regenerate_figures(args.output_dir, args)
        return

    if args.weights_only:
        regenerate_weight_diagnostics(args.output_dir, args)
        return

    if args.variance_only:
        regenerate_variance_diagnostics(args.output_dir)
        return

    if args.harmonic_only:
        regenerate_harmonic(args.output_dir, args.ell_max)
        return

    print("=" * 80)
    print(f"{n_channels}-CHANNEL ILC ON TEST PATCHES ({channel_mode})")
    print("=" * 80)

    data = np.load(args.test_results_npz)
    test_indices = data["test_indices"]
    n_test = len(test_indices)

    stats = np.load(args.stats_file)
    ref_indices = stats["test_indices"]
    assert len(test_indices) == len(ref_indices), (
        f"test_indices length {n_test} != stats test split {len(ref_indices)}"
    )
    assert np.array_equal(test_indices, ref_indices), (
        "test_indices in test_results.npz do not match singlefreq_normalization_stats.npz"
    )
    print(f"test_indices: n={n_test}, range=[{test_indices.min()}, {test_indices.max()}]")

    observed_b_220_ref = data.get("observed_b_all_scales", None)

    cmb_path = os.path.join(args.b_dir, CMB_FILE)
    print("\nLoading primary CMB (native amplitude from .npy)...")
    cmb_draw = load_test_subset(cmb_path, test_indices)
    cmb_draw = cmb_draw * args.cmb_scale
    print(f"  Applied cmb_scale={args.cmb_scale}")

    H, W = cmb_draw.shape[1], cmb_draw.shape[2]
    print(f"  cmb_draw shape: {cmb_draw.shape}")
    print(f"  output_dir: {args.output_dir}")

    print(f"\nLoading {n_channels} foreground channels (full load, extract, del each file)...")
    fg_stack = np.zeros((n_test, n_channels, H, W), dtype=np.float32)
    for ch, name in enumerate(channel_names):
        path = fg_path(name, args.b_dir, args.te_dir)
        fg_stack[:, ch] = load_test_subset(path, test_indices)
        print(f"  [{ch + 1}/{n_channels}] {name} loaded")

    validate_inputs(
        test_indices, n_test, fg_stack, cmb_draw, observed_b_220_ref, channel_names
    )

    results = run_ilc(
        fg_stack, cmb_draw, n_test, H, W, n_channels, b_channel_indices
    )
    diagnostics = compute_diagnostics(results["ilc_cmb"], cmb_draw, n_test)

    harmonic = None
    if not args.skip_harmonic:
        harmonic = compute_harmonic_correlation(
            results["ilc_cmb"], cmb_draw, ell_max=args.ell_max
        )

    save_outputs(
        args.output_dir,
        results,
        diagnostics,
        test_indices,
        args,
        cmb_path,
        cmb_draw,
        channel_names,
        n_channels,
        file_tag,
        channel_mode,
        title,
        b_channel_indices,
        harmonic=harmonic,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
