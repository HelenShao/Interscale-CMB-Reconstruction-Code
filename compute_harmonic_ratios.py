#!/usr/bin/env python3
"""Compute harmonic correlation rho and B-only ratios for V7 plot models."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from compare_models import (
    LABEL_BS_ONLY,
    LABEL_E_ONLY,
    LABEL_T_ONLY,
    LABEL_TE,
    LABEL_TEB,
    parse_cross_spectrum_file,
)

MODELS = [
    ("B_S-only", LABEL_BS_ONLY, "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_cmb_scaled/normalized_b_only", "normalized_b_only_best_model_153600_19200_norm"),
    ("T+E+B_S", LABEL_TEB, "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb", "unnormalized_teb_best_model_153600_19200_nonorm"),
    ("T-only", LABEL_T_ONLY, "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_t_only", "unnormalized_t_only_best_model_153600_19200_nonorm"),
    ("E-only", LABEL_E_ONLY, "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only", "unnormalized_e_only_best_model_153600_19200_nonorm"),
    ("T+E", LABEL_TE, "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_te_only", "unnormalized_te_only_best_model_153600_19200_nonorm"),
]

OUTPUT_PATH = "/scratch/gpfs/JDUNKLEY/hshao/old_data/model_comparison/rho_philcox18_t_b_e_te_teb_ratios.txt"


def main():
    data = {}
    for key, label, base, prefix in MODELS:
        path = os.path.join(base, f"{prefix}_cross_spectrum_stats.txt")
        data[key] = parse_cross_spectrum_file(path)

    ells = data["B_S-only"]["ell_array"]
    lines = []
    lines.append("Harmonic correlation rho_F,Fhat(l) from cross_spectrum_stats mean_cross_ps")
    lines.append("(normalized cross-power: C_cross / sqrt(C_pred * C_target))")
    lines.append("")

    lines.append("=" * 90)
    lines.append("TABLE 1: Absolute harmonic correlation  rho(l)")
    lines.append("=" * 90)
    header = "{:>6}".format("ell") + "".join("{:>12}".format(k) for k, _, _, _ in MODELS)
    lines.append(header)
    lines.append("-" * len(header))
    for i, ell_val in enumerate(ells):
        row = "{:6.0f}".format(ell_val)
        for key, _, _, _ in MODELS:
            row += "{:12.6f}".format(data[key]["mean_cross_ps"][i])
        lines.append(row)

    lines.append("")
    lines.append("Std across patches (per ell):")
    for key, label, _, _ in MODELS:
        lines.append(f"  {key}:")
        for i, ell_val in enumerate(ells):
            lines.append(f"    ell={ell_val:5.0f}: std = {data[key]['std_cross_ps'][i]:.6f}")

    lines.append("")
    lines.append("=" * 90)
    lines.append("TABLE 2: Ratio  rho_Model(l) / rho_B_S-only(l)  [what the V7 plot shows]")
    lines.append("=" * 90)
    ratio_keys = [k for k, _, _, _ in MODELS if k != "B_S-only"]
    header2 = "{:>6}".format("ell") + "".join("{:>12}".format(k) for k in ratio_keys)
    lines.append(header2)
    lines.append("-" * len(header2))
    for i, ell_val in enumerate(ells):
        bval = data["B_S-only"]["mean_cross_ps"][i]
        row = "{:6.0f}".format(ell_val)
        for key in ratio_keys:
            row += "{:12.6f}".format(data[key]["mean_cross_ps"][i] / bval)
        lines.append(row)

    lines.append("")
    lines.append("Ratio uncertainty from std (mean +/- std/b_only, plotted error bars):")
    for key, label, _, _ in MODELS:
        if key == "B_S-only":
            continue
        lines.append(f"  {key} ({label}):")
        for i, ell_val in enumerate(ells):
            bval = data["B_S-only"]["mean_cross_ps"][i]
            m = data[key]["mean_cross_ps"][i]
            s = data[key]["std_cross_ps"][i]
            ratio = m / bval
            err = s / bval
            lo = max(ratio - err, 0)
            hi = ratio + err
            lines.append(
                f"    ell={ell_val:5.0f}: {ratio:.6f}  (+/- {err:.6f})  -> [{lo:.6f}, {hi:.6f}]"
            )

    ratios = []
    ratio_lo = []
    ratio_hi = []
    for i, ell_val in enumerate(ells):
        bval = data["B_S-only"]["mean_cross_ps"][i]
        for key, _, _, _ in MODELS:
            if key == "B_S-only":
                continue
            m = data[key]["mean_cross_ps"][i]
            s = data[key]["std_cross_ps"][i]
            r = m / bval
            ratios.append(r)
            ratio_lo.append(max(r - s / bval, 0))
            ratio_hi.append(r + s / bval)

    lines.append("")
    lines.append("=" * 90)
    lines.append("SUMMARY for tick-mark selection")
    lines.append("=" * 90)
    lines.append(f"Ratio means:  min={min(ratios):.6f}, max={max(ratios):.6f}")
    lines.append(f"Ratio + err:  min lower={min(ratio_lo):.6f}, max upper={max(ratio_hi):.6f}")
    lines.append("B_S-only reference line: 1.000000 at all ell")

    text = "\n".join(lines)
    print(text)
    with open(OUTPUT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
