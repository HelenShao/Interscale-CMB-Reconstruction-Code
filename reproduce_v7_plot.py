#!/usr/bin/env python3
"""Reproduce rho_philcox18_t_b_e_te_teb.png (Variant 7) without overwriting the original."""

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
    plot_philcox18_with_models,
)

OUTPUT_DIR = "/scratch/gpfs/JDUNKLEY/hshao/old_data/model_comparison"
OUTPUT_NAME = "rho_philcox18_t_b_e_te_teb_reproduced.png"

MODEL_CONFIGS = [
    {
        "base_dir": "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_cmb_scaled/normalized_b_only",
        "prefix": "normalized_b_only_best_model_153600_19200_norm",
        "label": LABEL_BS_ONLY,
        "color": "#2E86AB",
        "marker": "o",
    },
    {
        "base_dir": "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb",
        "prefix": "unnormalized_teb_best_model_153600_19200_nonorm",
        "label": LABEL_TEB,
        "color": "#F77F00",
        "marker": "s",
    },
    {
        "base_dir": "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_t_only",
        "prefix": "unnormalized_t_only_best_model_153600_19200_nonorm",
        "label": LABEL_T_ONLY,
        "color": "#E63946",
        "marker": "^",
    },
    {
        "base_dir": "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_e_only",
        "prefix": "unnormalized_e_only_best_model_153600_19200_nonorm",
        "label": LABEL_E_ONLY,
        "color": "#06A77D",
        "marker": "D",
    },
    {
        "base_dir": "/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_te_only",
        "prefix": "unnormalized_te_only_best_model_153600_19200_nonorm",
        "label": LABEL_TE,
        "color": "#000000",
        "marker": "v",
    },
]

V7_LABELS = [LABEL_T_ONLY, LABEL_BS_ONLY, LABEL_E_ONLY, LABEL_TE, LABEL_TEB]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models_data = []
    for model_config in MODEL_CONFIGS:
        cross_spec_file = os.path.join(
            model_config["base_dir"],
            f"{model_config['prefix']}_cross_spectrum_stats.txt",
        )
        print(f"Loading {model_config['label']}: {cross_spec_file}")
        cross_spec_data = parse_cross_spectrum_file(cross_spec_file)
        models_data.append(
            {
                "cross_spec_data": cross_spec_data,
                "label": model_config["label"],
                "color": model_config["color"],
                "marker": model_config["marker"],
                "data": cross_spec_data,
            }
        )

    cross_spec_models_variants = [
        {
            "data": m["cross_spec_data"],
            "label": m["label"],
            "color": m["color"],
            "marker": m["marker"],
        }
        for m in models_data
    ]

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    print(f"Creating V7 reproduction: {output_path}")
    plot_philcox18_with_models(
        cross_spec_models_variants,
        V7_LABELS,
        output_path,
        no_legend=False,
        no_shaded_regions=True,
        use_line_styles=True,
        ratio_plot=True,
        log_yscale=True,
        dense_y_ticks=True,
    )
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
