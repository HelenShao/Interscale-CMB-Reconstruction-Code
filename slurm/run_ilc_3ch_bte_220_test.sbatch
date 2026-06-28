#!/bin/bash
#SBATCH --job-name=ilc_3ch_bte_220
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=ilc_3ch_bte_220_%j.out

# 3-channel ILC: B + T + E @ 220 GHz only.

set -euo pipefail

WORKDIR="/scratch/gpfs/JDUNKLEY/hshao/ILC_ML/toy_model/toy_model/DustFilaments/new_patches/new_patches"
cd "$WORKDIR"

TEST_NPZ="/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/test_results.npz"
OUTPUT_DIR="/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_3ch_bte_220_test"
B_DIR="/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_B"
TE_DIR="/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_T_E"

echo "Job started: $(date)"
mkdir -p "$OUTPUT_DIR"

python compute_ilc_12ch_test_patches.py \
  --single-freq-bte-only \
  --single-freq 220 \
  --test-results-npz "$TEST_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --b-dir "$B_DIR" \
  --te-dir "$TE_DIR" \
  --cmb-scale 1.0 \
  2>&1 | tee "${OUTPUT_DIR}/ilc_3ch_bte_220_test.log"

echo "Job finished: $(date)"
