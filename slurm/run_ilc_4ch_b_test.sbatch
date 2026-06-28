#!/bin/bash
#SBATCH --job-name=ilc_4ch_b_test
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=ilc_4ch_b_test_%j.out

# 4-channel ILC (B-only @ 4 frequencies) on n_test=19200 test patches, cmb_scale=0.05.
# Comparison baseline for 12ch / 8ch variants. Outputs go to a separate folder.

set -euo pipefail

WORKDIR="/scratch/gpfs/JDUNKLEY/hshao/ILC_ML/toy_model/toy_model/DustFilaments/new_patches/new_patches"
cd "$WORKDIR"

TEST_NPZ="/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/test_results.npz"
OUTPUT_DIR="/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_4ch_b_test"
B_DIR="/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_B"
TE_DIR="/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_T_E"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Memory: 80G requested"

mkdir -p "$OUTPUT_DIR"

python compute_ilc_12ch_test_patches.py \
  --input-b-only \
  --test-results-npz "$TEST_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --b-dir "$B_DIR" \
  --te-dir "$TE_DIR" \
  --cmb-scale 0.05 \
  2>&1 | tee "${OUTPUT_DIR}/ilc_4ch_b_test.log"

echo "Job finished: $(date)"
