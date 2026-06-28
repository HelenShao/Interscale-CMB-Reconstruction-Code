#!/bin/bash
#SBATCH --job-name=ilc_220ghz_all
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=ilc_220ghz_all_%j.out

# Single-frequency (220 GHz) ILC variants: B, B+T, B+E, B+T+E — no 95/145/270.
# Set CMB_SCALE to match your multi-freq comparison runs (1.0 for paper-style; 0.05 for UNet training scale).

set -euo pipefail

WORKDIR="/scratch/gpfs/JDUNKLEY/hshao/ILC_ML/toy_model/toy_model/DustFilaments/new_patches/new_patches"
cd "$WORKDIR"

TEST_NPZ="/scratch/gpfs/JDUNKLEY/hshao/old_data/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/test_results.npz"
B_DIR="/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_B"
TE_DIR="/scratch/gpfs/JDUNKLEY/hshao/old_data/new_DF_patches_T_E"
CMB_SCALE=1.0
FREQ=220

run_variant () {
  local FLAG="$1"
  local OUT="$2"
  local LOG="$3"
  echo ""
  echo "========== ${FLAG} -> ${OUT} =========="
  mkdir -p "$OUT"
  python compute_ilc_12ch_test_patches.py \
    "$FLAG" \
    --single-freq "$FREQ" \
    --test-results-npz "$TEST_NPZ" \
    --output-dir "$OUT" \
    --b-dir "$B_DIR" \
    --te-dir "$TE_DIR" \
    --cmb-scale "$CMB_SCALE" \
    2>&1 | tee "$LOG"
}

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "single-freq=${FREQ} GHz, cmb_scale=${CMB_SCALE}"

run_variant --single-freq-b-only \
  "/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_1ch_b_220_test" \
  "/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_1ch_b_220_test/ilc_1ch_b_220_test.log"

run_variant --single-freq-bt-only \
  "/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_2ch_bt_220_test" \
  "/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_2ch_bt_220_test/ilc_2ch_bt_220_test.log"

run_variant --single-freq-be-only \
  "/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_2ch_be_220_test" \
  "/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_2ch_be_220_test/ilc_2ch_be_220_test.log"

run_variant --single-freq-bte-only \
  "/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_3ch_bte_220_test" \
  "/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_products_3ch_bte_220_test/ilc_3ch_bte_220_test.log"

echo "Job finished: $(date)"
