#!/bin/bash
#SBATCH --job-name=ilc_12ch_test
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=ilc_12ch_test_%j.out

# 12-channel ILC on n_test=19200 test patches, cmb_scale=0.05 (~37 GiB peak RAM).
# Set ILC_ML_DATA_ROOT to your data tree (see paths_config.py).

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

DATA_ROOT="${ILC_ML_DATA_ROOT:-/scratch/gpfs/JDUNKLEY/hshao/old_data}"
TEST_NPZ="${ILC_ML_TEST_NPZ:-${DATA_ROOT}/singlefreq_unet_1024_0.05_scaled/unnormalized_teb/test_results.npz}"
OUTPUT_DIR="${ILC_ML_ILC_12CH_OUT:-${DATA_ROOT}/ILC_products_12ch_teb_test}"
B_DIR="${ILC_ML_B_DIR:-${DATA_ROOT}/new_DF_patches_B}"
TE_DIR="${ILC_ML_TE_DIR:-${DATA_ROOT}/new_DF_patches_T_E}"

mkdir -p "$OUTPUT_DIR"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "WORKDIR=$WORKDIR DATA_ROOT=$DATA_ROOT"

python compute_ilc_12ch_test_patches.py \
  --test-results-npz "$TEST_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --b-dir "$B_DIR" \
  --te-dir "$TE_DIR" \
  --cmb-scale 0.05 \
  2>&1 | tee "${OUTPUT_DIR}/ilc_12ch_test.log"

echo "Job finished: $(date)"
