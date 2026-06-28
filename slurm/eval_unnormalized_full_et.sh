#!/bin/bash
# Evaluate the trained model: unnormalized_full_et

MODEL_DIR="/scratch/gpfs/JDUNKLEY/hshao/old_data/ILC_interscale_unet_1024/unnormalized_full_et"
MODEL_PATH="${MODEL_DIR}/best_model_153600_19200_nonorm.pt"
OUTPUT_DIR="${MODEL_DIR}/eval_results"

cd /scratch/gpfs/hshao/ILC_ML/toy_model/DustFilaments/new_patches/new_patches

python eval_test_results.py \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --add-e-t

