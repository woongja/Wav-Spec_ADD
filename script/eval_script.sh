#!/bin/bash

# Evaluation script
# Usage: bash eval_script.sh <model_path> <config_path> <output_file>

MODEL_PATH=${1:-"out/cluster_baseline_k5.pth"}
CONFIG=${2:-"configs/cluster_baseline.yaml"}
OUTPUT_FILE=${3:-"results/eval_scores.txt"}

# Resource limits
# GPU: Use GPU 0
# CPU: Limit to 8 threads (adjust based on your system)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python main.py \
    --database_path /AISRC2/Dataset/ \
    --protocol_path protocols/protocol.txt \
    --config $CONFIG \
    --batch_size 32 \
    --eval \
    --model_path $MODEL_PATH \
    --eval_output $OUTPUT_FILE

echo "Evaluation complete. Results saved to $OUTPUT_FILE"
