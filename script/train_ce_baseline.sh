#!/bin/bash

# Training script for CE baseline
# Traditional cross-entropy loss for comparison

# Resource limits
# GPU: Use GPU 0
# CPU: Limit to 8 threads (adjust based on your system)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python main.py \
    --database_path /AISRC2/Dataset/ \
    --protocol_path protocols/protocol.txt \
    --config configs/ce_baseline.yaml \
    --batch_size 32 \
    --num_epochs 100 \
    --max_lr 1e-4 \
    --weight_decay 1e-4 \
    --patience 10 \
    --loss_type ce_only \
    --algo 3 \
    --rb_prob 0.5 \
    --comment ce_baseline \
    --model_save_path out/ce_baseline.pth
