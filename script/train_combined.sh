#!/bin/bash

# Training script for combined approach
# Uses both Cross-Entropy and Cluster metric loss

# Resource limits
# GPU: Use GPU 0
# CPU: Limit to 8 threads (adjust based on your system)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python main.py \
    --database_path /AISRC2/Dataset/ \
    --protocol_path protocols/protocol.txt \
    --config configs/cluster_combined.yaml \
    --batch_size 32 \
    --num_epochs 100 \
    --max_lr 1e-4 \
    --weight_decay 1e-4 \
    --patience 10 \
    --loss_type combined \
    --num_clusters 5 \
    --cluster_alpha 1.0 \
    --cluster_method kmeans \
    --cluster_update_freq 100 \
    --ce_weight 1.0 \
    --cluster_weight 0.5 \
    --algo 3 \
    --rb_prob 0.5 \
    --comment cluster_combined_k5 \
    --model_save_path out/cluster_combined_k5.pth
