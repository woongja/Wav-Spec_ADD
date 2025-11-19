#!/bin/bash

# Training script for pure cluster-based learning
# Uses only Bonafide-cluster metric loss

# Resource limits
# GPU: Use GPU 0
# CPU: Limit to 8 threads (adjust based on your system)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python main.py \
    --database_path /AISRC2/Dataset/ \
    --protocol_path protocols/protocol.txt \
    --config configs/cluster_baseline.yaml \
    --batch_size 32 \
    --num_epochs 100 \
    --max_lr 1e-4 \
    --weight_decay 1e-4 \
    --patience 10 \
    --loss_type cluster_only \
    --num_clusters 5 \
    --cluster_alpha 1.0 \
    --cluster_method kmeans \
    --cluster_update_freq 100 \
    --algo 3 \
    --rb_prob 0.5 \
    --comment cluster_baseline_k5 \
    --model_save_path out/cluster_baseline_k5.pth
