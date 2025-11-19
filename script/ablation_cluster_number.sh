#!/bin/bash

# Ablation study: Different number of clusters
# Tests K = 3, 5, 10, 15 to find optimal cluster number

# Resource limits
# GPU: Use GPU 0
# CPU: Limit to 8 threads

GPU_ID=0
NUM_THREADS=4

echo "Starting cluster number ablation study..."
echo "================================================"

# K = 3
echo "Training with K=3 clusters..."
CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=$NUM_THREADS python main.py \
    --database_path /AISRC2/Dataset/ \
    --protocol_path protocols/protocol.txt \
    --config configs/cluster_baseline.yaml \
    --batch_size 32 \
    --num_epochs 100 \
    --max_lr 1e-4 \
    --weight_decay 1e-4 \
    --patience 10 \
    --loss_type cluster_only \
    --num_clusters 3 \
    --cluster_alpha 1.0 \
    --cluster_method kmeans \
    --algo 3 \
    --rb_prob 0.5 \
    --comment cluster_k3 \
    --model_save_path out/cluster_k3.pth

echo "================================================"

# K = 5
echo "Training with K=5 clusters..."
CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=$NUM_THREADS python main.py \
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
    --algo 3 \
    --rb_prob 0.5 \
    --comment cluster_k5 \
    --model_save_path out/cluster_k5.pth

echo "================================================"

# K = 10
echo "Training with K=10 clusters..."
CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=$NUM_THREADS python main.py \
    --database_path /AISRC2/Dataset/ \
    --protocol_path protocols/protocol.txt \
    --config configs/cluster_baseline.yaml \
    --batch_size 32 \
    --num_epochs 100 \
    --max_lr 1e-4 \
    --weight_decay 1e-4 \
    --patience 10 \
    --loss_type cluster_only \
    --num_clusters 10 \
    --cluster_alpha 1.0 \
    --cluster_method kmeans \
    --algo 3 \
    --rb_prob 0.5 \
    --comment cluster_k10 \
    --model_save_path out/cluster_k10.pth

echo "================================================"

# K = 15
echo "Training with K=15 clusters..."
CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=$NUM_THREADS python main.py \
    --database_path /AISRC2/Dataset/ \
    --protocol_path protocols/protocol.txt \
    --config configs/cluster_baseline.yaml \
    --batch_size 32 \
    --num_epochs 100 \
    --max_lr 1e-4 \
    --weight_decay 1e-4 \
    --patience 10 \
    --loss_type cluster_only \
    --num_clusters 15 \
    --cluster_alpha 1.0 \
    --cluster_method kmeans \
    --algo 3 \
    --rb_prob 0.5 \
    --comment cluster_k15 \
    --model_save_path out/cluster_k15.pth

echo "================================================"
echo "Ablation study complete!"
echo "Check logs/ directory for results"
echo "Compare: logs/cluster_k3, logs/cluster_k5, logs/cluster_k10, logs/cluster_k15"
