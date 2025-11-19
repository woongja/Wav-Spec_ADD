# Cluster-based Learning for Audio Deepfake Detection

This project implements a novel **Bonafide-only Cluster Metric Loss** for Audio Deepfake Detection using SSL embeddings.

## Overview

### Key Idea

Unlike traditional CE/SCL approaches, this method reconstructs the SSL embedding space using cluster-based metric learning:

1. **Bonafide Clustering**: Extract SSL embeddings for bonafide samples and cluster them into K clusters
2. **Attraction Loss**: Pull bonafide samples toward their nearest cluster center
3. **Repulsion Loss**: Push spoof samples away from all bonafide cluster centers
4. **No Spoof Clusters**: Spoofs don't form their own clusters - they simply repel from the bonafide manifold

This creates a stable, well-structured bonafide manifold while keeping spoofs at a distance.

## Project Structure

```
Cluster-based_learning/
├── configs/                    # Configuration files
│   ├── cluster_baseline.yaml   # Pure cluster loss
│   ├── cluster_combined.yaml   # CE + Cluster loss
│   └── ce_baseline.yaml        # CE only (for comparison)
├── model/                      # Model implementations
│   ├── ssl_cluster_model.py    # SSL-based model with clustering
│   ├── cluster_loss.py         # Cluster metric loss functions
│   └── __init__.py
├── datautils/                  # Data loading utilities
│   ├── data_utils.py           # Dataset classes
│   ├── RawBoost.py             # Audio augmentation
│   └── __init__.py
├── core_scripts/               # Core utilities
├── script/                     # Training/evaluation scripts
├── main.py                     # Main training/eval script
└── README.md                   # This file
```

## Usage

### Training

```bash
# Pure cluster-based learning
bash script/train_cluster_baseline.sh

# Combined approach (CE + Cluster)
bash script/train_combined.sh

# CE baseline (for comparison)
bash script/train_ce_baseline.sh
```

### Evaluation

```bash
bash script/eval_script.sh out/cluster_baseline.pth configs/cluster_baseline.yaml results/eval_scores.txt
```

## Configuration Options

### Loss Configuration

- `--loss_type`: `cluster_only`, `combined`, or `ce_only`
- `--num_clusters`: Number of bonafide clusters (default: 10)
- `--cluster_alpha`: Weight for spoof repulsion (default: 1.0)
- `--cluster_method`: Clustering method - `kmeans` or `online`

## Experiment Ideas

1. **Cluster Number Ablation**: Try K = 5, 10, 15, 20
2. **Alpha Tuning**: Try α = 0.5, 1.0, 2.0, 5.0
3. **Pooling Strategies**: mean, max, attention
4. **Combined vs Pure**: Compare cluster_only vs combined loss

