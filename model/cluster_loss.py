"""
Bonafide-only Cluster Metric Loss

This module implements a novel metric learning approach for Audio Deepfake Detection:
1. Bonafide samples are clustered and pulled toward cluster centers
2. Spoof samples are pushed away from all Bonafide cluster centers
3. No separate clusters are created for spoofs - they simply repel from Bonafide manifold

This creates a stable Bonafide manifold while keeping spoofs at a distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


class BonafideClusterLoss(nn.Module):
    """
    Bonafide-only Cluster Metric Loss

    Args:
        num_clusters (int): Number of Bonafide clusters (K)
        embedding_dim (int): Dimension of SSL embeddings
        alpha (float): Weight for spoof repulsion loss
        cluster_update_freq (int): How often to update cluster centers (in iterations)
        cluster_method (str): Clustering method ('kmeans' or 'online')
    """

    def __init__(self, num_clusters=10, embedding_dim=256, alpha=1.0,
                 cluster_update_freq=100, cluster_method='kmeans'):
        super(BonafideClusterLoss, self).__init__()

        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.cluster_update_freq = cluster_update_freq
        self.cluster_method = cluster_method
        self.iteration = 0

        # Initialize Bonafide cluster centers
        # Shape: [num_clusters, embedding_dim]
        self.register_buffer('bonafide_centers', torch.randn(num_clusters, embedding_dim))
        self.register_buffer('centers_initialized', torch.tensor(False))

        # For online clustering: track cluster assignments and counts
        self.register_buffer('cluster_counts', torch.zeros(num_clusters))

    def initialize_centers(self, bonafide_embeddings):
        """
        Initialize cluster centers using K-means on Bonafide data

        Args:
            bonafide_embeddings: Tensor of shape [N, embedding_dim]
        """
        if bonafide_embeddings.size(0) < self.num_clusters:
            print(f"[WARNING] Not enough bonafide samples ({bonafide_embeddings.size(0)}) for {self.num_clusters} clusters")
            return

        # Run K-means on CPU for stability
        embeddings_np = bonafide_embeddings.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings_np)

        # Update centers
        self.bonafide_centers.copy_(torch.from_numpy(kmeans.cluster_centers_).to(bonafide_embeddings.device))
        self.centers_initialized.fill_(True)

        print(f"[INFO] Initialized {self.num_clusters} Bonafide cluster centers")

    def find_closest_center(self, embeddings):
        """
        Find the closest Bonafide cluster center for each embedding

        Args:
            embeddings: Tensor of shape [N, embedding_dim]

        Returns:
            distances: Tensor of shape [N] - distance to closest center
            indices: Tensor of shape [N] - index of closest center
        """
        # Compute pairwise distances: [N, num_clusters]
        distances = torch.cdist(embeddings, self.bonafide_centers, p=2)

        # Find minimum distance and corresponding center index
        min_distances, min_indices = torch.min(distances, dim=1)

        return min_distances, min_indices

    def update_centers_online(self, bonafide_embeddings, cluster_indices):
        """
        Update cluster centers using online/incremental approach

        Args:
            bonafide_embeddings: Tensor of shape [N_bonafide, embedding_dim]
            cluster_indices: Tensor of shape [N_bonafide] - assigned cluster for each sample
        """
        for k in range(self.num_clusters):
            # Find all embeddings assigned to cluster k
            mask = (cluster_indices == k)

            if mask.sum() > 0:
                cluster_samples = bonafide_embeddings[mask]

                # Update center using moving average
                # new_center = (old_count * old_center + new_samples) / (old_count + new_count)
                old_count = self.cluster_counts[k]
                new_count = cluster_samples.size(0)

                if old_count > 0:
                    self.bonafide_centers[k] = (
                        old_count * self.bonafide_centers[k] + cluster_samples.sum(dim=0)
                    ) / (old_count + new_count)
                else:
                    self.bonafide_centers[k] = cluster_samples.mean(dim=0)

                self.cluster_counts[k] += new_count

    def forward(self, embeddings, labels):
        """
        Compute Bonafide-only cluster metric loss

        Args:
            embeddings: Tensor of shape [batch_size, embedding_dim]
            labels: Tensor of shape [batch_size] - 0 for bonafide, 1 for spoof

        Returns:
            loss: Scalar tensor
            loss_dict: Dictionary with loss components for logging
        """
        # Normalize embeddings for better clustering
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Separate bonafide and spoof samples
        bonafide_mask = (labels == 0)
        spoof_mask = (labels == 1)

        bonafide_embeddings = embeddings[bonafide_mask]
        spoof_embeddings = embeddings[spoof_mask]

        # Initialize centers if not done yet
        if not self.centers_initialized and bonafide_embeddings.size(0) >= self.num_clusters:
            self.initialize_centers(bonafide_embeddings)

        # If centers not initialized yet, return zero loss
        if not self.centers_initialized:
            return torch.tensor(0.0, device=embeddings.device), {
                'total_loss': 0.0,
                'bonafide_loss': 0.0,
                'spoof_loss': 0.0
            }

        # Normalize centers
        self.bonafide_centers.data = F.normalize(self.bonafide_centers, p=2, dim=1)

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=embeddings.device)

        # 1. Bonafide attraction loss: pull toward closest center
        if bonafide_embeddings.size(0) > 0:
            bonafide_distances, bonafide_indices = self.find_closest_center(bonafide_embeddings)
            bonafide_loss = bonafide_distances.pow(2).mean()

            total_loss = total_loss + bonafide_loss
            loss_dict['bonafide_loss'] = bonafide_loss.item()

            # Update centers periodically
            if self.cluster_method == 'online' and self.iteration % self.cluster_update_freq == 0:
                self.update_centers_online(bonafide_embeddings, bonafide_indices)
        else:
            loss_dict['bonafide_loss'] = 0.0

        # 2. Spoof repulsion loss: push away from closest center
        if spoof_embeddings.size(0) > 0:
            spoof_distances, _ = self.find_closest_center(spoof_embeddings)
            # Negative distance to push away (maximize distance)
            spoof_loss = -self.alpha * spoof_distances.pow(2).mean()

            total_loss = total_loss + spoof_loss
            loss_dict['spoof_loss'] = spoof_loss.item()
        else:
            loss_dict['spoof_loss'] = 0.0

        loss_dict['total_loss'] = total_loss.item()

        self.iteration += 1

        return total_loss, loss_dict

    def predict(self, embeddings):
        """
        Predict whether samples are bonafide or spoof based on distance to clusters

        Args:
            embeddings: Tensor of shape [N, embedding_dim]

        Returns:
            predictions: Tensor of shape [N] - 0 for bonafide, 1 for spoof
            distances: Tensor of shape [N] - distance to closest center
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if not self.centers_initialized:
            # Return all as spoof if centers not initialized
            return torch.ones(embeddings.size(0), device=embeddings.device), torch.zeros(embeddings.size(0), device=embeddings.device)

        distances, _ = self.find_closest_center(embeddings)

        # Simple threshold-based prediction
        # Closer to centers -> Bonafide (0), Far from centers -> Spoof (1)
        # Threshold can be learned or set empirically
        threshold = 0.5  # This should be tuned on validation set
        predictions = (distances > threshold).long()

        return predictions, distances


class CombinedLoss(nn.Module):
    """
    Combined loss: Cross-Entropy + Bonafide Cluster Metric Loss

    This allows for a smooth transition or hybrid approach combining
    traditional classification with cluster-based metric learning.
    """

    def __init__(self, num_clusters=10, embedding_dim=256, alpha=1.0,
                 ce_weight=1.0, cluster_weight=1.0, **kwargs):
        super(CombinedLoss, self).__init__()

        self.ce_weight = ce_weight
        self.cluster_weight = cluster_weight

        # Cross-entropy loss with class weights (bonafide=0.1, spoof=0.9)
        weight = torch.FloatTensor([0.1, 0.9])
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

        # Cluster-based metric loss
        self.cluster_loss = BonafideClusterLoss(
            num_clusters=num_clusters,
            embedding_dim=embedding_dim,
            alpha=alpha,
            **kwargs
        )

    def forward(self, logits, embeddings, labels):
        """
        Compute combined loss

        Args:
            logits: Tensor of shape [batch_size, 2] - classification logits
            embeddings: Tensor of shape [batch_size, embedding_dim] - SSL embeddings
            labels: Tensor of shape [batch_size] - 0 for bonafide, 1 for spoof

        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with all loss components
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits, labels)

        # Cluster metric loss
        cluster_loss, cluster_loss_dict = self.cluster_loss(embeddings, labels)

        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.cluster_weight * cluster_loss

        # Build complete loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'cluster_total': cluster_loss_dict['total_loss'],
            'cluster_bonafide': cluster_loss_dict['bonafide_loss'],
            'cluster_spoof': cluster_loss_dict['spoof_loss']
        }

        return total_loss, loss_dict
