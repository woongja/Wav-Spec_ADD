"""
Cluster-based loss functions
Wrapper for compatibility with main.py training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BonafideClusterLoss(nn.Module):
    """
    Bonafide-only cluster metric learning
    - Bonafide samples clustered and pulled toward centers
    - Spoof samples pushed away from Bonafide manifold
    """
    def __init__(
        self,
        num_clusters=5,
        embedding_dim=512,
        alpha=1.0,
        cluster_update_freq=10,
        cluster_method='kmeans'
    ):
        super(BonafideClusterLoss, self).__init__()
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.cluster_update_freq = cluster_update_freq
        self.cluster_method = cluster_method

        # Initialize cluster centers
        self.register_buffer('cluster_centers', torch.randn(num_clusters, embedding_dim))
        self.register_buffer('update_counter', torch.tensor(0))

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, embedding_dim)
            labels: (B,) - 0 for fake, 1 for real
        Returns:
            loss: scalar
            components: dict with loss breakdown
        """
        bonafide_mask = labels == 1
        spoof_mask = labels == 0

        bonafide_emb = embeddings[bonafide_mask]
        spoof_emb = embeddings[spoof_mask]

        # Bonafide cluster loss (pull toward nearest center)
        if bonafide_emb.size(0) > 0:
            # Compute distances to all centers
            dist_to_centers = torch.cdist(bonafide_emb, self.cluster_centers)  # (N_bonafide, num_clusters)
            min_dist, _ = torch.min(dist_to_centers, dim=1)  # Distance to nearest center
            bonafide_loss = min_dist.mean()
        else:
            bonafide_loss = torch.tensor(0.0, device=embeddings.device)

        # Spoof repulsion loss (push away from all centers)
        if spoof_emb.size(0) > 0:
            dist_to_centers = torch.cdist(spoof_emb, self.cluster_centers)  # (N_spoof, num_clusters)
            min_dist, _ = torch.min(dist_to_centers, dim=1)
            # Negative distance = repulsion
            spoof_loss = -min_dist.mean()
        else:
            spoof_loss = torch.tensor(0.0, device=embeddings.device)

        # Total cluster loss
        cluster_loss = bonafide_loss + self.alpha * spoof_loss

        components = {
            'cluster_total': cluster_loss.item(),
            'cluster_bonafide': bonafide_loss.item(),
            'cluster_spoof': spoof_loss.item()
        }

        return cluster_loss, components


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy + Cluster Loss
    """
    def __init__(
        self,
        num_clusters=5,
        embedding_dim=512,
        alpha=1.0,
        ce_weight=1.0,
        cluster_weight=0.1,
        cluster_update_freq=10,
        cluster_method='kmeans'
    ):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.cluster_weight = cluster_weight

        # Cross-entropy loss
        weight = torch.FloatTensor([0.1, 0.9])
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

        # Cluster loss
        self.cluster_loss = BonafideClusterLoss(
            num_clusters=num_clusters,
            embedding_dim=embedding_dim,
            alpha=alpha,
            cluster_update_freq=cluster_update_freq,
            cluster_method=cluster_method
        )

    def forward(self, logits, embeddings, labels):
        """
        Args:
            logits: (B, num_classes)
            embeddings: (B, embedding_dim)
            labels: (B,) - 0 for fake, 1 for real
        Returns:
            loss: scalar
            components: dict with loss breakdown
        """
        # Cross-entropy loss
        ce = self.ce_loss(logits, labels)

        # Cluster loss
        cluster, cluster_components = self.cluster_loss(embeddings, labels)

        # Combined loss
        total_loss = self.ce_weight * ce + self.cluster_weight * cluster

        components = {
            'ce': ce.item(),
            **cluster_components
        }

        return total_loss, components
