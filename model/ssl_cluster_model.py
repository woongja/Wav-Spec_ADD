"""
SSL-based Model with Cluster Metric Learning

This model extracts embeddings from SSL models (wav2vec2, HuBERT, XLS-R, etc.)
and applies cluster-based metric learning for Audio Deepfake Detection.
"""

import torch
import torch.nn as nn
import fairseq


class SSLModel(nn.Module):
    """
    Self-Supervised Learning Model Wrapper
    Supports: wav2vec2, HuBERT, XLS-R, etc.
    """

    def __init__(self, ssl_model_path, device):
        super(SSLModel, self).__init__()

        # Load pre-trained SSL model
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_model_path])
        self.model = model[0]
        self.device = device

        # Get output dimension from the model
        self.out_dim = 1024  # Default for XLS-R 300M

        # Freeze SSL model by default (can be unfrozen for fine-tuning)
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_feat(self, input_data):
        """
        Extract features from SSL model

        Args:
            input_data: Raw waveform [batch, length] or [batch, length, 1]

        Returns:
            embeddings: [batch, time_frames, dim]
        """
        # Put model to correct device
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device, dtype=input_data.dtype)

        # Ensure input is [batch, length]
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # Extract features
        # Returns dict with 'x' key containing embeddings [batch, time, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']

        return emb

    def unfreeze(self):
        """Unfreeze SSL model for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("[INFO] SSL model unfrozen for fine-tuning")

    def freeze(self):
        """Freeze SSL model"""
        for param in self.model.parameters():
            param.requires_grad = False
        print("[INFO] SSL model frozen")


class Model(nn.Module):
    """
    Complete model for Cluster-based Audio Deepfake Detection

    Architecture:
        1. SSL Model (wav2vec2/HuBERT/XLS-R) -> frame-level features
        2. Projection layer -> reduce to embedding_dim
        3. Temporal pooling -> utterance-level embedding
        4. Classification head -> bonafide/spoof prediction

    The utterance-level embedding is used for cluster-based metric learning.
    """

    def __init__(self, args, device):
        super(Model, self).__init__()

        self.device = device
        self.embedding_dim = args.emb_size

        # 1. SSL Feature Extractor
        ssl_model_path = getattr(args, 'ssl_model_path', '/home/woongjae/wildspoof/xlsr2_300m.pt')
        self.ssl_model = SSLModel(ssl_model_path, device)

        # 2. Projection layer: SSL dim -> embedding_dim
        self.projection = nn.Linear(self.ssl_model.out_dim, self.embedding_dim)

        # 3. Batch normalization
        self.bn = nn.BatchNorm1d(self.embedding_dim)

        # 4. Activation
        self.activation = nn.SELU(inplace=True)

        # 5. Temporal pooling options
        self.pooling_type = getattr(args, 'pooling_type', 'mean')  # 'mean', 'max', 'attention'

        if self.pooling_type == 'attention':
            # Learnable attention pooling
            self.attention = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.Tanh(),
                nn.Linear(self.embedding_dim // 2, 1)
            )

        # 6. Classification head
        self.classifier = nn.Linear(self.embedding_dim, 2)

        print(f'[INFO] SSL Cluster Model initialized')
        print(f'  - SSL model: {ssl_model_path}')
        print(f'  - Embedding dim: {self.embedding_dim}')
        print(f'  - Pooling type: {self.pooling_type}')

    def forward(self, x, return_embedding=True):
        """
        Forward pass

        Args:
            x: Raw waveform [batch, length] or [batch, length, 1]
            return_embedding: If True, return both logits and embeddings

        Returns:
            logits: [batch, 2] - classification logits
            embedding: [batch, embedding_dim] - utterance-level embedding (if return_embedding=True)
        """
        # 1. Extract SSL features
        # Shape: [batch, time, ssl_dim]
        ssl_features = self.ssl_model.extract_feat(x)

        # 2. Project to embedding space
        # Shape: [batch, time, embedding_dim]
        projected = self.projection(ssl_features)

        # 3. Temporal pooling to get utterance-level embedding
        if self.pooling_type == 'mean':
            # Mean pooling over time dimension
            embedding = projected.mean(dim=1)  # [batch, embedding_dim]

        elif self.pooling_type == 'max':
            # Max pooling over time dimension
            embedding = projected.max(dim=1)[0]  # [batch, embedding_dim]

        elif self.pooling_type == 'attention':
            # Attention-based pooling
            # Compute attention weights
            attn_weights = self.attention(projected)  # [batch, time, 1]
            attn_weights = torch.softmax(attn_weights, dim=1)

            # Weighted sum
            embedding = (projected * attn_weights).sum(dim=1)  # [batch, embedding_dim]

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # 4. Batch normalization and activation
        embedding = self.bn(embedding)
        embedding = self.activation(embedding)

        # 5. Classification
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        else:
            return logits, None

    def unfreeze_ssl(self):
        """Unfreeze SSL model for fine-tuning"""
        self.ssl_model.unfreeze()

    def freeze_ssl(self):
        """Freeze SSL model"""
        self.ssl_model.freeze()


class LightweightModel(nn.Module):
    """
    Lightweight version without heavy SSL model - for testing

    Uses simple convolutional feature extraction instead of pre-trained SSL.
    Useful for quick experiments and debugging.
    """

    def __init__(self, args, device):
        super(LightweightModel, self).__init__()

        self.device = device
        self.embedding_dim = args.emb_size

        # Simple CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(100)  # Fixed length output
        )

        # Projection to embedding space
        self.projection = nn.Linear(256, self.embedding_dim)
        self.bn = nn.BatchNorm1d(self.embedding_dim)
        self.activation = nn.SELU()

        # Classification head
        self.classifier = nn.Linear(self.embedding_dim, 2)

        print(f'[INFO] Lightweight Model initialized (for testing)')
        print(f'  - Embedding dim: {self.embedding_dim}')

    def forward(self, x, return_embedding=True):
        """Forward pass"""
        # x: [batch, length] or [batch, length, 1]

        if x.ndim == 3:
            x = x.squeeze(-1)  # [batch, length]

        x = x.unsqueeze(1)  # [batch, 1, length]

        # Extract features
        features = self.feature_extractor(x)  # [batch, 256, 100]

        # Global average pooling
        embedding = features.mean(dim=2)  # [batch, 256]

        # Project to embedding space
        embedding = self.projection(embedding)  # [batch, embedding_dim]
        embedding = self.bn(embedding)
        embedding = self.activation(embedding)

        # Classification
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        else:
            return logits, None
