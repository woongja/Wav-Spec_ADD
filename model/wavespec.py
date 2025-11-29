"""
WaveSpec: Wave-Spectrogram Cross-Modal Aggregation for Audio Deepfake Detection
with PESQ-based Attention Modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
import torchaudio
from nnAudio.Spectrogram import CQT

# ============================================================================
# MS-CAM (Multi-Scale Channel Attention Module)
# ============================================================================
class MSCAM(nn.Module):
    """
    Multi-Scale Channel Attention Module
    Combines global and local context for attention generation
    """
    def __init__(self, dim, reduction=16):
        super(MSCAM, self).__init__()
        self.dim = dim

        # Global branch: GAP -> FC -> BN -> ReLU -> FC -> Sigmoid
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, D, T) -> (B, D, 1)
            nn.Conv1d(dim, dim // reduction, 1, bias=False),
            nn.BatchNorm1d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // reduction, dim, 1, bias=False),
        )

        # Local branch: Conv1D -> BN -> ReLU -> Conv1D -> Sigmoid
        self.local_branch = nn.Sequential(
            nn.Conv1d(dim, dim // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // reduction, dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            attention: (B, T, D) with values in [0, 1]
        """
        # Transpose for conv1d: (B, T, D) -> (B, D, T)
        x_t = x.transpose(1, 2)

        # Global attention
        global_att = self.global_branch(x_t)  # (B, D, 1)

        # Local attention
        local_att = self.local_branch(x_t)    # (B, D, T)

        # Combine and apply sigmoid
        attention = torch.sigmoid(global_att + local_att)  # (B, D, T)

        # Transpose back: (B, D, T) -> (B, T, D)
        attention = attention.transpose(1, 2)

        return attention


# ============================================================================
# AFF (Attentional Feature Fusion) with PESQ-based Modulation
# ============================================================================
class AFF(nn.Module):
    """
    Attentional Feature Fusion
    Fuses two feature maps using MS-CAM generated attention
    PESQ score modulates the attention strength
    """
    def __init__(self, dim, reduction=16, pesq_min=-0.5, pesq_max=4.5):
        super(AFF, self).__init__()
        self.mscam = MSCAM(dim, reduction)

        # PESQ normalization range
        self.pesq_min = pesq_min
        self.pesq_max = pesq_max

        # PESQ modulation parameters (learnable)
        self.pesq_alpha = nn.Parameter(torch.tensor(1.0))
        self.pesq_beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, y, pesq=None, use_pesq=True, debug=False):
        """
        Args:
            x: (B, T, D) - audio feature
            y: (B, T, D) - spectrogram feature
            pesq: (B,) or (B, 1) - PESQ scores (optional)
            use_pesq: bool - whether to use PESQ modulation
            debug: bool - whether to print debug information
        Returns:
            fused: (B, T, D)
            attention_info: dict with attention statistics (if debug=True)
        """
        # Element-wise sum
        z = x + y

        # Generate base attention map
        attention = self.mscam(z)  # (B, T, D)

        attention_info = {}

        if debug:
            attention_info['base_attention_mean'] = attention.mean().item()
            attention_info['base_attention_std'] = attention.std().item()
            attention_info['base_attention_min'] = attention.min().item()
            attention_info['base_attention_max'] = attention.max().item()

        # PESQ-based modulation
        if use_pesq and pesq is not None:
            # Ensure pesq has shape (B, 1, 1)
            if pesq.dim() == 1:
                pesq = pesq.unsqueeze(-1).unsqueeze(-1)  # (B,) -> (B, 1, 1)
            elif pesq.dim() == 2:
                pesq = pesq.unsqueeze(-1)  # (B, 1) -> (B, 1, 1)

            # Normalize PESQ to [0, 1] range
            pesq_normalized = (pesq - self.pesq_min) / (self.pesq_max - self.pesq_min)
            pesq_normalized = torch.clamp(pesq_normalized, 0.0, 1.0)  # Ensure within [0, 1]

            if debug:
                attention_info['pesq_raw_mean'] = pesq.mean().item()
                attention_info['pesq_normalized_mean'] = pesq_normalized.mean().item()

            # Q = sigmoid(alpha * PESQ_normalized + beta)
            Q = torch.sigmoid(self.pesq_alpha * pesq_normalized + self.pesq_beta)  # (B, 1, 1)

            if debug:
                attention_info['pesq_modulation_Q_mean'] = Q.mean().item()
                attention_info['pesq_alpha'] = self.pesq_alpha.item()
                attention_info['pesq_beta'] = self.pesq_beta.item()

            # Modulate attention
            attention = attention * Q  # (B, T, D) * (B, 1, 1) -> (B, T, D)

            if debug:
                attention_info['modulated_attention_mean'] = attention.mean().item()
                attention_info['modulated_attention_std'] = attention.std().item()

        # Weighted fusion
        fused = attention * x + (1 - attention) * y

        if debug:
            return fused, attention_info
        return fused


# ============================================================================
# CMFF (Cross-Modal Feature Fusion)
# ============================================================================
class CMFF(nn.Module):
    """
    Cross-Modal Feature Fusion Module
    Fuses audio and spectrogram features at a specific scale
    Optionally incorporates previous fusion result for hierarchical fusion
    """
    def __init__(self, audio_dim, spec_dim, output_dim, reduction=16, use_prev_fusion=False):
        super(CMFF, self).__init__()
        self.use_prev_fusion = use_prev_fusion

        # Dimension alignment
        self.audio_proj = nn.Conv1d(audio_dim, output_dim, 1) if audio_dim != output_dim else nn.Identity()
        self.spec_proj = nn.Conv1d(spec_dim, output_dim, 1) if spec_dim != output_dim else nn.Identity()

        if use_prev_fusion:
            self.prev_proj = nn.Conv1d(output_dim, output_dim, 1)

        # AFF module
        self.aff = AFF(output_dim, reduction)

    def forward(self, audio_feat, spec_feat, pesq=None, prev_fusion=None, use_pesq=True, debug=False):
        """
        Args:
            audio_feat: (B, T, audio_dim)
            spec_feat: (B, T, spec_dim)
            pesq: (B,) or (B, 1) - PESQ scores
            prev_fusion: (B, T, output_dim) - previous fusion result
            use_pesq: bool - whether to use PESQ modulation
            debug: bool - whether to print debug information
        Returns:
            fused: (B, T, output_dim)
            debug_info: dict with dimension and attention info (if debug=True)
        """
        debug_info = {}

        # Transpose for conv1d
        audio_t = audio_feat.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        spec_t = spec_feat.transpose(1, 2)

        if debug:
            debug_info['input_audio_shape'] = tuple(audio_feat.shape)
            debug_info['input_spec_shape'] = tuple(spec_feat.shape)

        # Project to same dimension
        audio_aligned = self.audio_proj(audio_t).transpose(1, 2)  # (B, T, output_dim)
        spec_aligned = self.spec_proj(spec_t).transpose(1, 2)

        if debug:
            debug_info['aligned_audio_shape'] = tuple(audio_aligned.shape)
            debug_info['aligned_spec_shape'] = tuple(spec_aligned.shape)

        # Add previous fusion if hierarchical
        if self.use_prev_fusion and prev_fusion is not None:
            prev_t = prev_fusion.transpose(1, 2)
            prev_aligned = self.prev_proj(prev_t).transpose(1, 2)
            spec_aligned = spec_aligned + prev_aligned

            if debug:
                debug_info['prev_fusion_shape'] = tuple(prev_fusion.shape)
                debug_info['spec_with_prev_shape'] = tuple(spec_aligned.shape)

        # Apply AFF with PESQ modulation
        if debug:
            fused, attention_info = self.aff(audio_aligned, spec_aligned, pesq, use_pesq=use_pesq, debug=True)
            debug_info['attention_info'] = attention_info
        else:
            fused = self.aff(audio_aligned, spec_aligned, pesq, use_pesq=use_pesq, debug=False)

        if debug:
            debug_info['output_fused_shape'] = tuple(fused.shape)
            return fused, debug_info

        return fused


# ============================================================================
# XLSR Encoder Wrapper
# ============================================================================
class WaveEncoderWrapper(nn.Module):
    """
    Wraps pretrained XLSR model and extracts features from layers [8, 16, 24]
    """
    def __init__(self, model_path='/home/woongjae/wildspoof/xlsr2_300m.pt', freeze=True, device='cuda'):
        super(WaveEncoderWrapper, self).__init__()
        # Load XLSR model using fairseq
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024  # XLSR output dimension
        self.feature_layers = [8, 16, 24]

        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze XLSR parameters"""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze XLSR parameters"""
        for param in self.model.parameters():
            param.requires_grad = True

    def extract_layer_features(self, waveform, layer_idx):
        """
        Extract features from a specific layer
        Args:
            waveform: (B, T_raw) - raw audio waveform
            layer_idx: int - layer index to extract
        Returns:
            features: (B, T, 1024)
        """
        # Ensure model is on correct device
        if next(self.model.parameters()).device != waveform.device:
            self.model.to(waveform.device, dtype=waveform.dtype)
            self.model.train()

        # Extract features from specific layer
        # XLSR returns features in shape (B, T, 1024)
        with torch.set_grad_enabled(self.training):
            result = self.model.extract_features(
                source=waveform,
                padding_mask=None,
                mask=False,
                layer=layer_idx
            )
            features = result['x']  # (B, T, 1024)

        return features

    def forward(self, waveform):
        """
        Args:
            waveform: (B, T_raw) - raw audio waveform
        Returns:
            features: dict with keys ['shallow', 'mid', 'deep']
                      each value: (B, T, 1024)
        """
        # Ensure model is on correct device
        if next(self.model.parameters()).device != waveform.device:
            self.model.to(waveform.device, dtype=waveform.dtype)
            self.model.train()

        # Extract features from specific layers
        features = {}

        # Layer 8 (shallow)
        result_shallow = self.model.extract_features(
            source=waveform,
            padding_mask=None,
            mask=False,
            layer=self.feature_layers[0]
        )
        features['shallow'] = result_shallow['x']  # (B, T, 1024)

        # Layer 16 (mid)
        result_mid = self.model.extract_features(
            source=waveform,
            padding_mask=None,
            mask=False,
            layer=self.feature_layers[1]
        )
        features['mid'] = result_mid['x']  # (B, T, 1024)

        # Layer 24 (deep)
        result_deep = self.model.extract_features(
            source=waveform,
            padding_mask=None,
            mask=False,
            layer=self.feature_layers[2]
        )
        features['deep'] = result_deep['x']  # (B, T, 1024)

        return features


# ============================================================================
# Spectrogram U-Net Encoder
# ============================================================================
class SpectrogramEncoder(nn.Module):
    """
    Lightweight U-Net style encoder for CQT spectrogram
    Outputs 3 scales: shallow, mid, deep
    """
    def __init__(self, hop_length=320, n_bins=84, bins_per_octave=12, output_dim=768):
        super(SpectrogramEncoder, self).__init__()

        # CQT transform
        self.cqt = CQT(
            sr=16000,
            hop_length=hop_length,
            fmin=32.7,
            n_bins=n_bins,              # ex) 84
            bins_per_octave=bins_per_octave,  # ex) 12
            filter_scale=1,
            pad_mode='reflect',
            trainable=False,
            output_format='Magnitude'
        )

        # U-Net style encoder
        # Shallow (high resolution)
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Mid (medium resolution)
        self.mid_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Deep (low resolution)
        self.deep_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Projection layers to match XLSR dimension (1024)
        # Use mean pooling over frequency axis: (B, C, T, F) -> (B, C, T)
        # Then project to output_dim with Conv1d
        self.shallow_proj = nn.Conv1d(64, output_dim, 1)  # (B, 64, T) -> (B, 1024, T)
        self.mid_proj = nn.Conv1d(128, output_dim, 1)      # (B, 128, T) -> (B, 1024, T)
        self.deep_proj = nn.Conv1d(256, output_dim, 1)     # (B, 256, T) -> (B, 1024, T)

    def forward(self, waveform, target_length=None):
        """
        Args:
            waveform: (B, T_raw) - raw audio waveform
            target_length: target temporal length to match XLSR
        Returns:
            features: dict with keys ['shallow', 'mid', 'deep']
                      each value: (B, T, 1024)
        """
        # Compute CQT spectrogram
        spec = self.cqt(waveform)  # (B, n_bins, T) - complex tensor

        # Convert complex to magnitude spectrogram
        # spec = torch.abs(spec)  # (B, n_bins, T)

        # Log scale
        spec = torch.log(spec + 1e-9)   # log scale
        spec = spec.unsqueeze(1)        # (B, 1, n_bins, T)

        # Encoder forward
        shallow_feat = self.shallow_conv(spec)  # (B, 64, n_bins, T)
        mid_feat = self.mid_conv(shallow_feat)  # (B, 128, n_bins//2, T//2)
        deep_feat = self.deep_conv(mid_feat)    # (B, 256, n_bins//4, T//4)

        # Mean pooling over frequency axis (mel bins)
        # (B, C, F, T) -> (B, C, T)
        shallow_pooled = torch.mean(shallow_feat, dim=2)  # (B, 64, T)
        mid_pooled = torch.mean(mid_feat, dim=2)          # (B, 128, T)
        deep_pooled = torch.mean(deep_feat, dim=2)        # (B, 256, T)

        # Project to output_dim with Conv1d
        shallow_out = self.shallow_proj(shallow_pooled).transpose(1, 2)  # (B, T, 1024)
        mid_out = self.mid_proj(mid_pooled).transpose(1, 2)              # (B, T, 1024)
        deep_out = self.deep_proj(deep_pooled).transpose(1, 2)           # (B, T, 1024)

        # Interpolate to target length if specified
        if target_length is not None:
            shallow_out = F.interpolate(shallow_out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
            mid_out = F.interpolate(mid_out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
            deep_out = F.interpolate(deep_out.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)

        features = {
            'shallow': shallow_out,
            'mid': mid_out,
            'deep': deep_out
        }

        return features


# ============================================================================
# ResNet18-based 1D Classifier
# ============================================================================
class ResNet1DBlock(nn.Module):
    """1D Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18Classifier(nn.Module):
    """
    ResNet18-based 1D classifier for temporal features
    """
    def __init__(self, input_dim=768, num_classes=2):
        super(ResNet18Classifier, self).__init__()

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNet1DBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNet1DBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, return_embedding=False):
        """
        Args:
            x: (B, T, D) - fused feature
            return_embedding: if True, return embedding before fc
        Returns:
            logits: (B, num_classes)
            embedding: (B, 512) if return_embedding=True
        """
        # Transpose for conv1d: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)

        x = self.input_proj(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # (B, 512, 1)
        embedding = x.squeeze(-1)  # (B, 512)

        logits = self.fc(embedding)

        if return_embedding:
            return logits, embedding
        return logits


# ============================================================================
# Single Center Loss
# ============================================================================
class SingleCenterLoss(nn.Module):
    """
    Single Center Loss (SCL)
    Pulls real samples toward center, pushes fake samples away
    """
    def __init__(self, feature_dim=512):
        super(SingleCenterLoss, self).__init__()
        self.center = nn.Parameter(torch.randn(feature_dim))

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, feature_dim)
            labels: (B,) - 0 for fake, 1 for real
        Returns:
            loss: scalar
        """
        # Distance to center
        dist = torch.sum((embeddings - self.center.unsqueeze(0)) ** 2, dim=1)  # (B,)

        # Real samples: minimize distance
        # Fake samples: maximize distance (minimize negative distance)
        real_mask = labels == 1
        fake_mask = labels == 0

        loss_real = dist[real_mask].mean() if real_mask.any() else 0.0
        loss_fake = -dist[fake_mask].mean() if fake_mask.any() else 0.0

        loss = loss_real + loss_fake

        return loss


# ============================================================================
# Main WaveSpec Model
# ============================================================================
class WaveSpec(nn.Module):
    """
    WaveSpec: Wave-Spectrogram Cross-Modal Aggregation
    with PESQ-based Attention Modulation
    """
    def __init__(
        self,
        xlsr_model_path='/home/woongjae/wildspoof/xlsr2_300m.pt',
        freeze_xlsr=True,
        output_dim=1024,  # XLSR output dimension is 1024
        num_classes=2,
        use_scl=True,
        use_pesq=True,  # Whether to use PESQ modulation
        device='cuda'
    ):
        super(WaveSpec, self).__init__()

        self.use_scl = use_scl
        self.use_pesq = use_pesq
        self.device = device

        # Encoders
        self.wave_encoder = WaveEncoderWrapper(xlsr_model_path, freeze=freeze_xlsr, device=device)
        self.spec_encoder = SpectrogramEncoder(output_dim=output_dim)

        # CMFF modules (3 scales)
        # XLSR output is 1024, so audio_dim should be 1024
        self.cmff_shallow = CMFF(
            audio_dim=1024,  # XLSR dimension
            spec_dim=output_dim,
            output_dim=output_dim,
            use_prev_fusion=False
        )

        self.cmff_mid = CMFF(
            audio_dim=1024,  # XLSR dimension
            spec_dim=output_dim,
            output_dim=output_dim,
            use_prev_fusion=True
        )

        self.cmff_deep = CMFF(
            audio_dim=1024,  # XLSR dimension
            spec_dim=output_dim,
            output_dim=output_dim,
            use_prev_fusion=True
        )

        # Classifier
        self.classifier = ResNet18Classifier(input_dim=output_dim, num_classes=num_classes)

        # Single Center Loss
        if use_scl:
            self.scl = SingleCenterLoss(feature_dim=512)

    def unfreeze_xlsr(self):
        """Unfreeze XLSR after first epoch"""
        self.wave_encoder.unfreeze()

    def forward(self, waveform, pesq=None, labels=None, debug=False):
        """
        Args:
            waveform: (B, T_raw) - raw audio waveform
            pesq: (B,) or (B, 1) - PESQ scores (optional)
            labels: (B,) - labels for SCL (optional)
            debug: bool - whether to print debug information
        Returns:
            outputs: dict containing:
                - logits: (B, num_classes)
                - embedding: (B, 512)
                - scl_loss: scalar (if use_scl and labels provided)
                - debug_info: dict (if debug=True)
        """
        debug_info = {}

        if debug:
            print("\n" + "="*80)
            print("WaveSpec Forward Pass - Debug Mode")
            print("="*80)
            print(f"Input waveform shape: {waveform.shape}")
            if pesq is not None:
                print(f"PESQ scores: {pesq}")
            print(f"use_pesq: {self.use_pesq}")

        # Extract audio features
        audio_feats = self.wave_encoder(waveform)

        if debug:
            print("\n[1] XLSR Audio Features:")
            for key, feat in audio_feats.items():
                print(f"  {key:10s}: {feat.shape}")
            debug_info['xlsr_features'] = {k: v.shape for k, v in audio_feats.items()}

        # Extract spectrogram features (align to audio temporal dimension)
        target_length = audio_feats['shallow'].size(1)
        spec_feats = self.spec_encoder(waveform, target_length=target_length)

        if debug:
            print(f"\n[2] Spectrogram Features (target_length={target_length}):")
            for key, feat in spec_feats.items():
                print(f"  {key:10s}: {feat.shape}")
            debug_info['spec_features'] = {k: v.shape for k, v in spec_feats.items()}

        # CMFF fusion at 3 scales
        # Shallow fusion
        if debug:
            print("\n[3] CMFF Shallow Fusion:")
            f_f_1, cmff1_info = self.cmff_shallow(
                audio_feats['shallow'],
                spec_feats['shallow'],
                pesq=pesq,
                use_pesq=self.use_pesq,
                debug=True
            )
            print(f"  Input audio: {cmff1_info['input_audio_shape']}")
            print(f"  Input spec:  {cmff1_info['input_spec_shape']}")
            print(f"  Output:      {cmff1_info['output_fused_shape']}")
            if 'attention_info' in cmff1_info:
                print(f"  Attention stats:")
                for k, v in cmff1_info['attention_info'].items():
                    print(f"    {k}: {v:.6f}")
            debug_info['cmff_shallow'] = cmff1_info
        else:
            f_f_1 = self.cmff_shallow(
                audio_feats['shallow'],
                spec_feats['shallow'],
                pesq=pesq,
                use_pesq=self.use_pesq
            )

        # Mid fusion (with previous fusion)
        if debug:
            print("\n[4] CMFF Mid Fusion:")
            f_f_2, cmff2_info = self.cmff_mid(
                audio_feats['mid'],
                spec_feats['mid'],
                pesq=pesq,
                prev_fusion=f_f_1,
                use_pesq=self.use_pesq,
                debug=True
            )
            print(f"  Input audio: {cmff2_info['input_audio_shape']}")
            print(f"  Input spec:  {cmff2_info['input_spec_shape']}")
            print(f"  Prev fusion: {cmff2_info.get('prev_fusion_shape', 'N/A')}")
            print(f"  Output:      {cmff2_info['output_fused_shape']}")
            if 'attention_info' in cmff2_info:
                print(f"  Attention stats:")
                for k, v in cmff2_info['attention_info'].items():
                    print(f"    {k}: {v:.6f}")
            debug_info['cmff_mid'] = cmff2_info
        else:
            f_f_2 = self.cmff_mid(
                audio_feats['mid'],
                spec_feats['mid'],
                pesq=pesq,
                prev_fusion=f_f_1,
                use_pesq=self.use_pesq
            )

        # Deep fusion (with previous fusion)
        if debug:
            print("\n[5] CMFF Deep Fusion:")
            f_f_3, cmff3_info = self.cmff_deep(
                audio_feats['deep'],
                spec_feats['deep'],
                pesq=pesq,
                prev_fusion=f_f_2,
                use_pesq=self.use_pesq,
                debug=True
            )
            print(f"  Input audio: {cmff3_info['input_audio_shape']}")
            print(f"  Input spec:  {cmff3_info['input_spec_shape']}")
            print(f"  Prev fusion: {cmff3_info.get('prev_fusion_shape', 'N/A')}")
            print(f"  Output:      {cmff3_info['output_fused_shape']}")
            if 'attention_info' in cmff3_info:
                print(f"  Attention stats:")
                for k, v in cmff3_info['attention_info'].items():
                    print(f"    {k}: {v:.6f}")
            debug_info['cmff_deep'] = cmff3_info
        else:
            f_f_3 = self.cmff_deep(
                audio_feats['deep'],
                spec_feats['deep'],
                pesq=pesq,
                prev_fusion=f_f_2,
                use_pesq=self.use_pesq
            )

        # Classification
        logits, embedding = self.classifier(f_f_3, return_embedding=True)

        if debug:
            print(f"\n[6] Classifier:")
            print(f"  Input fused feature: {f_f_3.shape}")
            print(f"  Output logits:       {logits.shape}")
            print(f"  Output embedding:    {embedding.shape}")
            debug_info['classifier'] = {
                'input_shape': tuple(f_f_3.shape),
                'logits_shape': tuple(logits.shape),
                'embedding_shape': tuple(embedding.shape)
            }

        outputs = {
            'logits': logits,
            'embedding': embedding
        }

        # Compute SCL if labels provided
        if self.use_scl and labels is not None:
            scl_loss = self.scl(embedding, labels)
            outputs['scl_loss'] = scl_loss

            if debug:
                print(f"\n[7] Single Center Loss:")
                print(f"  SCL Loss: {scl_loss.item():.6f}")
                debug_info['scl_loss'] = scl_loss.item()

        if debug:
            outputs['debug_info'] = debug_info
            print("\n" + "="*80)
            print("End of Forward Pass")
            print("="*80 + "\n")

        return outputs


# ============================================================================
# Wrapper for main.py compatibility
# ============================================================================
class Model(WaveSpec):
    """
    Wrapper class for compatibility with main.py training loop
    """
    def __init__(self, args, device):
        """
        Initialize WaveSpec model from args
        Args:
            args: argparse namespace with model config
            device: torch device
        """
        super(Model, self).__init__(
            xlsr_model_path=getattr(args, 'xlsr_model_path', '/home/woongjae/wildspoof/xlsr2_300m.pt'),
            freeze_xlsr=getattr(args, 'freeze_xlsr', True),
            output_dim=getattr(args, 'output_dim', 1024),
            num_classes=getattr(args, 'num_classes', 2),
            use_scl=getattr(args, 'use_scl', True),
            use_pesq=getattr(args, 'use_pesq', False),
            device=device
        )
