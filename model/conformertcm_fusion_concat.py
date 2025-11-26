import torch
import torch.nn as nn
import fairseq
from .conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones
from torch import Tensor

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

class MyConformer(nn.Module):
  def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
    super(MyConformer, self).__init__()
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.kernel_size=kernel_size
    self.n_encoders=n_encoders
    self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
    self.encoder_blocks=_get_clones( ConformerBlock( dim = emb_size, dim_head=self.dim_head, heads= heads, 
    ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size),
    n_encoders)
    self.class_token = nn.Parameter(torch.rand(1, emb_size))
    self.fc5 = nn.Linear(emb_size, 2)

  def forward(self, x, device): # x shape [bs, tiempo, frecuencia]
    x = x + self.positional_emb[:, :x.size(1), :]
    x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
    list_attn_weight = []
    for layer in self.encoder_blocks:
            x, attn_weight = layer(x) #[bs,1+tiempo,emb_size]
            list_attn_weight.append(attn_weight)
    embedding=x[:,0,:] #[bs, emb_size]
    out=self.fc5(embedding) #[bs,2]
    return out, list_attn_weight

class SSLModel(nn.Module): #W2V
    def __init__(self, ssl_pretrained_path, n_layers, extractor_type='first_last'):
        super(SSLModel, self).__init__()
        cp_path = ssl_pretrained_path   # Change the pre-trained XLSR model path.
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.total_layers = len(self.model.encoder.layers)
        self.n_layers = n_layers if n_layers is not None else self.total_layers
        self.extractor_type = extractor_type

        # Print layer info
        print(f"[INFO] Total W2V layers: {self.total_layers}")
        print(f"[INFO] Using n_layers: {self.n_layers}")
        print(f"[INFO] Extractor type: {self.extractor_type}")

        if self.extractor_type == 'first_last':
            print(f"[INFO] Extracting ONLY first (layer 0) and last (layer {self.n_layers-1}) layers")

        # Keep all layers but we'll only extract first and last
        # Don't slice layers here - we need all layers to get the last one


    def extract_feat(self, input_data, debug=False):
        """
        Extract features from SSL model

        Args:
            input_data: input audio tensor
            debug: if True, print detailed debugging information

        Returns:
            - If extractor_type == 'first_last': [B, 2, T, D] (first and last layer)
            - If extractor_type == 'layerwise': [B, n_layers, T, D] (all layers)
            - Otherwise: [B, T, D] (final layer only)
        """
        input_data = input_data.squeeze(1)
        dict_ = self.model(input_data, mask=False, features_only=True)
        x, layerresult = dict_['x'], dict_['layer_results']

        if self.extractor_type == 'first_last':
            # Extract only first and last layer
            first_layer = layerresult[0]  # First layer
            last_layer = layerresult[self.n_layers - 1]  # Last layer

            if debug:
                print("\n" + "="*80)
                print("[DEBUG SSLModel.extract_feat] First-Last Layer Extraction")
                print("="*80)
                print(f"Total layer_results available: {len(layerresult)}")
                print(f"Extracting layer 0 (first) and layer {self.n_layers - 1} (last)")
                print(f"First layer type: {type(first_layer)}")
                print(f"Last layer type: {type(last_layer)}")

            # Convert to tensor format [B, T, D]
            first_layer_feat = first_layer[0].permute(1, 0, 2) if isinstance(first_layer, tuple) else first_layer
            last_layer_feat = last_layer[0].permute(1, 0, 2) if isinstance(last_layer, tuple) else last_layer

            if debug:
                print(f"\n[First Layer Features]")
                print(f"  Shape: {first_layer_feat.shape}")
                print(f"  Mean: {first_layer_feat.mean().item():.6f}")
                print(f"  Std: {first_layer_feat.std().item():.6f}")
                print(f"  Min: {first_layer_feat.min().item():.6f}")
                print(f"  Max: {first_layer_feat.max().item():.6f}")
                print(f"  Has NaN: {torch.isnan(first_layer_feat).any().item()}")
                print(f"  Has Inf: {torch.isinf(first_layer_feat).any().item()}")

                print(f"\n[Last Layer Features]")
                print(f"  Shape: {last_layer_feat.shape}")
                print(f"  Mean: {last_layer_feat.mean().item():.6f}")
                print(f"  Std: {last_layer_feat.std().item():.6f}")
                print(f"  Min: {last_layer_feat.min().item():.6f}")
                print(f"  Max: {last_layer_feat.max().item():.6f}")
                print(f"  Has NaN: {torch.isnan(last_layer_feat).any().item()}")
                print(f"  Has Inf: {torch.isinf(last_layer_feat).any().item()}")

            # Stack: [B, 2, T, D]
            stacked = torch.stack([first_layer_feat, last_layer_feat], dim=1)

            if debug:
                print(f"\n[Stacked Output]")
                print(f"  Shape: {stacked.shape}")
                print("="*80 + "\n")

            return stacked

        elif self.extractor_type == 'layerwise':
            # Extract all n_layers
            return torch.stack([t[0].permute(1, 0, 2) if isinstance(t, tuple) else t for t in layerresult[:self.n_layers]], dim=1)

        else:
            # Return final output only
            return x

class FirstLastFusion(nn.Module):
    """
    Fusion module for first and last layer embeddings

    Supports multiple fusion strategies:
    - 'concat': concatenate first and last → linear projection
    - 'add': element-wise addition
    - 'weighted': learnable weighted sum
    """
    def __init__(self, input_dim=1024, output_dim=256, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        if fusion_type == 'concat':
            # Concat [first; last] → project to output_dim
            self.projection = nn.Linear(input_dim * 2, output_dim)
        elif fusion_type == 'add':
            # Simple addition → project to output_dim
            self.projection = nn.Linear(input_dim, output_dim)
        elif fusion_type == 'weighted':
            # Learnable weighted sum
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Weight for first layer
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        print(f"[INFO] FirstLastFusion - Type: {fusion_type}, Input: {input_dim}, Output: {output_dim}")

    def forward(self, layer_feats, debug=False):
        """
        Args:
            layer_feats: [B, 2, T, D] - stacked first and last layer features
            debug: if True, print detailed debugging information

        Returns:
            fused: [B, T, output_dim] - fused features
        """
        first_layer = layer_feats[:, 0, :, :]  # [B, T, D]
        last_layer = layer_feats[:, 1, :, :]   # [B, T, D]

        if debug:
            print("\n" + "="*80)
            print("[DEBUG FirstLastFusion.forward] Layer Fusion")
            print("="*80)
            print(f"Fusion type: {self.fusion_type}")
            print(f"\n[Before Fusion]")
            print(f"  First layer shape: {first_layer.shape}")
            print(f"  Last layer shape: {last_layer.shape}")
            print(f"  First layer - Mean: {first_layer.mean().item():.6f}, Std: {first_layer.std().item():.6f}")
            print(f"  Last layer  - Mean: {last_layer.mean().item():.6f}, Std: {last_layer.std().item():.6f}")

            # Compute correlation between first and last layer
            first_flat = first_layer.reshape(-1, first_layer.size(-1))
            last_flat = last_layer.reshape(-1, last_layer.size(-1))
            correlation = torch.nn.functional.cosine_similarity(first_flat, last_flat, dim=1).mean()
            print(f"  Cosine similarity between layers: {correlation.item():.6f}")

        if self.fusion_type == 'concat':
            # Concatenate along feature dimension
            concat_feat = torch.cat([first_layer, last_layer], dim=-1)  # [B, T, 2*D]

            if debug:
                print(f"\n[Concatenation]")
                print(f"  Concatenated shape: {concat_feat.shape}")
                print(f"  Concatenated - Mean: {concat_feat.mean().item():.6f}, Std: {concat_feat.std().item():.6f}")

            fused = self.projection(concat_feat)  # [B, T, output_dim]

        elif self.fusion_type == 'add':
            # Element-wise addition
            add_feat = first_layer + last_layer  # [B, T, D]

            if debug:
                print(f"\n[Element-wise Addition]")
                print(f"  Added shape: {add_feat.shape}")
                print(f"  Added - Mean: {add_feat.mean().item():.6f}, Std: {add_feat.std().item():.6f}")

            fused = self.projection(add_feat)    # [B, T, output_dim]

        elif self.fusion_type == 'weighted':
            # Learnable weighted sum
            if debug:
                print(f"\n[Weighted Sum]")
                print(f"  Alpha (first layer weight): {self.alpha.item():.6f}")
                print(f"  Beta (last layer weight): {(1 - self.alpha).item():.6f}")

            weighted_feat = self.alpha * first_layer + (1 - self.alpha) * last_layer  # [B, T, D]

            if debug:
                print(f"  Weighted shape: {weighted_feat.shape}")
                print(f"  Weighted - Mean: {weighted_feat.mean().item():.6f}, Std: {weighted_feat.std().item():.6f}")

            fused = self.projection(weighted_feat)  # [B, T, output_dim]

        if debug:
            print(f"\n[After Fusion & Projection]")
            print(f"  Fused shape: {fused.shape}")
            print(f"  Fused - Mean: {fused.mean().item():.6f}, Std: {fused.std().item():.6f}")
            print(f"  Fused - Min: {fused.min().item():.6f}, Max: {fused.max().item():.6f}")
            print(f"  Has NaN: {torch.isnan(fused).any().item()}")
            print(f"  Has Inf: {torch.isinf(fused).any().item()}")
            print("="*80 + "\n")

        return fused


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        ####
        # Create network wav2vec 2.0
        ####
        ssl_pretrained_path = getattr(args, 'ssl_path', '/home/woongjae/wildspoof/xlsr2_300m.pt')
        n_layers = getattr(args, 'n_layers', None)  # None = use all layers
        extractor_type = getattr(args, 'extractor_type', 'first_last')  # Default: first_last
        fusion_type = getattr(args, 'fusion_type', 'concat')  # Default: concat

        self.ssl_model = SSLModel(ssl_pretrained_path, n_layers, extractor_type)

        # Fusion module for first and last layer
        if extractor_type == 'first_last':
            self.fusion = FirstLastFusion(
                input_dim=1024,
                output_dim=args.emb_size,
                fusion_type=fusion_type
            )
            print(f'[INFO] W2V (First+Last Fusion) + Conformer')
        else:
            # Fallback to original linear projection for other modes
            self.LL = nn.Linear(1024, args.emb_size)
            print('[INFO] W2V (Standard) + Conformer')

        self.extractor_type = extractor_type
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer = MyConformer(
            emb_size=args.emb_size,
            n_encoders=args.num_encoders,
            heads=args.heads,
            kernel_size=args.kernel_size
        )

    def forward(self, x, debug=False):
        """
        Args:
            x: input audio [B, T] or [B, T, 1]
            debug: if True, print detailed debugging information (only first batch)

        Returns:
            out: [B, 2] class logits
            attn_score: attention weights from Conformer
        """
        # Extract SSL features
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1), debug=debug)
        # x_ssl_feat: [B, 2, T, D] for first_last mode
        #             [B, n_layers, T, D] for layerwise mode
        #             [B, T, D] for default mode

        if debug:
            print("\n" + "="*80)
            print("[DEBUG Model.forward] Main Model Forward Pass")
            print("="*80)
            print(f"Input shape: {x.shape}")
            print(f"SSL features shape: {x_ssl_feat.shape}")
            print(f"Extractor type: {self.extractor_type}")

        if self.extractor_type == 'first_last':
            # Fuse first and last layer
            x = self.fusion(x_ssl_feat, debug=debug)  # [B, T, emb_size]
        else:
            # Standard linear projection
            x = self.LL(x_ssl_feat)  # [B, T, emb_size]

        if debug:
            print(f"\n[After Fusion/Projection]")
            print(f"  Shape: {x.shape}")
            print(f"  Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

        # Batch normalization
        x = x.unsqueeze(dim=1)  # [B, 1, T, emb_size]
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)     # [B, T, emb_size]

        if debug:
            print(f"\n[After BatchNorm + SELU]")
            print(f"  Shape: {x.shape}")
            print(f"  Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
            print("="*80 + "\n")

        # Conformer
        out, attn_score = self.conformer(x, self.device)

        if debug:
            print(f"[After Conformer]")
            print(f"  Output shape: {out.shape}")
            print(f"  Output: {out[0]}")
            print("="*80 + "\n")

        return out, attn_score

