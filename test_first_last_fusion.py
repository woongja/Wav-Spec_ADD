"""
Test script to verify first-last layer fusion with debugging

Usage:
    python test_first_last_fusion.py --fusion_type concat
    python test_first_last_fusion.py --fusion_type add
    python test_first_last_fusion.py --fusion_type weighted
"""

import torch
import argparse
import sys
import os

# Add model path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.conformertcm_fusion_concat import Model


def test_first_last_fusion(fusion_type='concat'):
    """
    Test first-last layer fusion with debugging output
    """
    print("\n" + "="*80)
    print(f"Testing First-Last Layer Fusion (fusion_type={fusion_type})")
    print("="*80 + "\n")

    # Setup args
    class Args:
        def __init__(self):
            self.ssl_path = '/home/woongjae/wildspoof/xlsr2_300m.pt'
            self.extractor_type = 'first_last'
            self.fusion_type = fusion_type
            self.n_layers = 24  # Use 24 layers (0-23, extract layer 0 and 23)
            self.emb_size = 256
            self.num_encoders = 2
            self.heads = 4
            self.kernel_size = 16

    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print()

    # Create model
    print("Creating model...")
    model = Model(args, device)
    model.to(device)
    model.eval()

    print("\nModel created successfully!")
    print()

    # Create dummy input (batch_size=2, seq_len=16000 * 3 = 48000)
    batch_size = 2
    seq_len = 48000  # 3 seconds at 16kHz
    dummy_input = torch.randn(batch_size, seq_len, 1).to(device)

    print(f"Testing with dummy input: {dummy_input.shape}")
    print()

    # Forward pass with debug=True (only for first batch)
    print("Running forward pass with DEBUG=True...")
    print()

    with torch.no_grad():
        output, attn_scores = model(dummy_input, debug=True)

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)
    print(f"Output shape: {output.shape}")
    print(f"Output probabilities (first sample): {torch.softmax(output[0], dim=0)}")
    print(f"Number of attention score layers: {len(attn_scores)}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Test First-Last Layer Fusion')
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'add', 'weighted'],
                        help='Fusion type to test')
    args = parser.parse_args()

    test_first_last_fusion(args.fusion_type)


if __name__ == '__main__':
    main()
