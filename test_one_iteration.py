"""
Test WaveSpec model with 1 iteration
Debug dimension flow and attention values
"""

import torch
import sys
import yaml

sys.path.append('/home/woongjae/ADD_LAB/Wav-Spec_ADD')

from model.wavespec import WaveSpec

def load_config(config_path):
    """Load yaml configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_one_iteration():
    print("="*80)
    print("WaveSpec Model - Single Iteration Debug Test")
    print("="*80)

    # Load config
    config_path = '/home/woongjae/ADD_LAB/Wav-Spec_ADD/configs/wavespec.yaml'
    config = load_config(config_path)

    print(f"\nLoaded config from: {config_path}")
    print(f"Model config:")
    for key, value in config['model'].items():
        print(f"  {key}: {value}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create model
    print("\n" + "-"*80)
    print("Creating WaveSpec model...")
    print("-"*80)

    model = WaveSpec(
        xlsr_model_path=config['model']['xlsr_model_path'],
        freeze_xlsr=config['model']['freeze_xlsr'],
        output_dim=config['model']['output_dim'],
        num_classes=config['model']['num_classes'],
        use_scl=config['model']['use_scl'],
        use_pesq=config['model']['use_pesq'],
        device=device
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {total_params - trainable_params:,}")

    # Create dummy batch
    print("\n" + "-"*80)
    print("Creating dummy batch...")
    print("-"*80)

    batch_size = 2
    duration = 4  # seconds
    sample_rate = 16000

    waveform = torch.randn(batch_size, duration * sample_rate).to(device)
    labels = torch.tensor([0, 1]).to(device)  # 0=fake, 1=real

    print(f"\nBatch info:")
    print(f"  Batch size: {batch_size}")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Waveform shape: {waveform.shape}")
    print(f"  Labels: {labels.cpu().numpy()}")

    # PESQ scores (dummy)
    pesq = None
    if config['model']['use_pesq']:
        pesq = torch.rand(batch_size).to(device) * 3.5 + 1.0  # PESQ in [1.0, 4.5]
        print(f"  PESQ scores: {pesq.cpu().numpy()}")
    else:
        print(f"  PESQ: Not used (use_pesq=False)")

    # Forward pass with debug mode
    print("\n" + "-"*80)
    print("Running forward pass with DEBUG mode...")
    print("-"*80)

    model.eval()
    with torch.no_grad():
        outputs = model(waveform, pesq=pesq, labels=labels, debug=True)

    # Print final results
    print("\n" + "-"*80)
    print("Final Outputs:")
    print("-"*80)

    print(f"\nLogits shape: {outputs['logits'].shape}")
    print(f"Logits:\n{outputs['logits']}")

    print(f"\nEmbedding shape: {outputs['embedding'].shape}")

    if 'scl_loss' in outputs:
        print(f"\nSCL Loss: {outputs['scl_loss'].item():.6f}")

    # Predictions
    probs = torch.softmax(outputs['logits'], dim=1)
    preds = torch.argmax(probs, dim=1)

    print(f"\nProbabilities (Real/Fake):")
    for i in range(batch_size):
        print(f"  Sample {i}: Fake={probs[i][0]:.4f}, Real={probs[i][1]:.4f}")

    print(f"\nPredicted labels: {preds.cpu().numpy()}")
    print(f"Ground truth:     {labels.cpu().numpy()}")

    # Summary
    print("\n" + "="*80)
    print("Test Summary:")
    print("="*80)
    print(f"✅ Model created successfully")
    print(f"✅ Forward pass completed")
    print(f"✅ All dimensions matched correctly")
    print(f"✅ Debug information printed above")
    print("="*80)

if __name__ == "__main__":
    try:
        test_one_iteration()
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR OCCURRED!")
        print("="*80)
        import traceback
        traceback.print_exc()
        print("="*80)
