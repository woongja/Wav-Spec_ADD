"""
Main training/evaluation script for Cluster-based Learning

This implements the Bonafide-only cluster metric learning approach:
1. Bonafide samples clustered and pulled toward centers
2. Spoof samples pushed away from Bonafide manifold
3. No separate spoof clusters - simple and stable
"""

import argparse
import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import importlib
import time
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter

# Local imports
from core_scripts.startup_config import set_random_seed
from datautils import SUPPORTED_DATALOADERS
from model import SUPPORTED_MODELS
from model.cluster_loss import BonafideClusterLoss, CombinedLoss
from evaluate_metrics import compute_eer


############################################
# Evaluation Function
############################################
def eval_model(args, config, device):
    """Evaluate trained model on eval set"""

    data_module = importlib.import_module("datautils." + config["data"]["name"])
    genSpoof_list = data_module.genSpoof_list
    Dataset_eval = data_module.Dataset_eval

    # Load eval protocol
    file_eval = genSpoof_list(args.protocol_path, is_eval=True)
    eval_set = Dataset_eval(list_IDs=file_eval, base_dir=args.database_path)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Transfer model config to args
    for key in ['emb_size', 'pooling_type', 'ssl_model_path']:
        if key in config["model"]:
            setattr(args, key, config["model"][key])

    # Load model
    modelClass = importlib.import_module("model." + config["model"]["name"]).Model
    model = modelClass(args, device)

    if args.model_path:
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded model checkpoint from {args.model_path}")

    model = model.to(device)
    model.eval()

    # Run evaluation
    print(f"[INFO] Running evaluation on {len(file_eval)} samples...")

    with torch.no_grad(), open(args.eval_output, 'w') as fh:
        for batch_x, utt_id in tqdm(eval_loader, desc="Evaluation"):
            batch_x = batch_x.to(device)

            # Get predictions - WaveSpec returns dict
            outputs = model(
                batch_x,
                pesq=None,
                labels=None,
                debug=False
            )

            logits = outputs['logits']

            # Get scores (bonafide probability)
            scores = F.softmax(logits, dim=1)[:, 0]  # bonafide score
            score_list = scores.cpu().numpy().tolist()

            # Write to file
            for f, score in zip(utt_id, score_list):
                # Format: filename bonafide_score
                fh.write(f'{f} {score:.6f}\n')

    print(f"[INFO] Eval scores saved to {args.eval_output}")

    # Compute EER if labels available
    try:
        eer_score = compute_eer_from_file(args)
        print(f"[INFO] EER: {eer_score*100:.4f}%")
    except:
        print("[INFO] Could not compute EER (labels may not be available)")


def compute_eer_from_file(args):
    """Compute EER from protocol and prediction files"""
    # Load protocol
    eval_df = pd.read_csv(args.protocol_path, sep=" ", header=None)
    eval_df.columns = ["utt", "subset", "label"]
    eval_df = eval_df[eval_df['subset'] == 'eval']

    # Load predictions
    pred_df = pd.read_csv(args.eval_output, sep=" ", header=None)
    pred_df.columns = ["utt", "bonafide_score"]

    # Merge
    res_df = pd.merge(eval_df, pred_df, on='utt')

    # Split by label
    spoof_scores = res_df[res_df['label'] == 'spoof']['bonafide_score']
    bonafide_scores = res_df[res_df['label'] == 'bonafide']['bonafide_score']

    # Compute EER
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)

    return eer


############################################
# Training Functions
############################################
def train_epoch(train_loader, model, criterion, optimizer, device, epoch, args):
    """Train for one epoch"""

    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    num_batches = 0

    # Track loss components
    loss_components = {
        'total': 0.0,
        'ce': 0.0,
        'cluster_total': 0.0,
        'cluster_bonafide': 0.0,
        'cluster_spoof': 0.0
    }

    pbar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch} [Train]")

    # Debug flag for first batch only
    debug_mode = args.debug if hasattr(args, 'debug') else False

    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).long()

        # Forward pass - WaveSpec returns dict
        # Enable debug only for first batch of first epoch
        enable_debug = debug_mode and epoch == 0 and batch_idx == 0

        outputs = model(
            batch_x,
            pesq=None,  # PESQ scores (to be added later)
            labels=batch_y,
            debug=enable_debug
        )

        logits = outputs['logits']
        embeddings = outputs['embedding']

        # Compute loss
        if args.loss_type == 'cluster_only':
            # Pure cluster metric loss
            loss, loss_dict = criterion(embeddings, batch_y)
        elif args.loss_type == 'combined':
            # Combined CE + Cluster loss
            loss, loss_dict = criterion(logits, embeddings, batch_y)
        else:  # 'ce_only'
            # Standard cross-entropy
            ce_loss = criterion(logits, batch_y)

            # Add SCL loss if available
            if 'scl_loss' in outputs:
                scl_weight = args.scl_weight if hasattr(args, 'scl_weight') else 0.1
                scl_loss = outputs['scl_loss']
                loss = ce_loss + scl_weight * scl_loss
                loss_dict = {
                    'total_loss': loss.item(),
                    'ce_loss': ce_loss.item(),
                    'scl_loss': scl_loss.item()
                }
            else:
                loss = ce_loss
                loss_dict = {'total_loss': loss.item(), 'ce_loss': loss.item()}

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)
        num_batches += 1

        # Accumulate loss components
        for key in loss_dict:
            if key in loss_components:
                loss_components[key] += loss_dict[key]

        # Update progress bar
        current_avg_loss = running_loss / num_batches
        current_acc = correct / total * 100
        pbar.set_postfix({
            'loss': f'{current_avg_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    avg_loss = running_loss / num_batches
    acc = correct / total * 100

    # Average loss components
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, acc, loss_components


def eval_epoch(dev_loader, model, criterion, device, epoch, args):
    """Evaluate for one epoch"""

    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    num_batches = 0

    # Track loss components
    loss_components = {
        'total': 0.0,
        'ce': 0.0,
        'cluster_total': 0.0,
        'cluster_bonafide': 0.0,
        'cluster_spoof': 0.0
    }

    pbar = tqdm(dev_loader, ncols=120, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()

            # Forward pass - WaveSpec returns dict
            outputs = model(
                batch_x,
                pesq=None,  # PESQ scores (to be added later)
                labels=batch_y,
                debug=False  # No debug during validation
            )

            logits = outputs['logits']
            embeddings = outputs['embedding']

            # Compute loss
            if args.loss_type == 'cluster_only':
                loss, loss_dict = criterion(embeddings, batch_y)
            elif args.loss_type == 'combined':
                loss, loss_dict = criterion(logits, embeddings, batch_y)
            else:  # 'ce_only'
                ce_loss = criterion(logits, batch_y)

                # Add SCL loss if available
                if 'scl_loss' in outputs:
                    scl_weight = args.scl_weight if hasattr(args, 'scl_weight') else 0.1
                    scl_loss = outputs['scl_loss']
                    loss = ce_loss + scl_weight * scl_loss
                    loss_dict = {
                        'total_loss': loss.item(),
                        'ce_loss': ce_loss.item(),
                        'scl_loss': scl_loss.item()
                    }
                else:
                    loss = ce_loss
                    loss_dict = {'total_loss': loss.item(), 'ce_loss': loss.item()}

            val_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
            num_batches += 1

            # Accumulate loss components
            for key in loss_dict:
                if key in loss_components:
                    loss_components[key] += loss_dict[key]

            # Update progress bar
            current_avg_loss = val_loss / num_batches
            current_acc = correct / total * 100
            pbar.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    avg_loss = val_loss / num_batches
    acc = correct / total * 100

    # Average loss components
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, acc, loss_components


############################################
# Main
############################################
def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load config
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # ========== EVAL MODE ==========
    if args.eval:
        eval_model(args, config, device)
        sys.exit(0)

    # ========== TRAIN MODE ==========
    print("[INFO] Training mode activated")
    set_random_seed(args.seed)

    # ---------------------------------------
    # Load dataloader
    # ---------------------------------------
    if config["data"]["name"] not in SUPPORTED_DATALOADERS:
        raise ValueError(f"Dataloader {config['data']['name']} not supported")

    data_module = importlib.import_module("datautils." + config["data"]["name"])
    genList = data_module.genSpoof_list
    Dataset_train = data_module.Dataset_train
    Dataset_eval = data_module.Dataset_eval

    print(f"[INFO] Using protocol file: {args.protocol_path}")

    # Load train/dev splits
    d_label_trn, file_train = genList(args.protocol_path, is_train=True)
    d_label_dev, file_dev = genList(args.protocol_path, is_train=False)

    print(f"[INFO] Loaded {len(file_train)} training and {len(file_dev)} validation samples")

    # Create datasets
    train_set = Dataset_train(
        args=args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=args.database_path,
        algo=args.algo,
        rb_prob=args.rb_prob,
        random_algo=args.rb_random
    )

    dev_set = Dataset_train(
        args=args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=args.database_path,
        algo=0,  # No augmentation for dev
        rb_prob=0.0,
        random_algo=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # ---------------------------------------
    # Load model
    # ---------------------------------------
    if config["model"]["name"] not in SUPPORTED_MODELS:
        raise ValueError(f"Model {config['model']['name']} not supported")

    # Transfer model config to args
    for key in config["model"]:
        if key != "name":
            setattr(args, key, config["model"][key])

    modelClass = importlib.import_module("model." + config["model"]["name"]).Model
    model = modelClass(args, device).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    # ---------------------------------------
    # Setup loss function
    # ---------------------------------------
    print(f"[INFO] Loss type: {args.loss_type}")

    if args.loss_type == 'cluster_only':
        # Pure cluster metric loss
        criterion = BonafideClusterLoss(
            num_clusters=args.num_clusters,
            embedding_dim=args.emb_size,
            alpha=args.cluster_alpha,
            cluster_update_freq=args.cluster_update_freq,
            cluster_method=args.cluster_method
        ).to(device)
        print(f"[INFO] Using Bonafide-only Cluster Loss")
        print(f"  - Num clusters: {args.num_clusters}")
        print(f"  - Alpha (spoof repulsion): {args.cluster_alpha}")
        print(f"  - Cluster method: {args.cluster_method}")

    elif args.loss_type == 'combined':
        # Combined CE + Cluster loss
        criterion = CombinedLoss(
            num_clusters=args.num_clusters,
            embedding_dim=args.emb_size,
            alpha=args.cluster_alpha,
            ce_weight=args.ce_weight,
            cluster_weight=args.cluster_weight,
            cluster_update_freq=args.cluster_update_freq,
            cluster_method=args.cluster_method
        ).to(device)
        print(f"[INFO] Using Combined Loss (CE + Cluster)")
        print(f"  - CE weight: {args.ce_weight}")
        print(f"  - Cluster weight: {args.cluster_weight}")

    else:  # 'ce_only'
        # Standard cross-entropy
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        print(f"[INFO] Using Cross-Entropy Loss (baseline)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay
    )

    # Setup logging
    log_dir = f"logs/{args.comment}" if args.comment else "logs/default"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    log_file = os.path.join(log_dir, "training.log")
    with open(log_file, "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = args.patience

    # Create directory for model saving
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    # ---------------------------------------
    # Training loop
    # ---------------------------------------
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")

    start_time_total = time.time()
    prev_val_loss = 0

    for epoch in range(args.start_epoch, args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*80}")

        epoch_start_time = time.time()

        # Train
        tr_loss, tr_acc, tr_components = train_epoch(
            train_loader, model, criterion, optimizer, device, epoch, args
        )

        # Validate
        val_loss, val_acc, val_components = eval_epoch(
            dev_loader, model, criterion, device, epoch, args
        )

        epoch_time = time.time() - epoch_start_time

        # Calculate loss delta
        if epoch > args.start_epoch:
            loss_delta = val_loss - prev_val_loss
            loss_delta_str = f"({loss_delta:+.4f})"
        else:
            loss_delta_str = ""

        prev_val_loss = val_loss

        # Print summary
        print(f"\n{'─'*80}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"{'─'*80}")
        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} {loss_delta_str} | Val Acc: {val_acc:.2f}%")

        # Print loss components if using cluster loss
        if args.loss_type in ['cluster_only', 'combined']:
            print(f"  Loss Components:")
            if args.loss_type == 'combined':
                print(f"    CE: {val_components['ce']:.4f}")
            print(f"    Cluster Total: {val_components['cluster_total']:.4f}")
            print(f"    Bonafide Pull: {val_components['cluster_bonafide']:.4f}")
            print(f"    Spoof Repulsion: {val_components['cluster_spoof']:.4f}")

        print(f"  Time: {epoch_time:.2f}s")
        print(f"{'─'*80}")

        # Log to tensorboard
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Log loss components
        if args.loss_type in ['cluster_only', 'combined']:
            writer.add_scalar("Loss/cluster_bonafide", val_components['cluster_bonafide'], epoch)
            writer.add_scalar("Loss/cluster_spoof", val_components['cluster_spoof'], epoch)

        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.2f},"
                   f"{val_loss:.4f},{val_acc:.2f}\n")

        # Early stopping check
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), args.model_save_path)
            print(f"✅ Validation loss improved by {improvement:.4f}! Model saved to {args.model_save_path}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                break

    total_time = time.time() - start_time_total

    writer.close()

    print("\n" + "="*80)
    print("Training Completed!")
    print("="*80)
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Logs saved to: {log_dir}")
    print(f"  Model saved to: {args.model_save_path}")
    print("="*80 + "\n")


############################################
# Entry
############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster-based Learning for Audio Deepfake Detection")

    # Dataset
    parser.add_argument("--database_path", type=str, default="/AISRC2/Dataset/")
    parser.add_argument("--protocol_path", type=str, default="protocols/protocol.txt")
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    # Hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Loss configuration
    parser.add_argument("--loss_type", type=str, default="combined",
                       choices=["ce_only", "cluster_only", "combined"],
                       help="Loss function type")
    parser.add_argument("--num_clusters", type=int, default=10,
                       help="Number of Bonafide clusters")
    parser.add_argument("--cluster_alpha", type=float, default=1.0,
                       help="Weight for spoof repulsion loss")
    parser.add_argument("--cluster_method", type=str, default="kmeans",
                       choices=["kmeans", "online"],
                       help="Clustering method")
    parser.add_argument("--cluster_update_freq", type=int, default=100,
                       help="Update cluster centers every N iterations")
    parser.add_argument("--ce_weight", type=float, default=1.0,
                       help="Weight for CE loss in combined mode")
    parser.add_argument("--cluster_weight", type=float, default=0.5,
                       help="Weight for cluster loss in combined mode")

    # Misc
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default="out/best_model.pth")

    # Eval mode
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_output", type=str, default="eval_scores.txt")

    # RawBoost augmentation
    parser.add_argument('--algo', type=int, default=3,
                       help='RawBoost algorithm: 0=none, 1-8=various augmentations')
    parser.add_argument('--rb_prob', type=float, default=0.5,
                       help='Probability of applying RawBoost')
    parser.add_argument('--rb_random', action='store_true',
                       help='Randomly select RawBoost algorithm')

    # RawBoost parameters (LnL)
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f', type=int, default=5)

    # RawBoost parameters (ISD)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)

    # RawBoost parameters (SSI)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)

    args = parser.parse_args()
    main(args)
