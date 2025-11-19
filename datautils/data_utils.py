"""
Data utilities for Cluster-based Learning

Simple dataloader compatible with the cluster-based metric learning approach.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
import random
import soundfile as sf

SUPPORTED_DATALOADERS = ["data_utils"]


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """
    Parse protocol file and generate file lists

    Protocol format: <filename> <subset> <label>
    - filename: audio file name (with or without extension)
    - subset: 'train', 'dev', or 'eval'
    - label: 'bonafide' or 'spoof'

    Args:
        dir_meta: Path to protocol file
        is_train: If True, return train split
        is_eval: If True, return eval split

    Returns:
        For train/dev: (label_dict, file_list)
        For eval: file_list only
    """
    d_meta = {}
    file_list = []

    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            if len(parts) == 3:
                key, subset, label = parts
            elif len(parts) == 2:
                # Assume: filename subset (no label for eval)
                key, subset = parts
                label = None
            else:
                print(f"[WARN] Unexpected line format: {line.strip()}")
                continue

            if subset == 'train':
                file_list.append(key)
                if label:
                    # bonafide = 0, spoof = 1
                    d_meta[key] = 0 if label == 'bonafide' else 1

        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            if len(parts) >= 3:
                key, subset, label = parts[0], parts[1], parts[2] if len(parts) > 2 else None
            elif len(parts) == 2:
                # Could be: speaker_id filename OR filename subset
                # We'll assume filename subset
                key, subset = parts
            else:
                continue

            if subset == 'eval':
                file_list.append(key)

        return file_list

    else:  # dev
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            key, subset, label = parts[0], parts[1], parts[2]

            if subset == 'dev':
                file_list.append(key)
                # bonafide = 0, spoof = 1
                d_meta[key] = 0 if label == 'bonafide' else 1

        return d_meta, file_list


def pad(x, max_len=64000, padding_type="zero", random_start=True):
    """
    Pad or crop audio to fixed length

    Args:
        x: Input waveform (numpy array)
        max_len: Target length
        padding_type: 'zero' or 'repeat'
        random_start: If True, randomly crop from audio

    Returns:
        Padded/cropped waveform
    """
    x_len = len(x)

    if x_len >= max_len:
        # Crop
        if random_start:
            start = np.random.randint(0, x_len - max_len + 1)
            return x[start:start + max_len]
        else:
            return x[:max_len]
    else:
        # Pad
        if padding_type == "repeat":
            num_repeats = int(max_len / x_len) + 1
            return np.tile(x, num_repeats)[:max_len]
        else:  # zero padding
            padded = np.zeros(max_len, dtype=x.dtype)
            padded[:x_len] = x
            return padded


def apply_rawboost(x, args):
    """
    Apply RawBoost data augmentation

    Args:
        x: Waveform (numpy array)
        args: Arguments with RawBoost parameters

    Returns:
        Augmented waveform
    """
    if args.algo == 0:
        # No augmentation
        return x

    # Select algorithm
    if args.rb_random:
        # Random selection from available algorithms
        algo = random.choice([1, 2, 3])
    else:
        algo = args.algo

    # Apply augmentation
    if algo == 1:
        # LnL convolutive noise
        x = LnL_convolutive_noise(x, args)
    elif algo == 2:
        # ISD additive noise
        x = ISD_additive_noise(x, args)
    elif algo == 3:
        # SSI additive noise
        x = SSI_additive_noise(x, args)
    elif algo == 4:
        # Series: LnL + ISD + SSI
        x = LnL_convolutive_noise(x, args)
        x = ISD_additive_noise(x, args)
        x = SSI_additive_noise(x, args)
    elif algo == 5:
        # Series: LnL + ISD
        x = LnL_convolutive_noise(x, args)
        x = ISD_additive_noise(x, args)
    elif algo == 6:
        # Series: LnL + SSI
        x = LnL_convolutive_noise(x, args)
        x = SSI_additive_noise(x, args)
    elif algo == 7:
        # Series: ISD + SSI
        x = ISD_additive_noise(x, args)
        x = SSI_additive_noise(x, args)
    elif algo == 8:
        # Parallel: choose between LnL and ISD
        if random.random() < 0.5:
            x = LnL_convolutive_noise(x, args)
        else:
            x = ISD_additive_noise(x, args)

    # Normalize
    x = normWav(x, 0)

    return x


class Dataset_train(Dataset):
    """
    Training dataset for cluster-based learning

    Supports:
    - Standard protocol files
    - RawBoost augmentation
    - Variable-length audio (with padding/cropping)
    """

    def __init__(self, args, list_IDs, labels, base_dir, algo=3, rb_prob=0.5, random_algo=False):
        """
        Args:
            args: Argument object with RawBoost parameters
            list_IDs: List of file IDs
            labels: Dict mapping file ID to label (0=bonafide, 1=spoof)
            base_dir: Base directory for audio files
            algo: RawBoost algorithm (0=none, 1-8=different algorithms)
            rb_prob: Probability of applying RawBoost
            random_algo: If True, randomly select RawBoost algorithm
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.args = args
        self.algo = algo
        self.rb_prob = rb_prob
        self.random_algo = random_algo

        # Update args for augmentation
        self.args.algo = algo
        self.args.rb_prob = rb_prob
        self.args.rb_random = random_algo

        # Audio parameters
        self.max_len = 64000  # 4 seconds at 16kHz

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Returns:
            x: Waveform tensor [max_len, 1]
            y: Label (0 or 1)
        """
        file_id = self.list_IDs[index]
        y = self.labels[file_id]

        # Load audio
        audio_path = os.path.join(self.base_dir, file_id)
        if not audio_path.endswith('.flac'):
            audio_path = audio_path + '.flac'

        try:
            x, sr = sf.read(audio_path, dtype='float32')
        except:
            # Fallback: return silence if file not found
            print(f"[ERROR] Failed to load {audio_path}")
            x = np.zeros(self.max_len, dtype=np.float32)
            return torch.tensor(x).unsqueeze(-1), torch.tensor(y)

        # Apply RawBoost augmentation with probability
        if self.algo > 0 and random.random() < self.rb_prob:
            x = apply_rawboost(x, self.args)

        # Pad/crop to fixed length
        x = pad(x, max_len=self.max_len, padding_type="zero", random_start=True)

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # [max_len, 1]

        return x, torch.tensor(y, dtype=torch.long)


class Dataset_eval(Dataset):
    """
    Evaluation dataset (no augmentation, full-length audio)
    """

    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.max_len = 64000

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Returns:
            x: Waveform tensor [max_len, 1]
            utt_id: Utterance ID
        """
        utt_id = self.list_IDs[index]

        # Load audio
        audio_path = os.path.join(self.base_dir, utt_id)
        if not audio_path.endswith('.flac'):
            audio_path = audio_path + '.flac'

        try:
            x, sr = sf.read(audio_path, dtype='float32')
        except:
            print(f"[ERROR] Failed to load {audio_path}")
            x = np.zeros(self.max_len, dtype=np.float32)
            return torch.tensor(x).unsqueeze(-1), utt_id

        # Pad/crop (no random start for eval)
        x = pad(x, max_len=self.max_len, padding_type="zero", random_start=False)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)

        return x, utt_id
