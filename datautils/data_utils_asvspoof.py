import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from .RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
import random

SUPPORTED_DATALOADERS = ["data_utils_asvspoof"]

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """
    ASVspoof2019 protocol parser

    Supports two formats:
    1. Original: speaker_id file_id - - label (5 columns)
       Example: LA_0079 LA_T_1138215 - - bonafide
    2. Merged: file_id subset label (3 columns)
       Example: LA_T_1138215.flac train bonafide
    """
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            parts = line.strip().split()

            # Format 1: 5 columns (original ASVspoof2019)
            if len(parts) == 5:
                speaker_id, file_id, _, _, label = parts
                file_list.append(file_id + '.flac')
                d_meta[file_id + '.flac'] = 1 if label == 'bonafide' else 0

            # Format 2: 3 columns (merged protocol)
            elif len(parts) == 3:
                file_id, subset, label = parts
                if subset == 'train':
                    file_list.append(file_id)
                    d_meta[file_id] = 1 if label == 'bonafide' else 0
            else:
                print(f"[WARN] Unexpected train line format: {line.strip()}")
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            parts = line.strip().split()

            # Format 1: 5 columns
            if len(parts) == 5:
                speaker_id, file_id, _, _, label = parts
                file_list.append(file_id + '.flac')

            # Format 2: 3 columns
            elif len(parts) == 3:
                file_id, subset, label = parts
                if subset == 'eval':
                    file_list.append(file_id)
            else:
                print(f"[WARN] Unexpected eval line format: {line.strip()}")
        return file_list

    else:  # dev
        for line in l_meta:
            parts = line.strip().split()

            # Format 1: 5 columns
            if len(parts) == 5:
                speaker_id, file_id, _, _, label = parts
                file_list.append(file_id + '.flac')
                d_meta[file_id + '.flac'] = 1 if label == 'bonafide' else 0

            # Format 2: 3 columns
            elif len(parts) == 3:
                file_id, subset, label = parts
                if subset == 'dev':
                    file_list.append(file_id)
                    d_meta[file_id] = 1 if label == 'bonafide' else 0
            else:
                print(f"[WARN] Unexpected dev line format: {line.strip()}")
        return d_meta, file_list


def pad(
    x: np.ndarray,
    padding_type: str = "zero",
    max_len: int = 64000,
    random_start: bool = True
) -> np.ndarray:
    """
    Pad or crop an audio signal to a fixed length.

    Args:
        x: np.ndarray - input waveform
        padding_type: str - 'zero' or 'repeat'
        max_len: int - output length
        random_start: bool - if True, randomly choose crop start point
    """
    x_len = len(x)
    padded_x = None

    if max_len <= 0:
        raise ValueError("max_len must be >= 0")

    if x_len >= max_len:
        # 길면 자르기 (랜덤 스타트 선택 가능)
        if random_start:
            start = np.random.randint(0, x_len - max_len + 1)
            padded_x = x[start:start + max_len]
        else:
            padded_x = x[:max_len]

    else:
        # 짧으면 패딩 or 반복
        if padding_type == "repeat":
            num_repeats = int(max_len / x_len) + 1
            padded_x = np.tile(x, num_repeats)[:max_len]
        elif padding_type == "zero":
            padded_x = np.zeros(max_len, dtype=x.dtype)
            padded_x[:x_len] = x

    return padded_x


# ===================================================== #
# RawBoost 데이터 증강 (랜덤 적용)
# ===================================================== #
def process_Rawboost_feature(feature, sr, args, algo, prob=0.5, random_algo=False):
    """
    Args:
        feature: waveform
        sr: sampling rate
        args: argument parser object
        algo: 지정된 RawBoost 알고리즘 (1~8)
        prob: 증강을 적용할 확률 (default=0.5)
        random_algo: True면 1~8 중 랜덤 선택
    """

    # ---- 1. 확률적으로 증강 적용 여부 결정 ---- #
    if random.random() > prob or algo == 0:
        return feature  # 그대로 반환 (No augmentation)

    # ---- 2. 알고리즘 랜덤 선택 모드 ---- #
    if random_algo:
        algo = random.randint(1, 8)

    # ---- 3. 알고리즘에 따른 증강 적용 ---- #
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )

    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    elif algo == 3:
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)

    return feature


class Dataset_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo, rb_prob=0.5, random_algo=False, random_start=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 64600
        self.rb_prob = rb_prob  # RawBoost 확률
        self.random_algo = random_algo  # 알고리즘 무작위 선택 여부
        self.random_start = random_start  # 패딩 시 랜덤 스타트 여부

    def __len__(self):
        return len(self.list_IDs)

    def _find_file_path(self, utt_id):
        """파일 경로를 자동으로 찾음 (ASVspoof2019 train/dev 폴더 지원)"""
        # 일반 경로 먼저 확인
        wav_path = os.path.join(self.base_dir, utt_id)
        if os.path.exists(wav_path):
            return wav_path

        # ASVspoof2019 구조: base_dir/ASVspoof2019_LA_{subset}/flac/{file}
        # Train 파일 (LA_T_로 시작)
        if utt_id.startswith('LA_T_'):
            return os.path.join(self.base_dir, "ASVspoof2019_LA_train", "flac", utt_id)
        # Dev 파일 (LA_D_로 시작)
        elif utt_id.startswith('LA_D_'):
            return os.path.join(self.base_dir, "ASVspoof2019_LA_dev", "flac", utt_id)
        # Eval 파일 (LA_E_로 시작)
        elif utt_id.startswith('LA_E_'):
            return os.path.join(self.base_dir, "ASVspoof2019_LA_eval", "flac", utt_id)
        # Fallback
        else:
            return wav_path

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        wav_path = self._find_file_path(utt_id)
        X, fs = librosa.load(wav_path, sr=16000)

        # 랜덤 RawBoost 적용
        X = process_Rawboost_feature(
            X, fs, self.args, self.algo,
            prob=self.rb_prob, random_algo=self.random_algo
        )

        X_pad = pad(X, max_len=self.cut, random_start=self.random_start)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target


class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def _find_file_path(self, utt_id):
        """파일 경로를 자동으로 찾음 (ASVspoof2019 train/dev/eval 폴더 지원)"""
        # 일반 경로 먼저 확인
        wav_path = os.path.join(self.base_dir, utt_id)
        if os.path.exists(wav_path):
            return wav_path

        # ASVspoof2019 구조: base_dir/ASVspoof2019_LA_{subset}/flac/{file}
        # Train 파일 (LA_T_로 시작)
        if utt_id.startswith('LA_T_'):
            return os.path.join(self.base_dir, "ASVspoof2019_LA_train", "flac", utt_id)
        # Dev 파일 (LA_D_로 시작)
        elif utt_id.startswith('LA_D_'):
            return os.path.join(self.base_dir, "ASVspoof2019_LA_dev", "flac", utt_id)
        # Eval 파일 (LA_E_로 시작)
        elif utt_id.startswith('LA_E_'):
            return os.path.join(self.base_dir, "ASVspoof2019_LA_eval", "flac", utt_id)
        # Fallback
        else:
            return wav_path

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        wav_path = self._find_file_path(utt_id)
        X, fs = librosa.load(wav_path, sr=16000)
        X_pad = pad(X, max_len=self.cut, random_start=False)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id
