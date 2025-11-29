#!/bin/bash

# ========================
# WaveSpec Training Script
# ASVspoof2019 LA Dataset
# ========================

# 데이터베이스 경로 (train/dev 자동 감지 - data_utils_asvspoof에서 처리)
DATABASE_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019"

# 병합된 프로토콜 파일 (train + dev, subset 정보 포함)
PROTOCOL_PATH="/home/woongjae/ADD_LAB/Wav-Spec_ADD/protocols/ASVspoof2019_LA_train_dev.txt"

# Config 및 저장 경로
CONFIG_FILE="/home/woongjae/ADD_LAB/Wav-Spec_ADD/configs/wavespec.yaml"
MODEL_SAVE_PATH="/home/woongjae/ADD_LAB/Wav-Spec_ADD/out/wavespec_baseline_asvspoof2019.pth"
COMMENT="wavespec_baseline_asvspoof2019_no_pesq"

# ========================
# WaveSpec Training 실행
# ========================
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=MIG-8cdeef83-092c-5a8d-a748-452f299e1df0 python /home/woongjae/ADD_LAB/Wav-Spec_ADD/main.py \
  --database_path ${DATABASE_PATH} \
  --protocol_path ${PROTOCOL_PATH} \
  --config ${CONFIG_FILE} \
  --batch_size 8 \
  --num_epochs 100 \
  --max_lr 1e-6 \
  --weight_decay 1e-4 \
  --patience 10 \
  --seed 1234 \
  --model_save_path ${MODEL_SAVE_PATH} \
  --comment ${COMMENT} \
  --algo 3
