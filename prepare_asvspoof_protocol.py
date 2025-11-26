"""
ASVspoof2019 프로토콜 파일 병합 스크립트
Train과 Dev 프로토콜을 합쳐서 하나의 파일로 생성
"""

import os

# 경로 설정
TRAIN_PROTOCOL = "/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = "/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
OUTPUT_PROTOCOL = "/home/woongjae/ADD_LAB/SSL_Fusion_ADD/protocols/ASVspoof2019_LA_train_dev.txt"

def merge_protocols(train_path, dev_path, output_path):
    """
    Train과 Dev 프로토콜 파일을 병합
    Format: file_id subset label
    """
    print(f"[INFO] Reading train protocol from: {train_path}")
    train_lines = []
    with open(train_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                speaker_id, file_id, _, _, label = parts
                # Format: file_id subset label
                train_lines.append(f"{file_id}.flac train {label}\n")

    print(f"[INFO] Reading dev protocol from: {dev_path}")
    dev_lines = []
    with open(dev_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                speaker_id, file_id, _, _, label = parts
                # Format: file_id subset label
                dev_lines.append(f"{file_id}.flac dev {label}\n")

    # 병합
    merged_lines = train_lines + dev_lines

    print(f"[INFO] Writing merged protocol to: {output_path}")
    with open(output_path, 'w') as f:
        f.writelines(merged_lines)

    # 통계 출력
    print(f"\n[STATS] Train samples: {len(train_lines)}")
    print(f"[STATS] Dev samples: {len(dev_lines)}")
    print(f"[STATS] Total samples: {len(merged_lines)}")

    # 레이블 분포 확인
    bonafide_count = sum(1 for line in merged_lines if line.strip().endswith('bonafide'))
    spoof_count = sum(1 for line in merged_lines if line.strip().endswith('spoof'))

    print(f"\n[LABEL DISTRIBUTION]")
    print(f"  Bonafide: {bonafide_count} ({100*bonafide_count/len(merged_lines):.2f}%)")
    print(f"  Spoof: {spoof_count} ({100*spoof_count/len(merged_lines):.2f}%)")

    print(f"\n[SUCCESS] Protocol file saved to: {output_path}")

if __name__ == "__main__":
    merge_protocols(TRAIN_PROTOCOL, DEV_PROTOCOL, OUTPUT_PROTOCOL)
