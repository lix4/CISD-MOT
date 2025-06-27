#!/bin/bash
#SBATCH --job-name=AVIVDNet_v5_6_1_classification_224x224
#SBATCH --partition=spartacus-tl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=./runs/AVIVDNet_v5_6_1_classification_224x224/MOT_classification.txt
#SBATCH --error=./runs/AVIVDNet_v5_6_1_classification_224x224/MOT_classification.err

# === 激活 Conda 环境 ===
source ~/.bashrc        # 确保 conda 命令可用（如果你用 bash）
# conda activate py39  # <<< 替换为你实际的环境名

# === 跑你的训练脚本 ===
python train_classifier.py
