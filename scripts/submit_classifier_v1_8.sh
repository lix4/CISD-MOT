#!/bin/bash
#SBATCH --job-name=MOT_classification_v1_8
#SBATCH --partition=gods
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --output=./runs/MOT_classification_v1_8/MOT_classification.txt
#SBATCH --error=./runs/MOT_classification_v1_8/MOT_classification.err

# === 激活 Conda 环境 ===
source ~/.bashrc        # 确保 conda 命令可用（如果你用 bash）
# conda activate py39  # <<< 替换为你实际的环境名

# === 跑你的训练脚本 ===
python train_classifier_af.py
