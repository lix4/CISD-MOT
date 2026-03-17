#!/bin/bash
#SBATCH --job-name=MOT_classification_cl_v1_7_yolo_ab_bgdy
#SBATCH --partition=beasts,gods,spartacus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --output=./runs/MOT_classification_cl_v1_7_yolo_ab_bgdy/MOT_classification.txt
#SBATCH --error=./runs/MOT_classification_cl_v1_7_yolo_ab_bgdy/MOT_classification.err

# === 激活 Conda 环境 ===
source ~/.bashrc        # 确保 conda 命令可用（如果你用 bash）
# conda activate py39  # <<< 替换为你实际的环境名

# === 跑你的训练脚本 ===
python train_cl_debug.py
