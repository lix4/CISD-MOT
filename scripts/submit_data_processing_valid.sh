#!/bin/bash
#SBATCH --job-name=MOT_detection_af
#SBATCH --partition=spartacus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --output=./dp_valid.out
#SBATCH --error=./dp_valid.err

# === 激活 Conda 环境 ===
source ~/.bashrc        # 确保 conda 命令可用（如果你用 bash）
# conda activate py39  # <<< 替换为你实际的环境名

# === 跑你的训练脚本 ===
cd datasets
python dataset_processing.py
