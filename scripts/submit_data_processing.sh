#!/bin/bash
#SBATCH --job-name=MOT_data_processing
#SBATCH --partition=beasts,gods,spartacus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00

# === 激活 Conda 环境 ===
source ~/.bashrc        # 确保 conda 命令可用（如果你用 bash）
# conda activate py39  # <<< 替换为你实际的环境名

# === 跑你的训练脚本 ===
cd datasets
python dataset_processing_lh_hm.py
