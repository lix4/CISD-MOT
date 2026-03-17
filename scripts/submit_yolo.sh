#!/bin/bash
#SBATCH --job-name=MOT_detection_yolo11x_MEB_ft
#SBATCH --partition=beasts,spartacus,spartacus-tl,gods
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --output=./runs/MOT_detection_yolo11x_MEB_ft/MOT_detection_out.txt
#SBATCH --error=./runs/MOT_detection_yolo11x_MEB_ft/MOT_detection_err.log

# === 激活 Conda 环境 ===
source ~/.bashrc        # 确保 conda 命令可用（如果你用 bash）
# conda activate py39  # <<< 替换为你实际的环境名

# === 获取作业名并传入 Python 脚本 ===
JOB_NAME=$SLURM_JOB_NAME

# === 跑你的训练脚本 ===
python object_detection.py train --epochs 100 --patience 30 --lr 1e-4 --job_name "$JOB_NAME" --weights ./runs/MOT_detection_yolo11s_aug/best_model.pt --od_model yolo11s.yaml
