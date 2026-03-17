#!/bin/bash
#SBATCH --job-name=MOT_classification_cl_v1_9_yolo11s_audio_bbox_dis
#SBATCH --partition=tolga-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=./runs/MOT_classification_cl_v1_9_yolo11s_audio_bbox_dis/MOT_classification.txt
#SBATCH --error=./runs/MOT_classification_cl_v1_9_yolo11s_audio_bbox_dis/MOT_classification.err

# === 激活 Conda 环境 ===
source ~/.bashrc        # 确保 conda 命令可用（如果你用 bash）
# conda activate py39  # <<< 替换为你实际的环境名

# === 获取作业名并传入 Python 脚本 ===
JOB_NAME=$SLURM_JOB_NAME

# === 跑你的训练脚本 ===
python train_cl.py --job_name "$JOB_NAME" --model_key "audio_bbox_dis"
