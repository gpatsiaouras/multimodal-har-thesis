#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=train_skeleton_default
#SBATCH --output /home/cn226306/scheduled_jobs_output/train_skeleton_default.%J.log

#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1

# Prepare
cd /home/cn226306/multimodal-har-thesis/
source venv/bin/activate
cd /home/cn226306/multimodal-har-thesis/src

# Execute
python train_skeleton.py