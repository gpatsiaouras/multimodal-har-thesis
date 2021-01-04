#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=train_skeleton_1000_epochs
#SBATCH --output /home/cn226306/scheduled_jobs_output/train_skeleton_1000_epochs.%J.log

#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1

# Prepare
cd /home/cn226306/multimodal-har-thesis/
source venv/bin/activate
cd /home/cn226306/multimodal-har-thesis/src

# Execute
python train_skeleton.py parameters/skeleton/1000_epochs.yaml