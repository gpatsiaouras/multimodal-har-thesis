#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=skeleton_lstm_ep1000
#SBATCH --output /home/cn226306/scheduled_jobs_output/skeleton_lstm_ep1000.%J.log

#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# Prepare
cd /home/cn226306/multimodal-har-thesis/
source venv/bin/activate
cd /home/cn226306/multimodal-har-thesis/src

# Execute
python train_skeleton.py parameters/skeleton/lstm_ep1000.yaml