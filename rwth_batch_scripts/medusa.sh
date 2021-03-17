#!/bin/bash

#SBATCH --job-name=medusa500
#SBATCH --output /home/cn226306/scheduled_jobs_output/%J_output.txt

#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1


# Create the logs folder and save a copy of the configuration
DATE=$(date '+%Y%m%d_%H%M')
PARAM_FILE="/home/cn226306/multimodal-har-thesis/src/parameters/utd_mhad/medusa.yaml"
LOG_FOLDER="/home/cn226306/scheduled_jobs_output/$SLURM_JOB_ID"
mkdir -p $LOG_FOLDER
cp $PARAM_FILE $LOG_FOLDER/config_used.yaml
echo "Script executed at $DATE" > $LOG_FOLDER/$DATE.txt

# Prepare
cd /home/cn226306/multimodal-har-thesis/
source venv/bin/activate
cd /home/cn226306/multimodal-har-thesis/src

# Execute
python -u train_medusa.py --experiment "medusa/default_ep500" --gpu 0 --epochs 500
