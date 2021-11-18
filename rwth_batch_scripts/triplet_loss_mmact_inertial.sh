#!/bin/bash

#SBATCH --job-name=iner_sm
#SBATCH --output /home/cn226306/scheduled_jobs_output/%J_output.txt

#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1


# Create the logs folder and save a copy of the configuration
DATE=$(date '+%Y%m%d_%H%M')
PARAM_FILE="/home/cn226306/multimodal-har-thesis/src/parameters/mmact/triplet_loss.yaml"
LOG_FOLDER="/home/cn226306/scheduled_jobs_output/$SLURM_JOB_ID"
mkdir -p $LOG_FOLDER
cp $PARAM_FILE $LOG_FOLDER/config_used.yaml
echo "Script executed at $DATE" > $LOG_FOLDER/$DATE.txt

# Prepare
cd /home/cn226306/multimodal-har-thesis/
source venv/bin/activate
cd /home/cn226306/multimodal-har-thesis/src

# Execute
python -u train_triplet_loss.py --experiment "exper_mmact_tl/inertial_semi_hard" --gpu 0 --no_scheduler --modality inertial --param_file $PARAM_FILE --epochs 500 --semi_hard
