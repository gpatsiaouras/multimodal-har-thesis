#!/bin/bash

#SBATCH --job-name=med_e32_h512
#SBATCH --output /home/cn226306/scheduled_jobs_output/%J_output.txt

#SBATCH --time=01:30:00
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
python -u train_medusa.py --experiment "medusa/exper_em32_h512_ep100" --gpu 0 --out_size 32 --mlp_hidden_size 512
