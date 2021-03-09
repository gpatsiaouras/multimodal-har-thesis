#!/bin/bash

#SBATCH --job-name=mmact_sdfdi
#SBATCH --output /home/cn226306/scheduled_jobs_output/mmact_sdfdi.%J.log

#SBATCH --time=01:10:00
#SBATCH --gres=gpu:1


# Create the logs folder and save a copy of the configuration
DATE=$(date '+%Y%m%d_%H%M')
PARAM_FILE="/home/cn226306/multimodal-har-thesis/src/parameters/mmact/default.yaml"
LOG_FOLDER="/home/cn226306/scheduled_jobs_output/mmact_sdfdi_$SLURM_JOB_ID"
mkdir -p $LOG_FOLDER
cp $PARAM_FILE $LOG_FOLDER/config_used.yaml
echo "Script executed at $DATE" > $LOG_FOLDER/$DATE.txt

# Prepare
cd /home/cn226306/multimodal-har-thesis/
source venv/bin/activate
cd /home/cn226306/multimodal-har-thesis/src

# Execute
python -u train.py --modality sdfdi --param_file $PARAM_FILE --gpu 0