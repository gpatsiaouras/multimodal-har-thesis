#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=mmact_sdfdi
#SBATCH --output /home/cn226306/scheduled_jobs_output/mmact_sdfdi.%J/mmact_sdfdi.%J.log

#SBATCH --time=01:10:00
#SBATCH --gres=gpu:1


# Create the logs folder and save a copy of the configuration
PARAM_FILE=default.js
CONFIG_FILE=/home/cn226306/multimodal-har-thesis/src/parameters/mmact/$PARAM_FILE
DATE=$(date '+%Y%m%d_%H%M')
mkdir -p /home/cn226306/scheduled_jobs_output/mmact_sdfdi.%J
cp $CONFIG_FILE /home/cn226306/scheduled_jobs_output/mmact_sdfdi.%J/config_used.yaml
echo "Script executed at $DATE".txt > "$DATE".txt

# Prepare
cd /home/cn226306/multimodal-har-thesis/
source venv/bin/activate
cd /home/cn226306/multimodal-har-thesis/src

# Execute
python train.py --modality sdfdi --param_file parameters/mmact/$PARAM_FILE