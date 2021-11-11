#!/bin/bash
#
#SBATCH --job-name=model_interp
#SBATCH -p mignot
#SBATCH --time=2-00:00
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH --mem=32GB
#SBATCH -C GPU_SKU:V100_SXM2
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/log_model_interp.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/log_model_interp.err
############################################################

# Load custom conda environment
source $HOME/miniconda3/bin/activate
conda activate base

# Change directory
cd $HOME/SleepAge/Scripts/

# Run command
python model_interpretation.py