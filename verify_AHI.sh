#!/bin/bash
#
#SBATCH --job-name=verify_AHI
#SBATCH -p mignot,normal,owners
#SBATCH --time=2-00:00
#SBATCH -c 4
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/log_ahi.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/log_ahi.err
############################################################

# Load custom conda environment
source $HOME/miniconda3/bin/activate
conda activate base

# Change directory
cd $HOME/SleepAge/Scripts/utils/

# Run command
python verify_AHI.py