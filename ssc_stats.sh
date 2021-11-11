#!/bin/bash
#
#SBATCH --job-name=ssc_stats
#SBATCH -p mignot,normal,owners
#SBATCH --time=2-00:00
#SBATCH -c 4
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/log_ssc_stats.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/log_ssc_stats.err
############################################################

# Load custom conda environment
source $HOME/miniconda3/bin/activate
conda activate base

# Run command
python ssc_stats.py