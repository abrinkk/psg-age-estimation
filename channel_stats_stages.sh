#!/bin/bash
#
#SBATCH --job-name=channel_stats_stages
#SBATCH -p mignot,normal,owners
#SBATCH --time=2-00:00
#SBATCH -c 4
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/log_cs_stages.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/log_cs_stages.err
############################################################

# Load custom conda environment
source $HOME/miniconda3/bin/activate
conda activate base

# Change directory
cd $HOME/SleepAge/Scripts/utils/

# Run command
python channelStatisticsSTAGES.py