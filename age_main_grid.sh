#!/bin/bash
#
#SBATCH --job-name=main
#SBATCH -p mignot,normal,owners
#SBATCH --time=10:00
#SBATCH -c 1
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/submit_main_grid.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/submit_main_grid.err
############################################################

# Load custom conda environment
source $HOME/miniconda3/bin/activate
conda activate base

# Change directory
cd $HOME/SleepAge/Scripts/

# Run command
python age_main_grid.py