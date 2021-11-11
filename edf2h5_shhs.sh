#!/bin/bash
#
#SBATCH --job-name=edf2h5_shhs
#SBATCH -p mignot,normal,owners
#SBATCH --time=2-00:00
#SBATCH -c 4
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/log_shhs.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/log_shhs.err
############################################################

# Load custom conda environment
source $HOME/miniconda3/bin/activate
conda activate base

# Change directory
cd $HOME/SleepAge/Scripts/

# Run command
python python psg2h5.py --input_folder /oak/stanford/groups/mignot/shhs/polysomnography/edfs/shhs1/ --output_folder /scratch/users/abk26/nAge/ --cohort 'shhs-v1' --n -1 --split 0.1765 0.0126 0.8109