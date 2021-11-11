#!/bin/bash
#
#SBATCH --job-name=edf2h5_cfs
#SBATCH -p mignot,normal,owners
#SBATCH --time=2-00:00
#SBATCH -c 4
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/log_cfs.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/log_cfs.err
############################################################

# Load custom conda environment
source $HOME/miniconda3/bin/activate
conda activate base

# Change directory
cd $HOME/SleepAge/Scripts/

# Run command
python psg2h5.py --input_folder /oak/stanford/groups/mignot/cfs/polysomnography/edfs/ --output_folder /scratch/users/abk26/nAge/ --cohort 'cfs' --n -1 --split 0.7 0.1 0.2