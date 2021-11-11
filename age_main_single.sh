#!/bin/bash
#
#SBATCH --job-name=main
#SBATCH -p mignot
#SBATCH --time=2-00:00
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH --mem=32GB
#SBATCH -C GPU_SKU:V100_SXM2
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/log_main.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/log_main.err
############################################################

# Load custom conda environment
source $HOME/miniconda3/bin/activate
conda activate base

# Change directory
cd $HOME/SleepAge/Scripts/

# Run command
python age_main.py --pre_train True --pre_hyperparam 1e-3 1e-5 0.75 0.1