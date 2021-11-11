#!/bin/bash
#
#SBATCH --job-name=main
#SBATCH -p mignot,owners,gpu
#SBATCH --time=2-00:00
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH -C GPU_SKU:V100_SXM2
#SBATCH --mem=32GB
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/log_age_estimate.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/log_age_estimate.err
############################################################

# Load custom conda environment
source $HOME/miniconda3/bin/activate
conda activate base

# Change directory
cd $HOME/SleepAge/Scripts/

# Run command

python age_estimate.py --input_folder /scratch/users/abk26/nAge/all/ --pre_hyperparam 1e-3 1e-5 0.75 0.1 1 32 5 0 0 --model_name 5