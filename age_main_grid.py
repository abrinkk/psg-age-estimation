import os
import pdb
import sys
import tempfile

JOBS = [
    ('age_grid_1', 'python age_main.py --pre_train True --pre_hyperparam 1e-3 1e-5 0.75 0.1'),
	('age_grid_2', 'python age_main.py --pre_train True --pre_hyperparam 1e-3 1e-5 0.75 0.0'),
	('age_grid_3', 'python age_main.py --pre_train True --pre_hyperparam 1e-3 1e-2 0.75 0.1'),
	('age_grid_4', 'python age_main.py --pre_train True --pre_hyperparam 1e-3 1e-5 0.25 0.1'),
	('age_grid_4', 'python age_main.py --pre_train True --pre_hyperparam 1e-4 1e-5 0.75 0.1'),
	('age_grid_4', 'python age_main.py --pre_train True --pre_hyperparam 5e-4 1e-5 0.75 0.1')
    ]


def submit_job(jobname, experiment):

    content = '''#!/bin/bash
#
#SBATCH --job-name={0}
#SBATCH -p mignot
#SBATCH --time=7-00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:1
#SBATCH --mem 32GB
#SBATCH -C GPU_SKU:V100_SXM2
#SBATCH --output=/home/users/abk26/SleepAge/Scripts/logs/{0}.out
#SBATCH --error=/home/users/abk26/SleepAge/Scripts/logs/{0}.err
##################################################

source $HOME/miniconda3/bin/activate
conda activate base
cd $HOME/SleepAge/Scripts/

{1}
'''
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.format(jobname, experiment).encode())
    os.system('sbatch {}'.format(j.name))


if __name__ == '__main__':

    for job in JOBS:
        submit_job(job[0], job[1])

    print('All jobs have been submitted!')