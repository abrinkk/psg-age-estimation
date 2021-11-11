# Setup imports
import os
import json
import numpy as np
import pandas as pd
import argparse

import torch
from torch.utils import data

from config import Config
from pretrainer import PreTrainer
from m_psg2label import M_PSG2FEAT
from psg_dataset import PSG_pretrain_Dataset
from utils.utils_loss import age_loss
from utils.util_json import NumpyEncoder

os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Age estimation interpretaiton.')
parser.add_argument('--m_run', type=str, default='eeg5', nargs='?',
                    const=True, help='Model to investigate')
parser.add_argument('--atr_method', type=str, default='int_grad', nargs='?',
                    const=True, help='Interpretation method')

args = parser.parse_args()

# Setup data and model
#m_run = 'resp5'
m_run = args.m_run
#atr_method = 'int_grad'
atr_method = args.atr_method
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config
config = Config()
config.return_only_pred = True
config.do_f = 0.0
config.pre_channel_drop = False
config.model_dir = os.path.join(config.data_dir, 'model' + '_' + m_run)
config.model_F_path = os.path.join(config.model_dir, 'modelF')
config.model_L_path = os.path.join(config.model_dir, 'modelL')
config.save_dir = config.model_dir
if m_run == 'eeg5':
    config.only_eeg = 1
    config.n_channels = 2
elif m_run == 'eegeogemg5':
    config.only_eeg = 2
    config.n_channels = 5
elif m_run == 'ecg5':
    config.only_eeg = 3
    config.n_channels = 1
elif m_run == 'resp5':
    config.only_eeg = 4
    config.n_channels = 5
elif m_run == '5':
    config.only_eeg = 0
    config.n_channels = 12
config.interp_dir = os.path.join(config.model_dir, 'interpretation', atr_method)

# ds params
pretrain_params = {'batch_size': 2,
                       'num_workers': config.pre_n_workers,
                       'pin_memory': True}

# Setup data loaders
datasets_pre = PSG_pretrain_Dataset(config, 'test', n=-1)
dataloaders_pre = data.DataLoader(datasets_pre, shuffle=False, **pretrain_params)

# Initialize network
model_F = M_PSG2FEAT(config).to(device)

# Baseline prediction
#baseline = torch.zeros(1, 12, 5*128*60).to(device)
#out = model_F(baseline)
#print(out)

# Pretraining and evaluation
config.save_dir = config.model_F_path
loss_fn = age_loss(device, gamma_cov=0.01)
pretrainer = PreTrainer(model_F, loss_fn, config,
                 device=device,
                 num_epochs=config.pre_max_epochs,
                 patience=config.pre_patience,
                 resume=True)


# Run method
#pretrainer.activation_maximization(config.am_dir)
pretrainer.interpret_model(dataloaders_pre, save_path=config.interp_dir, atr_method=atr_method)