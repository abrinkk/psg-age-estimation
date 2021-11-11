import argparse
import time
from utils.str2bool import str2bool

import os
import json
import numpy as np
import pandas as pd
import ast
from pathlib import Path

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from config import Config
from pretrainer import PreTrainer
from trainer import Trainer
from m_psg2label import M_PSG2FEAT, M_FEAT2LABEL
from psg_dataset import PSG_pretrain_Dataset, PSG_feature_Dataset
from utils.utils_loss import age_loss
from utils.util_json import NumpyEncoder
from utils.utils_bo import bayesian_opt_eval

np.random.seed(0)
torch.manual_seed(0)

def main(args):
    """Estimates age in edfs in a specified directory

    Args:
        args (dict): See age_estimation.py -h
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Config
    config = Config()

    # Adjust config for hyperparameters
    hyper_param_string = 'lr_{:.7f}_l2_{:.7f}_dof_{:.3f}_doc_{:.3f}_lf_{:.0f}_bs_{:.0f}_sl_{:.0f}_sd_{:.0f}_eeg_{:.0f}_'.format(args.pre_hyperparam[0], args.pre_hyperparam[1], args.pre_hyperparam[2], args.pre_hyperparam[3], args.pre_hyperparam[4], args.pre_hyperparam[5], args.pre_hyperparam[6], args.pre_hyperparam[7], args.pre_hyperparam[8])
    # Learning rate for pretraining (not used)
    config.pre_lr = args.pre_hyperparam[0]
    # L2 regularization (not used)
    config.l2 = args.pre_hyperparam[1]
    # Drop rate for latent space (not used)
    config.do_f = args.pre_hyperparam[2]
    # Drop rate for input channels (not used)
    config.pre_channel_drop_prob = args.pre_hyperparam[3]
    # Loss function
    if args.pre_hyperparam[4] == 1:
        config.loss_func = 'huber'
    elif args.pre_hyperparam[4] == 2:
        config.loss_func = 'l1'
    elif args.pre_hyperparam[4] == 3:
        config.loss_func = 'nll_normal'
    elif args.pre_hyperparam[4] == 4:
        config.loss_func = 'nll_gamma'
    elif args.pre_hyperparam[4] == 5:
        config.loss_func = 'l2'
    # Batch size for pretraining
    config.pre_batch_size = int(args.pre_hyperparam[5])
    # Epoch size
    config.epoch_size = int(args.pre_hyperparam[6]*128*60)
    if args.pre_hyperparam[6] != 5:
        epoch_size_scale = args.pre_hyperparam[6] / 5
        config.pre_batch_size = int(config.pre_batch_size / epoch_size_scale)
        config.pad_length = int(config.pad_length / epoch_size_scale)
    # Only predict age in sleep (based on hypnograms)
    config.pre_only_sleep = args.pre_hyperparam[7]
    # Select channels
    config.only_eeg = args.pre_hyperparam[8]
    if config.only_eeg == 1:
        # C3-A2, C4-A1
        config.n_channels = 2
    elif config.only_eeg == 2:
        # C3-A2, C4-A1, L-EOG, R-EOG, Chin EMG
        config.n_channels = 5
    elif config.only_eeg == 3:
        # ECG
        config.n_channels = 1
    elif config.only_eeg == 4:
        # Airflow, nasal pressure, abd, thorax, SaO2
        config.n_channels = 5

    # Test name
    test_name = os.path.basename(os.path.normpath(args.input_folder))

    # Set config path to input path
    config.pretrain_dir = os.path.normpath(args.input_folder)
    config.F_train_dir = config.pretrain_dir + '_F_' + args.model_name
    Path(config.F_train_dir).mkdir(parents=True, exist_ok=True)

    # Set model paths
    config.model_dir = os.path.join(config.data_dir, 'model_' + args.model_name)
    config.save_dir = config.model_dir
    config.model_F_path = os.path.join(config.model_dir, 'modelF')
    config.model_L_path = os.path.join(config.model_dir, 'modelL')
    config.model_L_BO_path = config.model_L_path
    config.BO_expe_path = os.path.join(config.model_dir, 'exp')

    # Set model specifications
    model_hyperparam_path = os.path.join(config.model_dir,'hyperparam.txt')
    with open(model_hyperparam_path) as f:
        for line in f:
            model_hyperparam = ast.literal_eval(line)
    config.lstm_n = model_hyperparam['lstmn']
    config.do_l = model_hyperparam['do']
    config.l2 = model_hyperparam['l2']
    config.lr = model_hyperparam['lr']
    config.net_size_scale = model_hyperparam['size']

    # Tensorboard writer
    writer_name = 'runs/nAge_' + hyper_param_string + time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(writer_name)

    # Set up dataloader parameters for pretraining
    pretrain_params = {'batch_size': config.pre_batch_size,
                       'num_workers': config.pre_n_workers,
                       'pin_memory': True}
    train_params = {'batch_size': config.batch_size,
                       'num_workers': config.n_workers,
                       'pin_memory': True}

    # Setup data loaders for pretraining
    datasets_pre = {}
    dataloaders_pre = {}
    for subset in [test_name]:
        datasets_pre[subset] = PSG_pretrain_Dataset(config, subset, n=-1)
        dataloaders_pre[subset] = data.DataLoader(datasets_pre[subset],
                                              shuffle=False,
                                              **pretrain_params)
    
    # Initialize network
    model_F = M_PSG2FEAT(config).to(device)

    # Pretraining and evaluation
    config.save_dir = config.model_F_path
    loss_fn = age_loss(device, config.loss_func)
    pretrainer = PreTrainer(model_F, loss_fn, config, writer,
                 device=device,
                 num_epochs=config.pre_max_epochs,
                 patience=config.pre_patience,
                 resume=True)

    # pretesting
    metrics, predictions = pretrainer.evaluate_performance(dataloaders_pre[test_name], len(dataloaders_pre[test_name]))
    print('\nFinished evaluation')
    metrics.to_csv(os.path.join(config.model_dir, 'metrics_pre_' + test_name + '.csv'))
    with open(os.path.join(config.model_dir, 'predictions_pre_' + test_name + '.json'), 'w') as fp:
        json.dump(predictions, fp, sort_keys=True, indent=4, cls=NumpyEncoder)

    # store features and predictions
    # Setup data loaders for savefeat
    if config.pre_only_sleep == 1:
        config.pre_only_sleep = 0
        datasets_pre = {}
        dataloaders_pre = {}
        for subset in [test_name]:
            datasets_pre[subset] = PSG_pretrain_Dataset(config, subset, n=-1)
            dataloaders_pre[subset] = data.DataLoader(datasets_pre[subset],
                                                shuffle=False,
                                                **pretrain_params)

    pretrainer.save_features(dataloaders_pre)
    
    # Setup data loaders for training
    datasets = {}
    dataloaders = {}
    for subset in [test_name]:
        datasets[subset] = PSG_feature_Dataset(config, subset, n=-1)
        dataloaders[subset] = data.DataLoader(datasets[subset],
                                              shuffle=False,
                                              **train_params)

    # Training and evaluation
    config.save_dir = config.model_L_path
    loss_fn = age_loss(device, config.loss_func)

    # Initiate network
    model_L = M_FEAT2LABEL(config).to(device)

    # Initiate trainer
    trainer = Trainer(model_L, loss_fn, config, writer,
                 device=device,
                 num_epochs=config.max_epochs,
                 patience=config.patience,
                 resume=True)

    # testing
    metrics, predictions = trainer.evaluate_performance(dataloaders[test_name], len(dataloaders[test_name]))
    print('\nFinished evaluation')
    metrics.to_csv(os.path.join(config.model_dir, 'metrics_' + test_name + '.csv'))
    with open(os.path.join(config.model_dir, 'predictions_' + test_name + '.json'), 'w') as fp:
        json.dump(predictions, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
    
    # Close writer
    writer.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate Biological Age from Polysomnograms.')
    parser.add_argument('--input_folder', type=str, default='H:\\nAge\\test_mortality',
                        help='folder with edf files to preprocess.')
    parser.add_argument('--pre_hyperparam', nargs=9, type=float, default=[1e-3, 1e-5, 0.75, 0.1, 1, 32, 5, 0, 1],
                        help='Pretraining hyperparameters [learning rate, l2, dropout features, dropout channels, loss function, batch size, sequence length, only sleep data, only eeg].')
    parser.add_argument('--model_name', type=str, default='eeg5',
                        help='folder with edf files to preprocess.')
    args = parser.parse_args()
    print(args)
    main(args)
