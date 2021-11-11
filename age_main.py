import argparse
import time
from utils.str2bool import str2bool

import os
import json
import numpy as np
import pandas as pd

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

# Seeds for replication
np.random.seed(0)
torch.manual_seed(0)

def main(args):
    """trains and tests age estimation models

    Args:
        args (dict): See age_main.py -h
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Config
    config = Config()

    # Adjust config for hyperparameters
    hyper_param_string = 'lr_{:.7f}_l2_{:.7f}_dof_{:.3f}_doc_{:.3f}_lf_{:.0f}_bs_{:.0f}_sl_{:.0f}_sd_{:.0f}_eeg_{:.0f}_'.format(args.pre_hyperparam[0], args.pre_hyperparam[1], args.pre_hyperparam[2], args.pre_hyperparam[3], args.pre_hyperparam[4], args.pre_hyperparam[5], args.pre_hyperparam[6], args.pre_hyperparam[7], args.pre_hyperparam[8])
    # Learning rate for pretraining
    config.pre_lr = args.pre_hyperparam[0]
    # L2 regularization
    config.l2 = args.pre_hyperparam[1]
    # Drop rate for latent space
    config.do_f = args.pre_hyperparam[2]
    # Drop rate for input channels
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
    for subset in ['train', 'val', 'test']:
        datasets_pre[subset] = PSG_pretrain_Dataset(config, subset, n=-1 if subset == 'train' else -1)
        dataloaders_pre[subset] = data.DataLoader(datasets_pre[subset],
                                              shuffle=True if subset == 'train' else False,
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
                 resume=args.pre_train_resume)

    # pretraining
    if args.pre_train:
        pretrainer.train_and_validate(dataloaders_pre['train'], dataloaders_pre['val'])
        print('\nFinished training')
        
    # pretesting
    if args.test_pre:
        metrics, predictions = pretrainer.evaluate_performance(dataloaders_pre['test'], len(dataloaders_pre['train']))
        print('\nFinished evaluation')
        metrics.to_csv(os.path.join(config.model_dir, 'metrics_pre.csv'))
        with open(os.path.join(config.model_dir, 'predictions_pre.json'), 'w') as fp:
            json.dump(predictions, fp, sort_keys=True, indent=4, cls=NumpyEncoder)

    # store features and predictions
    if args.save_feat:
        # Setup data loaders for savefeat
        if config.pre_only_sleep == 1:
            config.pre_only_sleep = 0
            datasets_pre = {}
            dataloaders_pre = {}
            for subset in ['train', 'val', 'test']:
                datasets_pre[subset] = PSG_pretrain_Dataset(config, subset, n=-1 if subset == 'train' else -1)
                dataloaders_pre[subset] = data.DataLoader(datasets_pre[subset],
                                                    shuffle=True if subset == 'train' else False,
                                                    **pretrain_params)

        pretrainer.save_features(dataloaders_pre)
    
    # Setup data loaders for training
    datasets = {}
    dataloaders = {}
    for subset in ['train', 'val', 'test']:
        datasets[subset] = PSG_feature_Dataset(config, subset, n=-1 if subset == 'train' else -1)
        dataloaders[subset] = data.DataLoader(datasets[subset],
                                              shuffle=True if subset == 'train' else False,
                                              **train_params)

    # Training and evaluation
    config.save_dir = config.model_L_path
    loss_fn = age_loss(device, config.loss_func)

    # training
    if args.train and args.bo:
        bo_params = bayesian_opt_eval(args, config, M_FEAT2LABEL, loss_fn, dataloaders, device, writer_name)
        config.do_l = bo_params['do']
        config.lr = bo_params['lr']
        config.l2 = bo_params['l2']
        config.net_size_scale = bo_params['size']
        config.lstm_n = bo_params['lstmn']
        config.save_dir = config.model_L_path
        print('Optimal hyperparameters: ', bo_params)
    
    # Initiate network
    model_L = M_FEAT2LABEL(config).to(device)

    # Initiate trainer
    trainer = Trainer(model_L, loss_fn, config, writer,
                 device=device,
                 num_epochs=config.max_epochs,
                 patience=config.patience,
                 resume=args.train_resume)

    if args.train:
        trainer.train_and_validate(dataloaders['train'], dataloaders['val'])
        print('\nFinished training')
        #return
    
    # testing
    if args.test:
        metrics, predictions = trainer.evaluate_performance(dataloaders['test'], len(dataloaders['train']))
        print('\nFinished evaluation')
        metrics.to_csv(os.path.join(config.model_dir, 'metrics.csv'))
        with open(os.path.join(config.model_dir, 'predictions.json'), 'w') as fp:
            json.dump(predictions, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
    
    # Close writer
    writer.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Age Estimation from Polysomnograms.')
    parser.add_argument('--pre_train', type=str2bool, default=True, nargs='?',
                        const=True, help='To pretrain model.')
    parser.add_argument('--pre_hyperparam', nargs=9, type=float, default=[1e-3, 1e-5, 0.75, 0.1, 1, 32, 5, 0, 1],
                        help='Pretraining hyperparameters [learning rate, l2, dropout features, dropout channels, loss function, batch size, sequence length, only sleep data, only eeg].')
    parser.add_argument('--pre_train_resume', type=str2bool, default=False, nargs='?',
                        const=True, help='To resume preivously pretrained model')
    parser.add_argument('--test_pre', type=str2bool, default=True, nargs='?',
                        const=True, help='To train preivously pretrained model')
    parser.add_argument('--save_feat', type=str2bool, default=True, nargs='?',
                        const=True, help='Save/overwrite model F features')
    parser.add_argument('--train', type=str2bool, default=True, nargs='?',
                        const=True, help='To train model.')
    parser.add_argument('--bo', type=str2bool, default=False, nargs='?',
                        const=True, help='To perform bayesian hyperparameter optimization.')
    parser.add_argument('--train_resume', type=str2bool, default=False, nargs='?',
                        const=True, help='To resume preivously trained model')
    parser.add_argument('--test', type=str2bool, default=True, nargs='?',
                        const=True, help='To test preivously trained model')

    args = parser.parse_args()
    print(args)
    main(args)
