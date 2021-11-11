import argparse
import collections
import time

import numpy as np
import os
import json

import torch
import torch.nn as nn
import torch.optim as module_optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from config import Config
from pretrainer import PreTrainer
from m_psg2label import M_PSG2FEAT, M_FEAT2LABEL
from psg_dataset import PSG_pretrain_Dataset
from utils.utils_loss import multi_label_loss

np.random.seed(0)
torch.manual_seed(0)

def main(args):

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Config
    config = Config()

    # Tensorboard writer
    writer_name = 'runs/nAge_' + time.strftime("%Y%m%d_%H%M%S")
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
        datasets_pre[subset] = PSG_pretrain_Dataset(config, subset)
        dataloaders_pre[subset] = data.DataLoader(datasets_pre[subset],
                                              shuffle=True if subset == 'train' else False,
                                              **pretrain_params)
    
    # Initialize networks
    model_F = M_PSG2FEAT(config).to(device)
    model_L = M_FEAT2LABEL(config).to(device)

    # Pretraining and evaluation
    config.save_dir = config.model_F_path
    loss_fn = multi_label_loss(device)
    pretrainer = PreTrainer(model_F, loss_fn, config, writer,
                 device=device,
                 num_epochs=config.pre_max_epochs,
                 patience=config.pre_patience,
                 resume=args.pre_train_resume)
    # pretraining
    if args.pre_train:
        pretrainer.train_and_validate(dataloaders_pre['train'], dataloaders_pre['val'])
        print('\nFinished training')
        #return
    # pretesting
    if args.test_pre:
        metrics, predictions = pretrainer.evaluate_performance(dataloaders_pre['test'])
        print('\nFinished evaluation')
        metrics.to_csv(os.path.join(config.model_dir, 'metrics.csv'))
        with open(os.path.join(config.model_dir, 'predictions.json'), 'w') as fp:
            json.dump(predictions, fp, sort_keys=True, indent=4)
    # store features and predictions
    if args.save_feat:
        pretrainer.save_features(dataloaders_pre)
    
    # Setup data loaders for training
    # TODO: Do that

    # Close writer
    writer.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Age Estimation from Polysomnograms.')
    parser.add_argument('--pre_train', type=bool, default=True,
                        help='To pretrain model.')
    parser.add_argument('--pre_train_resume', type=bool, default=False,
                        help='To resume preivously pretrained model')
    parser.add_argument('--test_pre', type=bool, default=True,
                        help='To train preivously pretrained model')
    parser.add_argument('--save_feat', type=bool, default=False,
                        help='Save/overwrite model F features')
    parser.add_argument('--train', type=bool, default=False,
                        help='To train model.')
    parser.add_argument('--bo', type=bool, default=False,
                        help='To perform bayesian hyperparameter optimization.')
    parser.add_argument('--train_resume', type=bool, default=False,
                        help='To resume preivously trained model')
    parser.add_argument('--test', type=bool, default=False,
                        help='To test preivously trained model')

    args = parser.parse_args()
    main(args)
