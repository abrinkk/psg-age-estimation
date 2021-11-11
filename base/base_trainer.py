import copy
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class BaseTrainer(object):
    def __init__(self, network, loss_fn, config,
                 device='cpu',
                 num_epochs=100,
                 patience=None,
                 resume=None,
                 scheduler=None):
        """Base class for network trainers

        Args:
            network (nn.Moduke): Neural network.
            loss_fn ([type]): a loss function
            config (Config): an instance of the config class with set attributes.
            device (str, optional): device for cpu or gpu processing. Defaults to 'cpu'.
            num_epochs (int, optional): maximum number of epochs. Defaults to 100.
            patience (int, optional): early stopping patience. Defaults to None.
            resume (boolean, optional): to resume existing checkpoint. Defaults to None.
            scheduler (None, optional): Learning rate scheduler (not used). Defaults to None.
        """
        super().__init__()
        self.config = config
        self.network = network
        self.loss_fn = loss_fn
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience if patience else self.num_epochs
        self.resume = resume
        self.save_dir = config.save_dir
        self.scheduler = scheduler
        self.l2 = config.l2
        self.lr = config.lr

        if self.resume:
            self.checkpoint = torch.load(os.path.join(self.save_dir, 'latest_checkpoint.tar'))
            self.network.load_state_dict(self.checkpoint['network_state_dict'])

        # Move network to device and initialize optimizer
        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(
            [{'params': [p for name, p in self.network.named_parameters() if 'weight' in name], 'weight_decay': self.l2},
            {'params': [p for name, p in self.network.named_parameters() if 'weight' not in name], 'weight_decay':0}],
            lr=self.lr)

        if self.resume:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        self.train_losses = []
        self.eval_losses = []

    def on_begin_epoch(self):
        """Callback function for the start of each training epoch
        """
        print(f'\nEpoch nr. {self.current_epoch + 1} / {self.num_epochs}')

    def on_end_epoch(self):
        """Callback function for end of each training epoch
        """

        if self.eval_losses[-1] < self.best_loss:
            self.best_loss = self.eval_losses[-1]
            self.best_network = copy.deepcopy(self.network)
            self.last_update = self.current_epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.save_dir:
            self.save_checkpoint(os.path.join(self.save_dir, 'latest_checkpoint.tar'))

    def on_begin_training(self):
        """Callback function for start of training
        """
        self.network.train()

    def on_begin_validation(self):
        """Callback function for start of validation
        """
        self.network.eval()

    def on_begin(self):
        """Callback function for start of training and validation
        """
        self.best_loss = np.inf
        self.best_network = None
        self.patience_counter = 0
        self.current_epoch = 0
        self.last_update = None
        self.start_epoch = 0

    def on_end(self):
        """Callback function for end of training and validation
        """
        self.on_end_epoch()
        pd.DataFrame({'train': self.train_losses, 'eval': self.eval_losses}).to_csv(os.path.join(self.save_dir, 'history.csv'))

    def save_checkpoint(self, checkpoint_path):
        """Saves checkpoints

        Args:
            checkpoint_path (str): Path to checkpoint
        """
        checkpoint = {'best_loss': self.best_loss,
                      'best_network': self.best_network,
                      'start_epoch': self.current_epoch,
                      'last_update': self.last_update,
                      'network_state_dict': self.network.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'patience_counter': self.patience_counter,
                      'train_losses': self.train_losses,
                      'eval_losses': self.eval_losses}
        torch.save(checkpoint, checkpoint_path)

    def restore_checkpoint(self):
        """Restores checkpoint
        """
        self.best_loss = self.checkpoint['best_loss']
        self.best_network = self.checkpoint['best_network']
        self.patience_counter = self.checkpoint['patience_counter']
        self.start_epoch = self.checkpoint['start_epoch'] + 1
        self.last_update = self.checkpoint['last_update']
        self.train_losses = self.checkpoint['train_losses']
        self.eval_losses = self.checkpoint['eval_losses']

    def train_and_validate(self):
        """Function to implement for training and validation

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def train_step(self):
        """Function to implement to process a batch during training

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def validate_step(self):
        """Function to implement to process a batch during validation

        Raises:
            NotImplementedError
        """
        raise NotImplementedError