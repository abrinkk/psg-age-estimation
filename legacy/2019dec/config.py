import os
import numpy as np

class Config(object):
    def __init__(self, hyperparameters = {'lr': 0.0001}):
        
        # Datapaths
        self.data_dir = 'H:\\nAge\\'
        self.model_dir = os.path.join(self.data_dir, 'model')
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.test_dir = os.path.join(self.data_dir,'test')
        self.train_F_dir = os.path.join(self.data_dir, 'train_F')
        self.val_F_dir = os.path.join(self.data_dir, 'val_F')
        self.test_F_dir = os.path.join(self.data_dir,'test_F')
        
        # Checkpoint
        self.model_F_path = os.path.join(self.model_dir,'modelF')
        self.model_L_path = os.path.join(self.model_dir,'modelL')
        
        # Target labels
        self.label = ['age','bmi','sex']
        
        # Network
        self.n_channels = 12
        self.n_class = 1 + 1 + 2
        self.epoch_size = 5*60*128
        
        # Pretraining
        self.pre_max_epochs = 20
        self.pre_batch_size = 32
        self.pre_lr = 5e-4
        self.pre_n_workers = 0
        
        # Training
        self.max_epochs = 200
        self.batch_size = 64
        self.lr = 5e-4
        self.n_workers = 0
        self.pad_length = 120
        self.train_label_idx = 0

