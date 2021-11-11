import os
import numpy as np

class Config(object):
    def __init__(self):
        
        # Datapaths
        self.data_dir = 'H:\\nAge\\'
        self.model_dir = os.path.join(self.data_dir, 'model')
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.test_dir = os.path.join(self.data_dir, 'test')
        self.train_cache_dir = os.path.join(self.data_dir, 'train_cache')
        self.val_cache_dir = os.path.join(self.data_dir, 'val_cache')
        self.test_cache_dir = os.path.join(self.data_dir, 'test_cache')
        self.train_F_dir = os.path.join(self.data_dir, 'train_F')
        self.val_F_dir = os.path.join(self.data_dir, 'val_F')
        self.test_F_dir = os.path.join(self.data_dir, 'test_F')
        
        # Checkpoint
        self.save_dir = self.model_dir
        self.model_F_path = os.path.join(self.model_dir, 'modelF')
        self.model_L_path = os.path.join(self.model_dir, 'modelL')
        self.model_L_BO_path = self.model_L_path
        self.BO_expe_path = os.path.join(self.model_dir, 'exp')

        self.return_only_pred = False

        # Pretraining
        # label-config
        self.pre_label = ['age', 'bmi', 'sex']
        self.pre_label_size = [1, 1, 2]
        self.pre_n_class = sum(self.pre_label_size)
        # network-config
        self.n_channels = 12
        # train-config
        self.pre_max_epochs = 20
        self.pre_patience = 3
        self.pre_batch_size = 32
        self.pre_lr = 1e-3
        self.pre_n_workers = 0
        self.do_f = 0.75
        self.pre_epoch_augmentation = False
        self.pre_channel_drop = True
        self.pre_channel_drop_prob = 0.1

        # Training
        # label-config
        self.label = ['age']
        self.label_cond = ['q_low', 'q_high', 'bmi', 'sex']
        self.label_cond_size = [7, 7, 1, 1]
        self.n_class = 1
        # train-config
        self.do_l = 0.5
        self.max_epochs = 200
        self.batch_size = 64
        self.lr = 5e-4
        self.l2 = 1e-5
        self.n_workers = 0
        self.pad_length = 120
        self.cond_drop_prob = 0.5
        # network-config
        self.epoch_size = 5*60*128
        self.return_att_weights = False