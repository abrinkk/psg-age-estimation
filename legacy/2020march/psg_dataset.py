import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import h5py

class PSG_Dataset(Dataset):

    def __init__(self, config, mode, return_filename = False):
        self.config = config
        self.mode = mode
        self.filepath = os.path.join(self.config.data_dir, mode)
        self.filenames = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        self.label_name = config.pre_label
        self.label_cond_name = config.label_cond
        self.epoch_size = config.epoch_size
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.return_filename = return_filename

    def __len__(self):
        return len(self.filenames)

    def load_h5(self, filename):
        with h5py.File(filename, "r") as f:
            data = np.array(f['PSG'])
            label = np.concatenate([np.array(f.attrs[i], ndmin=1) for i in self.label_name]).astype(np.float32)
            label_cond = np.concatenate([np.array(f.attrs[i], ndmin=1) for i in self.label_cond_name]).astype(np.float32)
        return data, label, label_cond

    def one_hot_labels(self, y):
        one_hot_y = np.eye(self.n_class)[int(y - 1)]
        return one_hot_y

    def reshape2epochs(self,data):
        # Number of epochs
        N_epochs = data.shape[1] // self.epoch_size
        # Remove excess to fit epochs
        data = data[:,:self.epoch_size*N_epochs]
        # Reshape to [epochs, channels, time]
        data = np.stack(np.split(data, N_epochs, 1), axis = 0)
        return data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = os.path.join(self.filepath, self.filenames[idx])
        data, label, label_cond = self.load_h5(filename)
        data = self.reshape2epochs(data)
        #label = int(label - 1)
        if not self.return_filename:
            return data, label, label_cond
        else:
            return data, label, label_cond, filename

class PSG_epoch_Dataset(Dataset):

    def __init__(self, config, data, label, label_cond):
        self.config = config
        self.label_name = config.pre_label
        self.label_cond_name = config.label_cond
        self.epoch_size = config.epoch_size
        self.n_channels = config.n_channels
        self.data = data
        self.label = label
        self.label_cond = label_cond

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        epochs = self.data[idx]
        label_e = self.label
        label_cond_e = self.label_cond
        return epochs, label_e, label_cond_e
    

class PSG_pretrain_Dataset(Dataset):
    
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.filepath = os.path.join(self.config.data_dir, mode)
        self.filenames = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        self.label_name = config.pre_label
        self.label_cond_name = config.label_cond
        self.epoch_size = config.epoch_size
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.T_epochs, self.f_order, self.f_e_order = self.get_epoch_order()
        self.lr_finder = False
        self.augmentation = config.pre_epoch_augmentation
        self.channel_drop = config.pre_channel_drop
        self.channel_drop_p = config.pre_channel_drop_prob

    def get_epoch_order(self):
        # Get number of epochs
        N_epochs = []
        for file in self.filenames:
            filename = os.path.join(self.filepath,file)
            with h5py.File(filename, "r") as f:
                N_epochs.append((f['PSG'].shape[1] // self.epoch_size))
        # Total epochs
        T_epochs = np.sum(N_epochs)
        
        # Cumulative sum of epochs
        C_epochs = np.cumsum(N_epochs)
        
        # Overall order of epochs
        epoch_order = np.random.permutation(T_epochs)
        epoch_order_0 = np.concatenate(([0],C_epochs))
        
        # File idx
        file_idx_order = np.array([np.where(C_epochs > x)[0][0] for x in epoch_order])
        filename_order = [self.filenames[x] for x in file_idx_order]
        
        # Epoch order in file
        file_epoch_order = np.array([epoch_order[x] - epoch_order_0[file_idx_order[x]] for x in range(T_epochs)])
        
        return T_epochs, filename_order, file_epoch_order
        
    def __len__(self):
        return int(self.T_epochs)

    def load_h5(self, filename, e_idx):
        with h5py.File(filename,"r") as f:
            # Extract data chunk
            if self.augmentation:
                noise_idx = np.random.randint(-self.epoch_size / 2, self.epoch_size / 2)
                if e_idx*self.epoch_size + noise_idx < 0 or (e_idx + 1)*self.epoch_size + noise_idx >= f['PSG'].shape[1]:
                    noise_idx = 0
                data = np.array(f['PSG'][:, noise_idx + e_idx*self.epoch_size:noise_idx + (e_idx + 1)*self.epoch_size])
            else:
                data = np.array(f['PSG'][:, e_idx*self.epoch_size:(e_idx + 1)*self.epoch_size])
            label = np.concatenate([np.array(f.attrs[i], ndmin=1) for i in self.label_name]).astype(np.float32)
            label_cond = np.concatenate([np.array(f.attrs[i], ndmin=1) for i in self.label_cond_name]).astype(np.float32)
        return data, label, label_cond

    def one_hot_labels(self, y):
        one_hot_y = np.eye(self.n_class)[int(y - 1)]
        return one_hot_y

    def drop_channel(self, data):
        drop_idx = np.random.rand(data.shape[0]) < self.channel_drop_p
        data[drop_idx] = 0.0
        data = data / (1 - self.channel_drop_p)
        return data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.f_order[idx]
        e_idx = self.f_e_order[idx]
        data, label, label_cond = self.load_h5(os.path.join(self.filepath, filename), e_idx)
        # Drop Channels
        data = self.drop_channel(data)
        # Transform to classification label
        #label = int(label - 1)
        if self.lr_finder:
            return data, label
        return data, label, label_cond

class PSG_feature_Dataset(Dataset):
    
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.filepath = os.path.join(self.config.data_dir, mode)
        self.filenames = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        self.label_name = config.label
        self.label_cond_name = config.label_cond
        self.label_cond_size = config.label_cond_size
        self.n_class = config.n_class
        self.pad_length = config.pad_length
        self.pad_dim = 0
        self.cond_drop_p = config.cond_drop_prob
        self.cond_label_scale = {'q_low': 1.0, 'q_high': 1.0, 'age': 88.5, 'bmi': 84.8, 'sex': 1.0}

    def __len__(self):
        return len(self.filenames)

    def load_h5(self, filename):
        with h5py.File(filename, "r") as f:
            data = np.array(f['PSG'])
            label = np.concatenate([np.array(f.attrs[i], ndmin=1) for i in self.label_name]).astype(np.float32)
            label_cond = np.concatenate([np.array(f.attrs[i], ndmin=1) for i in self.label_cond_name]).astype(np.float32)
        return data, label, label_cond

    def drop_cond_label(self, label_cond):
        drop_idx = np.random.rand(label_cond.shape[0]) < self.cond_drop_p
        label_cond[drop_idx] = 0.0
        label_cond = label_cond / (1 - self.cond_drop_p)
        return label_cond

    def scale_cond_labels(self, label_cond):
        cond_idx_start = 0
        for idx, i in enumerate(self.label_cond_name):
            cond_idx_end = cond_idx_start + self.label_cond_size[idx]
            label_cond[cond_idx_start:cond_idx_end] /= self.cond_label_scale[i]
            cond_idx_start = cond_idx_end
        return label_cond

    def pad_psg_feat(self, data):
        pad_size = list(data.shape)
        pad_size[self.pad_dim] = self.pad_length - data.shape[self.pad_dim]
        if self.pad_length < data.shape[self.pad_dim]:
            if self.pad_dim == 0:
                padded_data = data[:self.pad_length]
            else:
                padded_data = data
        elif self.pad_length > data.shape[self.pad_dim]:
            if self.pad_dim == 0:
                padded_data = np.concatenate((data, np.zeros(tuple(pad_size))), axis = self.pad_dim)
            else:
                padded_data = data
        else:
            padded_data = data
            
        return padded_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = os.path.join(self.filepath,self.filenames[idx])
        data, label, label_cond = self.load_h5(filename)
        data = self.pad_psg_feat(data)
        label_cond = self.scale_cond_labels(label_cond)
        if self.mode == 'train':
            label_cond = self.drop_cond_label(label_cond)
        return data.astype(np.float32), label.astype(np.float32), label_cond.astype(np.float32)