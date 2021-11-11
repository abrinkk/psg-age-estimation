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
        self.label_name = config.label
        self.epoch_size = config.epoch_size
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.return_filename = return_filename

    def __len__(self):
        return len(self.filenames)

    def load_h5(self, filename):
        with h5py.File(filename,"r") as f:
            data = np.array(f['PSG'])
            label = float(np.array(f.attrs[self.label_name]))
        return data, label

    def one_hot_labels(self, y):
        one_hot_y = np.eye(self.n_class)[int(y - 1)]
        return one_hot_y

    def reshape2epochs(self,data):
        # Number of epochs
        N_epochs = data.shape[1] // self.epoch_size
        # Remove excess to fit epochs
        data = data[:,:self.epoch_size*N_epochs]
        # Reshape to [epochs, channels, time]
        data = np.stack(np.split(data, N_epochs, 1),axis = 0)
        return data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = os.path.join(self.filepath,self.filenames[idx])
        data, label = self.load_h5(filename)
        data = self.reshape2epochs(data)
        #label = int(label - 1)
        if not self.return_filename:
            return data, label
        else:
            return data, label, filename

class PSG_epoch_Dataset(Dataset):

    def __init__(self, config, data, label):
        self.config = config
        self.label_name = config.label
        self.epoch_size = config.epoch_size
        self.n_channels = config.n_channels
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        epochs = self.data[idx]
        label_e = self.label
        return epochs, label_e
    

class PSG_pretrain_Dataset(Dataset):
    
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.filepath = os.path.join(self.config.data_dir, mode)
        self.filenames = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        self.label_name = config.label
        self.epoch_size = config.epoch_size
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.T_epochs, self.f_order, self.f_e_order = self.get_epoch_order()

    def get_epoch_order(self):
        # Get number of epochs
        N_epochs = []
        for file in self.filenames:
            filename = os.path.join(self.filepath,file)
            with h5py.File(filename,"r") as f:
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
        file_epoch_order =  np.array([epoch_order[x] - epoch_order_0[file_idx_order[x]] for x in range(T_epochs)])
        
        return T_epochs, filename_order, file_epoch_order
        
    def __len__(self):
        return int(self.T_epochs)

    def load_h5(self, filename, e_idx):
        with h5py.File(filename,"r") as f:
            # Extract data chunk
            data = np.array(f['PSG'][:,e_idx*self.epoch_size:(e_idx + 1)*self.epoch_size])
            label = np.array(f.attrs[self.label_name])
        return data, label

    def one_hot_labels(self, y):
        one_hot_y = np.eye(self.n_class)[int(y - 1)]
        return one_hot_y

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.f_order[idx]
        e_idx = self.f_e_order[idx]
        data, label = self.load_h5(os.path.join(self.filepath,filename),e_idx)
        # Transform to classification label
        #label = int(label - 1)
        return data, label

class PSG_feature_Dataset(Dataset):
    
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.filepath = os.path.join(self.config.data_dir, mode)
        self.filenames = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        self.label_name = config.label
        self.n_class = config.n_class
        self.pad_length = config.pad_length
        self.pad_dim = 0

    def __len__(self):
        return len(self.filenames)

    def load_h5(self, filename):
        with h5py.File(filename,"r") as f:
            data = np.array(f['PSG'])
            label = np.array(f.attrs[self.label_name])
        return data, label

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
        data, label = self.load_h5(filename)
        data = self.pad_psg_feat(data)
        #label = int(label - 1)
        return data.astype(np.float32), label.astype(np.float32)







