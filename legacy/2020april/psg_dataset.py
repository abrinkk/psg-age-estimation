import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import h5py
import copy
import time

from joblib import Memory
from joblib import delayed
from tqdm import tqdm
from utils.get_h5_data import get_h5_data
from utils.parallel_bar import ParallelExecutor
from config import Config

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
        self.cachepath = os.path.join(self.config.data_dir,mode + '_cache')
        self.filenames = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        self.num_records = len(self.filenames)
        self.label_name = config.pre_label
        self.label_cond_name = config.label_cond
        self.epoch_size = config.epoch_size
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.lr_finder = False
        self.augmentation = config.pre_epoch_augmentation
        self.channel_drop = config.pre_channel_drop
        self.channel_drop_p = config.pre_channel_drop_prob

        # Generate memory maps
        self.cache_data = True
        self.n_jobs = 4
        self.psgs = {}
        self.attrs = {}
        get_data = get_h5_data
        if self.cache_data:
            memory = Memory(self.cachepath, mmap_mode='r', verbose=0)
            get_data = memory.cache(get_h5_data)
        print(f'Number of recordings: {self.num_records}')
        data = ParallelExecutor(n_jobs=self.n_jobs, prefer='threads')(total=len(self.filenames))(delayed(get_data)(
            filename=os.path.join(self.filepath, record)) for record in self.filenames
            )
        for record, (psg, attrs) in zip(self.filenames, data):
            self.psgs[record] = {'data': psg,
                                'length': psg.shape[1],
                                'reduced_length': int(psg.shape[1] // self.epoch_size)}
            self.attrs[record] = attrs

        self.indexes = [((i, record), (j * self.epoch_size, self.epoch_size)) for i, record in enumerate(self.psgs.keys())
                        for j in np.arange(self.psgs[record]['reduced_length'])]

    def __len__(self):
        return len(self.indexes)

    def drop_channel(self, data):
        drop_idx = np.random.rand(data.shape[0]) < self.channel_drop_p
        data[drop_idx] = 0.0
        data = data / (1 - self.channel_drop_p)
        return data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get record ID and epoch position
        record = self.indexes[idx][0][1]
        position = self.indexes[idx][1][0] + range(self.indexes[idx][1][1])
        # Extract data and attribures
        data = self.psgs[record]['data'][:, position]
        attrs = self.attrs[record]
        # Select labels and conditioning labels
        label = np.concatenate([np.array(attrs[i], ndmin=1) for i in self.label_name]).astype(np.float32)
        label_cond = np.concatenate([np.array(attrs[i], ndmin=1) for i in self.label_cond_name]).astype(np.float32)
        # Drop Channels
        #data = self.drop_channel(copy.deepcopy(data))
        # Output dict
        out = {'fid': record,
                'position': position,
                'data': torch.from_numpy(data),
                'label': torch.from_numpy(label),
                'label_cond': torch.from_numpy(label_cond)}
        return out
                

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

if __name__ == '__main__':
    # Set up config
    config = Config()
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets
    ds = PSG_pretrain_Dataset(config, 'train')

    # Create dataloader
    dl = DataLoader(ds, shuffle=False, batch_size=32, num_workers=0, pin_memory=False)

    # data bar
    bar_data = tqdm(dl, total=len(dl), desc=f'Loss: {np.inf:.04f}')

    # Iterate datasets
    for i in range(10):
        time_start = time.time()
        batch = next(iter(bar_data))
        print('')
        print('Batch time: {:.3f}'.format(time.time() - time_start))
    print(f'Recording: {batch["fid"]}')
    print(f'Position in recording: {batch["position"]}')
    print(f'Data: {batch["data"].shape}')
    print(f'Labels:{batch["label"].shape}')
    print(f'Conditioning labels:{batch["label_cond"].shape}')
