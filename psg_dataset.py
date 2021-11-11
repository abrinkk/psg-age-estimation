import os
from collections.abc import Iterable
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import time
import pandas as pd

from joblib import Memory
from joblib import delayed
from tqdm import tqdm
from utils.get_h5_data import get_h5_size, get_h5_ssc
from utils.parallel_bar import ParallelExecutor
from config import Config

np.random.seed(0)
torch.manual_seed(0)

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
    def __init__(self, config, mode, n=-1):
        """Dataset for pretraining of age estimation models

        Args:
            config (Config): An instance of the config class with set attributes.
            mode (str): An option {'train', 'val', 'test', 'test_mortality','other'} to control data read
            n (int, optional): Number of files to read. To read all choose: n = -1. Defaults to -1.
        """
        self.config = config
        self.mode = mode
        # self.filepath = os.path.join(self.config.data_dir, mode)
        self.cachepath = os.path.join(self.config.data_dir, mode + '_cache')
        self.filepath = os.path.join(self.config.pretrain_dir)
        self.filenames_all = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        # Read in lists
        self.estimate_mode = False
        if mode == 'train':
            df_pretrain = pd.read_csv(self.config.list_split_train, delimiter=',')
        elif mode == 'val':
            df_pretrain = pd.read_csv(self.config.list_split_val, delimiter=',')
        elif mode == 'test':
            df_pretrain = pd.read_csv(self.config.list_split_test, delimiter=',')
        elif mode == 'test_mortality':
            df_pretrain_1 = pd.read_csv(self.config.list_split_train, delimiter=',')
            df_pretrain_2 = pd.read_csv(self.config.list_split_val, delimiter=',')
            df_pretrain = pd.concat([df_pretrain_1, df_pretrain_2])
            df_pretrain['names'] = df_pretrain['names'].str.replace('_EDFAndScore', '')
        else:
            self.estimate_mode = True
        if not self.estimate_mode:
            list_pretrain = list(df_pretrain.iloc[:, 4])
            list_pretrain = [x.lower() for x in list_pretrain]
        # Exclude files not in list
        self.filenames = []
        for f in self.filenames_all:
            if self.estimate_mode:
                self.filenames.append(f)
            else:
                filename_base = os.path.basename(f)[:-5]
                # Assumes only WSC data has filename of length 14
                # TODO: streamline list names to fit exported names
                if len(filename_base) == 14:
                    filelistname = filename_base[:7].lower()
                else:
                    filelistname = filename_base.lower()
                if filelistname in list_pretrain:
                    self.filenames.append(f)

        
        if (n > 0):
            self.filenames = self.filenames[:n]
        self.num_records = len(self.filenames)
        self.label_name = config.pre_label
        self.label_cond_name = config.label_cond
        self.epoch_size = config.epoch_size
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.lr_finder = False
        self.channel_drop = config.pre_channel_drop
        self.channel_drop_p = config.pre_channel_drop_prob
        self.only_sleep_data = config.pre_only_sleep
        self.only_eeg = config.only_eeg

        # Generate memory maps
        self.cache_data = False
        self.n_jobs = 4
        self.psgs = {}
        self.attrs = {}
        self.hyp = {}
        get_data = get_h5_size
        get_ssc = get_h5_ssc
        if self.cache_data:
            memory = Memory(self.cachepath, mmap_mode='r', verbose=0)
            get_data = memory.cache(get_h5_size)
        print(f'Number of recordings: {self.num_records}')
        data = ParallelExecutor(n_jobs=self.n_jobs, prefer='threads')(total=len(self.filenames))(delayed(get_data)(
            filename=os.path.join(self.filepath, record)) for record in self.filenames
            )
        ssc = ParallelExecutor(n_jobs=self.n_jobs, prefer='threads')(total=len(self.filenames))(delayed(get_ssc)(
            filename=os.path.join(self.filepath, record)) for record in self.filenames
            )
        for record, (data_size, attrs), hyp in zip(self.filenames, data, ssc):
            self.psgs[record] = {'length': data_size,
                                 'reduced_length': int(data_size // self.epoch_size)}
            self.attrs[record] = attrs
            self.hyp[record] = hyp

        if self.only_sleep_data == 1:
            self.indexes = []
            for i, record in enumerate(self.psgs.keys()):
                for j in np.arange(self.psgs[record]['reduced_length']):
                    # Include if less or equal to 2 wake periods
                    ssc_idx = [j * (self.epoch_size // (128*30)), (j + 1) * (self.epoch_size // (128*30))]
                    if len(self.hyp[record]) >= ssc_idx[1]:
                        ssc_batch = self.hyp[record][ssc_idx[0]:ssc_idx[1]]
                        if sum(ssc_batch == 1) <= 5 or self.mode == 'save_feat':
                            self.indexes.append(((i, record), (j * self.epoch_size, (j + 1) * self.epoch_size)))
        else:
            self.indexes = [((i, record), (j * self.epoch_size, (j + 1) * self.epoch_size)) for i, record in enumerate(self.psgs.keys())
                        for j in np.arange(self.psgs[record]['reduced_length'])]
        
        self.check_h5_quality()

    def check_h5_quality(self):
        """This iterates all recordings in attrs and checks that keys
            1: Are not missing
            2: Are the correct data type
            3: Are not nan
        """
        has_error = False
        for idx, record in enumerate(self.attrs):
            if idx == 0:
                base_attrs = self.attrs[record]
            else:
                for key in base_attrs:
                    if key not in self.attrs[record]:
                        print('Error: missing key. Record: ', record)
                        has_error = True
                    elif type(base_attrs[key]) != type(self.attrs[record][key]):
                        print('Error: wrong data type. Key: ', key, '. Record: ', record)
                        has_error = True
                    if isinstance(self.attrs[record][key], Iterable):
                        if any(self.attrs[record][key] != self.attrs[record][key]):
                            print('Error: nan data type. Key: ', key, '. Record: ', record)
                            has_error = True
                    else:
                        if self.attrs[record][key] != self.attrs[record][key]:
                            print('Error: nan data type. Key: ', key, '. Record: ', record)
                            has_error = True
                
        if has_error:
            print('Error.')
        else:
            print('Data quality ensured.')
        return

    def load_h5(self, filename, position):
        """Loads an epoch of polysomnography from an h5 file

        Args:
            filename (str): filename of h5 file
            position (list[int]): indicies of epoch to extract

        Returns:
            data (numpy.array): polysomnography data of size [n_channels, epoch_size]
        """
        with h5py.File(os.path.join(self.filepath, filename),"r", rdcc_nbytes=100*1024**2) as f:
            # Extract data chunk
            if self.only_eeg == 1:
                # C3-A2, C4-A1
                data = np.array(f['PSG'][:self.n_channels, position[0]:position[1]])
            elif self.only_eeg == 2:
                # C3-A2, C4-A1, L-EOG, R-EOG, Chin EMG
                data = np.array(f['PSG'][[0, 1, 2, 3, 5], position[0]:position[1]])
            elif self.only_eeg == 3:
                # ECG
                data = np.array(f['PSG'][4:5, position[0]:position[1]])
            elif self.only_eeg == 4:
                # Airflow, nasal pressure, abd, thorax, SaO2
                data = np.array(f['PSG'][-self.n_channels:, position[0]:position[1]])
            else:
                data = np.array(f['PSG'][:, position[0]:position[1]])
        return data

    def __len__(self):
        """Get length of dataset iterator

        Returns:
            len (size): length of dataset iterator
        """
        return len(self.indexes)

    def drop_channel(self, data):
        """This drops input channels at random during training

        Args:
            data (numpy.array): polysomnography epoch

        Returns:
            data (numpy.array): polysomnography epoch with droppped channels
        """
        if self.channel_drop and self.mode == 'train':
            drop_idx = np.random.rand(data.shape[0]) < self.channel_drop_p
            data[drop_idx] = 0.0
            data = data / (1.0 - self.channel_drop_p)
        return data

    def __getitem__(self, idx):
        """Gets a single batch element from the dataset

        Args:
            idx (int): index to read

        Returns:
            out (dict): a dict {'fid': filename, 
                                'position': position of epoch in file,
                                'data': polysomnography epoch,
                                'label': age,
                                'label_cond': extra conditional labels to input,
                                'all_attrs': all attributes in dict}
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get record ID and epoch position
        record = self.indexes[idx][0][1]
        position = [self.indexes[idx][1][0], self.indexes[idx][1][1]]
        # Extract data and attribures
        data = self.load_h5(record, position)
        attrs = self.attrs[record]
        # Select labels and conditioning labels
        label = np.concatenate([np.array(attrs[i], ndmin=1) for i in self.label_name]).astype(np.float32)
        label_cond = [np.array(attrs[i], ndmin=1) for i in self.label_cond_name]
        label_cond = np.array(label_cond) if not label_cond else np.concatenate(label_cond).astype(np.float32)
        # Drop Channels
        data = self.drop_channel(data)
        # Output dict
        out = {'fid': record,
                'position': position,
                'data': torch.from_numpy(data.astype(np.float32)),
                'label': torch.from_numpy(label.astype(np.float32)),
                'label_cond': torch.from_numpy(label_cond.astype(np.float32)),
                'all_attrs': attrs}
        return out
                

class PSG_feature_Dataset(Dataset):
    def __init__(self, config, mode, n=-1):
        """Dataset for pretraining of age estimation models

        Args:
            config (Config): An instance of the config class with set attributes.
            mode (str): An option {'train', 'val', 'test', 'test_mortality','other'} to control data read
            n (int, optional): Number of files to read. To read all choose: n = -1. Defaults to -1.
        """
        self.config = config
        self.mode = mode
        # self.filepath = os.path.join(self.config.data_dir, mode + '_F')
        # self.filenames = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        self.filepath = os.path.join(self.config.F_train_dir)
        self.filenames_all = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        # Read in lists
        self.estimate_mode = False
        if mode == 'train':
            df_train = pd.read_csv(self.config.list_split_train, delimiter=',')
        elif mode == 'val':
            df_train = pd.read_csv(self.config.list_split_val, delimiter=',')
        elif mode == 'test':
            df_train = pd.read_csv(self.config.list_split_test, delimiter=',')
        elif mode == 'test_mortality':
            df_train_1 = pd.read_csv(self.config.list_split_train, delimiter=',')
            df_train_2 = pd.read_csv(self.config.list_split_val, delimiter=',')
            df_train = pd.concat([df_train_1, df_train_2])
            df_train['names'] = df_train['names'].str.replace('_EDFAndScore', '')
        else:
            self.estimate_mode = True
        if not self.estimate_mode:
            list_train = list(df_train.iloc[:, 4])
            list_train = [x.lower() for x in list_train]
        # Exclude files not in lsit
        self.filenames = []
        for f in self.filenames_all:
            if self.estimate_mode:
                self.filenames.append(f)
            else:
                filename_base = os.path.basename(f)[:-5]
                # Assumes only WSC data has filename of length 14
                # TODO: streamline list names to fit exported names
                if len(filename_base) == 14:
                    filelistname = filename_base[:7].lower()
                else:
                    filelistname = filename_base.lower()
                if filelistname in list_train:
                    self.filenames.append(f)

        if (n > 0):
            self.filenames = self.filenames[:n]
        self.label_name = config.label
        self.label_cond_name = config.label_cond
        self.label_cond_size = config.label_cond_size
        self.n_class = config.n_class
        self.pad_length = config.pad_length
        self.pad_dim = 0
        self.cond_drop = config.cond_drop
        self.cond_drop_p = config.cond_drop_prob
        self.cond_label_scale = {'q_low': 1.0, 'q_high': 1.0, 'age': 88.5, 'bmi': 84.8, 'sex': 1.0}

    def __len__(self):
        """Get length of dataset iterator

        Returns:
            len (size): length of dataset iterator
        """
        return len(self.filenames)

    def load_h5(self, filename):
        """Loads latent space representation of a polysomnography from an h5 file

        Args:
            filename (str): filename of h5 file

        Returns:
            data (numpy.array): polysomnography data of size [n_epochs, n_features]
            label (numpy.array): age
            label_cond (numpy.array): extra conditional labels to input
            attrs (dict): all attributes from file
        """
        with h5py.File(filename, "r") as f:
            # Load features
            data = np.array(f['PSG'])
            # Load attributes
            attrs = {}
            for k, v in f.attrs.items():
                attrs[k] = v
            # Extract labels and conditioning labels
            label = np.concatenate([np.array(attrs[i], ndmin=1) for i in self.label_name]).astype(np.float32)
            label_cond = [np.array(attrs[i], ndmin=1) for i in self.label_cond_name]
            label_cond = np.array(label_cond) if not label_cond else np.concatenate(label_cond).astype(np.float32)

        return data, label, label_cond, attrs

    def drop_cond_label(self, label_cond):
        """Dropout for extra input labels

        Args:
            label_cond (numpy.array): extra conditional labels to input

        Returns:
            label_cond (numpy.array): extra conditional labels to input after dropout
        """
        if self.cond_drop and self.mode == 'train':
            drop_idx = np.random.rand(label_cond.shape[0]) < self.cond_drop_p
            label_cond[drop_idx] = 0.0
            label_cond = label_cond / (1 - self.cond_drop_p)
        return label_cond

    def scale_cond_labels(self, label_cond):
        """Scaling of extra input labels

        Args:
            label_cond (numpy.array): extra conditional labels to input

        Returns:
            label_cond (numpy.array): extra conditional labels to input after scaling
        """
        cond_idx_start = 0
        for idx, i in enumerate(self.label_cond_name):
            cond_idx_end = cond_idx_start + self.label_cond_size[idx]
            label_cond[cond_idx_start:cond_idx_end] /= self.cond_label_scale[i]
            cond_idx_start = cond_idx_end
        return label_cond

    def pad_psg_feat(self, data):
        """This pads the latent space with zeros or removes additional epochs

        Args:
            data (numpy.array): polysomnography data of size [n_epochs, n_features]

        Returns:
            padded_data (numpy.array): padded polysomnography data of size [n_channels, epoch_size]
        """
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
        """Gets a single batch element from the dataset

        Args:
            idx (int): index to read

        Returns:
            out (dict): a dict {'fid': filename, 
                                'position': position of epoch in file,
                                'data': polysomnography data of size [n_epochs, n_features]
                                'label': age,
                                'label_cond': extra conditional labels to input,
                                'all_attrs': all attributes in dict}
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        record = self.filenames[idx]
        file_id = os.path.join(self.filepath, record)
        data, label, label_cond, attrs = self.load_h5(file_id)
        data = self.pad_psg_feat(data)
        label_cond = self.scale_cond_labels(label_cond)
        label_cond = self.drop_cond_label(label_cond)
        out = {'fid': record,
                'data': torch.from_numpy(data.astype(np.float32)),
                'label': torch.from_numpy(label.astype(np.float32)),
                'label_cond': torch.from_numpy(label_cond.astype(np.float32)),
                'all_attrs': attrs}
        return out

if __name__ == '__main__':

    # Set up config
    config = Config()
    config.epoch_size = int(1*128*60)
    config.pre_batch_size = 32
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets
    for val_set in ['train', 'val' ,'test']:
        ds = PSG_pretrain_Dataset(config, val_set, n = -1)
        # ds_f = PSG_feature_Dataset(config, 'train', n = 100)

        # Create dataloader
        dl = DataLoader(ds, shuffle=False, batch_size=32, num_workers=0, pin_memory=True)
        # dl_f = DataLoader(ds_f, shuffle=False, batch_size=64, num_workers=0, pin_memory=False)

        # data bar
        bar_data = tqdm(dl, total=len(dl), desc=f'Loss: {np.inf:.04f}')
        # bar_data_f = tqdm(dl_f, total=len(dl_f), desc=f'Loss: {np.inf:.04f}')

        # Iterate dataset 1
        for i in range(5):
            try:
                time_start = time.time()
                batch = next(iter(bar_data))
                print('')
                print('Batch time: {:.3f}'.format(time.time() - time_start))
                print(batch["fid"])
                print(batch['position'])
            except Exception as e:
                print(batch["fid"])
                print(e)
        print(f'Recording: {batch["fid"]}')
        print(f'Position in recording: {batch["position"]}')
        print(f'Data: {batch["data"].shape}')
        print(f'Labels:{batch["label"].shape}')
        print(f'Conditioning labels:{batch["label_cond"].shape}')
        print(f'All attributes:{batch["all_attrs"]}')

    # Iterate dataset f
    # for i in range(10):
    #     time_start = time.time()
    #     batch = next(iter(bar_data_f))
    #     print('')
    #     print('Batch time: {:.3f}'.format(time.time() - time_start))
    # print(f'Recording: {batch["fid"]}')
    # print(f'Data: {batch["data"].shape}')
    # print(f'Labels:{batch["label"].shape}')
    # print(f'Conditioning labels:{batch["label_cond"].shape}')
