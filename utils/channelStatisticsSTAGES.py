import numpy as np
import pyedflib
import matplotlib
import matplotlib.pyplot as plt
import time
import os
from zipfile import ZipFile

# edf_file_paths = ['H:\\STAGES\\polysomnograms\\']
# tmp_folder = 'H:\\nAge\\tmp\\'
edf_file_paths = ['/oak/stanford/groups/mignot/psg/STAGES/deid/']
tmp_folder = '/scratch/users/abk26/nAge/tmp/'

all_channels = []
all_channels_fs = {}
all_channels_count = {}
edfs_per_path = 3000

skip_file = False

m = len(edf_file_paths)
for j in range(m):
    input_folder = edf_file_paths[j]
    filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    n = len(filenames)
    for i in range(min(edfs_per_path,n)):
        filename = input_folder + filenames[i]
        if filename.endswith(".zip"):
            with ZipFile(filename, 'r') as zip_f:
                for f in zip_f.namelist():
                    if f.endswith(".edf"):
                        edf_file = zip_f.extract(f, tmp_folder)
                        try:
                            f = pyedflib.EdfReader(edf_file)
                            skip_file = False
                        except:
                            skip_file = True
        else:
            try:
                f = pyedflib.EdfReader(filename)
                skip_file = False
            except:
                skip_file = True

        # Skip file=
        if skip_file:
            continue

        # Channel labels
        channel_labels = f.getSignalLabels()
        # Sampling frequencies
        fss = f.getSampleFrequencies()
        # Get all channel names
        new_idx = [i for i, x in enumerate(channel_labels) if x not in all_channels]
        all_channels = all_channels + [channel_labels[x] for x in new_idx]
        c = len(channel_labels)
        for k in range(c):
            if k in new_idx:
                all_channels_fs[channel_labels[k]] = [np.float32(fss[k])]
                all_channels_count[channel_labels[k]] = 1
            else:
                all_channels_fs[channel_labels[k]].append(np.float32(fss[k]))
                all_channels_count[channel_labels[k]] += 1
                
        f._close()
        if filename.endswith(".zip"):
            os.remove(edf_file)

c = len(all_channels)
for k in range(c):
    all_channels_fs[all_channels[k]] = [min(all_channels_fs[all_channels[k]]), max(all_channels_fs[all_channels[k]]), np.mean(all_channels_fs[all_channels[k]] )]
print(all_channels_fs, all_channels_count)
#dict(sorted(all_channels_fs.items(), key=lambda x: x[0]))