import numpy as np
import pyedflib
import matplotlib
import matplotlib.pyplot as plt
import time
import os

edf_file_paths = ['H:\\shhs\\polysomnography\\edfs\\shhs1\\',
                  'G:\\mros\\polysomnography\\edfs\\visit1\\',
                  'G:\\wsc\\polysomnography\\edfs\\',
                  'G:\\wsc2\\polysomnography\\edfs\\',
                  'G:\\cfs\\polysomnography\\edfs\\',
                  'H:\\data\\kassel_continuous\Baseline\\DeNoPa\\',
                  'H:\\data\\kassel_continuous\Baseline\\DKS\\',
                  'H:\\data\\kassel_continuous\FollowUp1\\DeNoPa\\',
                  'H:\\data\\kassel_continuous\FollowUp1\\DKS\\']

all_channels = []
all_channels_fs = {}
edfs_per_path = 50

m = len(edf_file_paths)
for j in range(m):
    input_folder = edf_file_paths[j]
    filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    n = len(filenames)
    for i in range(min(edfs_per_path,n)):
        filename = input_folder + filenames[i]
        f = pyedflib.EdfReader(filename)
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
            else:
                all_channels_fs[channel_labels[k]].append(np.float32(fss[k]))
                
        f._close()

c = len(all_channels)
for k in range(c):
    all_channels_fs[all_channels[k]] = [min(all_channels_fs[all_channels[k]]), max(all_channels_fs[all_channels[k]]), np.mean(all_channels_fs[all_channels[k]] )]
print(all_channels_fs)
#dict(sorted(all_channels_fs.items(), key=lambda x: x[0]))