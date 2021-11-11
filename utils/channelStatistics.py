import numpy as np
import pyedflib
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import glob

#edf_file_paths = ['G:\\stanford_irbd\\edf_control\\',
#                  'G:\\stanford_irbd\\edf_irbd\\']
edf_file_paths = ['G:\\GlostrupRBD\\']
edf_file_paths_sub = [True]
#edf_file_paths_sub = [False, False]

#edf_file_paths = ['H:\\homepap\\polysomnography\\edfs\\lab\\full\\']
#edf_file_paths = ['/oak/stanford/groups/mignot/psg/SSC/APOE_deid/']

all_channels = []
all_channels_fs = {}
all_channels_count = {}
edfs_per_path = 500

m = len(edf_file_paths)
for j in range(m):
    input_folder = edf_file_paths[j]
    if edf_file_paths_sub[j]:
        pattern = os.path.join(input_folder, '**', '*.edf')
        filenames = glob.glob(pattern, recursive=True)
    else:
        filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(".edf")]
    n = len(filenames)
    for i in range(min(edfs_per_path,n)):
        if edf_file_paths_sub[j]:
            filename = filenames[i]
        else:
            filename = input_folder + filenames[i]
        try:
            f = pyedflib.EdfReader(filename)
        except Exception as e:
            print(e)
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

c = len(all_channels)
for k in range(c):
    all_channels_fs[all_channels[k]] = [min(all_channels_fs[all_channels[k]]), max(all_channels_fs[all_channels[k]]), np.mean(all_channels_fs[all_channels[k]] )]
print(all_channels_fs, all_channels_count)
#dict(sorted(all_channels_fs.items(), key=lambda x: x[0]))