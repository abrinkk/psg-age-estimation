import numpy as np
import mne
import matplotlib
import matplotlib.pyplot as plt
import time
import os

#edf_file_paths = ['G:\\stanford_irbd\\edf_control\\',
#                  'G:\\stanford_irbd\\edf_irbd\\']
#edf_file_paths = ['H:\\SSC\\polysomnography\\edfs\\',
#                   'H:\\STAGES\\polysomnograms\\STNF\\',
#                    'G:\\wsc\\polysomnography\\edfs\\']
edf_file_paths = ['/oak/stanford/groups/mignot/psg/SSC/APOE_deid/',
                    '/oak/stanford/groups/mignot/psg/STAGES/deid/edf_sym_linked/',
                    '/oak/stanford/groups/mignot/psg/WSC_EDF/']

all_channels = []
all_channels_fs = {}
all_channels_count = {}
edfs_per_path = 200

m = len(edf_file_paths)
for j in range(m):
    input_folder = edf_file_paths[j]
    filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(".edf")]
    n = len(filenames)
    for i in range(min(edfs_per_path,n)):
        filename = input_folder + filenames[i]
        f = mne.io.read_raw_edf(filename, preload=False, verbose=False, stim_channel=None)
        # Channel labels
        channel_labels = f.ch_names
        # Sampling frequencies
        fss = f.info['sfreq']
        # Get all channel names
        new_idx = [i for i, x in enumerate(channel_labels) if x not in all_channels]
        all_channels = all_channels + [channel_labels[x] for x in new_idx]
        c = len(channel_labels)
        for k in range(c):
            if k in new_idx:
                all_channels_fs[channel_labels[k]] = [np.float32(fss)]
                all_channels_count[channel_labels[k]] = 1
            else:
                all_channels_fs[channel_labels[k]].append(np.float32(fss))
                all_channels_count[channel_labels[k]] += 1
                

c = len(all_channels)
for k in range(c):
    all_channels_fs[all_channels[k]] = [min(all_channels_fs[all_channels[k]]), max(all_channels_fs[all_channels[k]]), np.mean(all_channels_fs[all_channels[k]] )]
print(all_channels_fs, all_channels_count)
#dict(sorted(all_channels_fs.items(), key=lambda x: x[0]))