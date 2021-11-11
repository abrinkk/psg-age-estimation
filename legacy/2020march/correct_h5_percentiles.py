
import os
import numpy as np
import h5py

h5_folders = ['H:\\nAge\\test\\',
                'H:\\nAge\\test_F\\',
                'H:\\nAge\\train\\',
                'H:\\nAge\\train_F\\',
                'H:\\nAge\\val\\',
                'H:\\nAge\\val_F\\']


m = len(h5_folders)
for j in range(m):
    input_folder = h5_folders[j]
    filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    n = len(filenames)
    for i in range(n):
        filename = input_folder + filenames[i]
        if 'shhs' in filename:
            with h5py.File(filename, "r+") as f:
                q5 = f.attrs['q_low']
                q95 = f.attrs['q_high']
                f.attrs['q_low'] = q5 / 1000000.0
                f.attrs['q_high'] = q95 / 1000000.0
                if i == 0:
                    print(q5, q95)
                    print(f.attrs['q_low'], f.attrs['q_high'])
