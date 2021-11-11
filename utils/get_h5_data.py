import time

import h5py
import numpy as np


def get_h5_interp(filename):
    time.sleep(0.05)
    with h5py.File(filename, 'r') as h5:
        interpretation = h5['Interpretation'][:]
        delta = h5['Delta'][:]

    return interpretation, delta

def get_h5_am(filename):
    time.sleep(0.05)
    with h5py.File(filename, 'r') as h5:
        am_data = h5['am_data'][:]

    return am_data

def get_h5_ssc(filename):
    time.sleep(0.05)
    with h5py.File(filename, 'r') as h5:
        ssc = h5['SSC'][:]
    return ssc

def get_h5_size(filename):
    with h5py.File(filename, 'r') as h5:
        data_size = h5['PSG'].shape[1]
        attrs = {}
        for k, v in h5.attrs.items():
            attrs[k] = v.astype(np.float32)
    return data_size, attrs

def get_h5_data(filename):
    time.sleep(0.05)
    with h5py.File(filename, 'r') as h5:
        data = h5['PSG'][:]
        attrs = {}
        for k, v in h5.attrs.items():
            attrs[k] = v.astype(np.float32)

    return data, attrs

def get_chunk_h5_data(filename, pos):
    time.sleep(0.05)
    with h5py.File(filename, 'r') as h5:
        data = h5['PSG'][:, pos[0]:pos[1]]
        attrs = {}
        for k, v in h5.attrs.items():
            attrs[k] = v.astype(np.float32)

    return data, attrs

