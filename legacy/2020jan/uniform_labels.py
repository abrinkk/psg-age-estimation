import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py
import pandas as pd

dataset_file_paths = ['G:\\cfs\\datasets\\cfs-visit5-dataset-0.4.0.csv']
dataset_label_keys = ['age']
dataset_id_keys    = ['nsrrid']
dataset_n_samples  = [511]

training_IDs = []

for i in range(len(dataset_file_paths)):
    # Read files
    file = dataset_file_paths[i]
    l_key = dataset_label_keys[i]
    id_key = dataset_id_keys[i]
    n_samples = dataset_n_samples[i]
    df = pd.read_csv(file)
    ids = np.array(df[id_key])
    age = np.array(df[l_key])
    # Allocate list for storing ids
    dataset_IDs = []
    dataset_lab = []
    # Compute sample probability to get uniform distribution
    hist_age = np.histogram(age, np.arange(100))
    p_age = np.ones(len(age))
    p_age = p_age / hist_age[0][age.astype(int)]
    p_age = p_age / sum(hist_age[0] != 0)
    # Sample data with uniform label distribution
    sample_ids = np.random.choice(ids, n_samples, replace = False, p = p_age)
    training_IDs.append(sample_ids)

