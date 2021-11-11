'''
Inspired by https://github.com/Stanford-STAGES/stroke-deep-learning
'''

import numpy as np
import pyedflib
from fractions import Fraction
from scipy.signal import resample_poly, iirfilter, sosfiltfilt, sosfreqz
from sklearn.preprocessing import StandardScaler
import csv
import matplotlib
import matplotlib.pyplot as plt
import time
import os

# Root path
root_path = 'C:\\Users\\andbri\\Desktop\\SleepSum\\'

# Channel label dictionaries
channel_alias = {'C3': ['C3-A2', 'EEG C3-A2', 'C3-M2', 'C3-avg', 'C3-x','EEG(sec)'],
                 'C4': ['C4-A1', 'EEG C4-A1', 'C4-M1', 'C4-avg', 'C4-x', 'EEG'],
                 'Chin': ['Chin1-Chin2', 'Chin EMG', 'EMG'],
                 'Leg': ['LLEG1-RLEG1','LAT-RAT'],
                 'Arm': ['ARM1-ARM2'],
                 'EOGL': ['EOGL','LEOG-M2','EEG LOC-A2','LEOG-M2','LEOG-x', 'EOG(L)'],
                 'EOGR': ['EOGR', 'EEG ROC-A1','REOG-M1','REOGM1','ROCM1','ROC-M1','ROCA1','ROCA2','REOGx','REOG-x', 'EOG(R)'],
                 'ECG': ['ECG','EKG1-EKG2', 'ECG ECGL-ECGR','EKG1AVG','EKG1EKG2','LLEG1EKG2','EKG'],
                 'Airflow': ['Airflow','AIRFLOW','OralTherm','NasaslTherm','Nasal/OralTherm','Resp FLOW','FLOW', 'NEW AIR', 'AIRFLOW'],
                 'NasalP': ['NasalPres','NasalP','NASAL PRES','Cannula Flow'],
                 'Abd': ['ABD', 'ABDO EFFORT', 'Abd','Abdomen','Abdominal','Resp ABD', 'ABDO RES'],
                 'Chest': ['CHEST', 'Chest', 'Resp CHEST', 'THOR EFFORT','Thoracic', 'THOR RES'],
                 'OSat': ['OSAT','SpO2','SaO2'],
                 'Mic': ['Snore','SNORE', 'SOUND']}

unref_channel_alias = {'C3': ['C3'], 
                       'C4': ['C4'], 
                       'Chin': ['L Chin', 'LChin', 'Chin2', 'CHIN2','EMG2'], 
                       'Leg': ['LLEG', 'Leg L', 'L Leg','LAT2','EMG LAT1-LAT2','LLEG1-LLEG2'],
                       'Arm': ['ARM L','ArmL','EMG ArmL'],
                       'EOGL': ['LOC'],
                       'EOGR': ['ROC'],
                       'ECG': ['ECGL', 'ECG L','ECG2'],
                       'Airflow': [],
                       'NasalP': [],
                       'Abd': [],
                       'Chest': [],
                       'OSat': [],
                       'Mic': []}

ref_channel_alias = {'C3': ['A2', 'M2'], 
                     'C4': ['A1', 'M1'], 
                     'Chin': ['R Chin', 'RChin', 'Chin1', 'CHIN1','EMG1'], 
                     'Leg': ['RLEG','Leg R','RAT2','R Leg','RAT1','RAT2','EMG RAT1-RAT2','RLEG1-RLEG2'],
                     'Arm': ['ARM R','ArmR','EMG ArmR'],
                     'EOGL': ['A2','M2'],
                     'EOGR': ['A1','M1'],
                     'ECG': ['ECGR','ECG1','ECG R'],
                     'Airflow': [],
                     'NasalP': [],
                     'Abd': [],
                     'Chest': [],
                     'OSat': [],
                     'Mic': []}

# Channel pre-processing options
des_fs = {'C3': 128, 'C4': 128, 'Chin': 128, 'Leg': 128, 'Arm': 128, 'EOGL': 128, 'EOGR': 128, 'ECG': 128, 'Airflow': 128, 'NasalP': 128, 'Abd': 128, 'Chest': 128, 'OSat': 128, 'Mic': 128}
hp_fs = {'C3': 0.5, 'C4': 0.5, 'Chin': 0.5, 'Leg': 0.5, 'Arm': 0.5, 'EOGL': 0.5, 'EOGR': 0.5, 'ECG': 0.5, 'Airflow': 0.5, 'NasalP': 0.5, 'Abd': 0.5, 'Chest': 0.5, 'OSat': 0.5, 'Mic': 0.5}

# Channels to correct units
channel_correct_units = ['C3','C4','Chin','Leg','Arm','EOGL','EOGR','ECG']

# Example data
edf_path = 'train\\A0054_7.EDF'
#channels = ['C3','Chin','Leg','EOGL','EOGR','ECG']

def psg_highpass(cutoff, fs, order=5, plot_opt = 0):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = iirfilter(order, normal_cutoff, rp = 1, rs = 40 , btype='highpass', analog=False, output = 'sos', ftype = 'ellip')
    if plot_opt == 1:
        w, h = sosfreqz(sos)
        plt.semilogx(w*fs/(2*np.pi), 20 * np.log10(abs(h)))
        plt.title('Filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(cutoff, color='green') # cutoff frequency
        plt.show()
    return sos

def psg_highpass_filter(data, cutoff, fs, order=5, plot_opt = 0):
    sos = psg_highpass(cutoff, fs, order=order, plot_opt=plot_opt)
    y = sosfiltfilt(sos, data)
    return y

def rescale(x, mode):
    if mode == 'hard':
        x_s = 2*(x - min(x))/(max(x) - min(x)) - 1
    elif mode == 'soft':
        q5 = np.percentile(x, 5)
        q95 = np.percentile(x, 95)
        x_s = 2*(x - q5)/(q95 - q5) - 1
        return x_s, q5, q95
    elif mode == 'standardize':
        scaler = StandardScaler()
        x_s = scaler.fit_transform(x)
    else:
        x_s = x
    return x_s

def get_label_names(labels, cohort):
    if cohort == 'cfs':
        labels = ['SEX' if x == 'sex' else x for x in labels]
        labels = ['ESSSCOR' if x == 'ess' else x for x in labels]
    elif cohort == 'shhs-v1':
        labels = ['age_s1' if x == 'age' else x for x in labels]
        labels = ['bmi_s1' if x == 'bmi' else x for x in labels]
        labels = ['gender' if x == 'sex' else x for x in labels]
        labels = ['ESS_s1' if x == 'ess' else x for x in labels]
    return labels

def load_psg_labels(filename, label, cohort):
    label = get_label_names(label, cohort)
    if cohort == 'cfs':
        file_id = filename[-10:-4]
        with open('G:\\cfs\\datasets\\cfs-visit5-dataset-0.4.0.csv','r') as readFile:
            reader = csv.reader(readFile)
            csv_data = list(reader)
            header = csv_data[0]
            id_list = [item[0].lower() for item in csv_data]
            index_var = [header.index(x) for x in label]
            index_id = id_list.index(file_id)
            lab_val = [csv_data[index_id][x] for x in index_var]
            return lab_val
    elif cohort == 'shhs-v1':
        file_id = filename[-10:-4]
        with open('H:\\shhs\\datasets\\shhs1-dataset-0.15.0.csv','r') as readFile:
            reader = csv.reader(readFile)
            csv_data = list(reader)
            header = csv_data[0]
            id_list = [item[0].lower() for item in csv_data]
            index_var = [header.index(x) for x in label]
            index_id = id_list.index(file_id)
            lab_val = [csv_data[index_id][x] for x in index_var]
            # Modify values
            lab_val = [2 - float(lab_val[i]) if label[i] == 'gender' else lab_val[i] for i in range(len(label))]
            return lab_val
    elif cohort == 'mros-v1':
        file_id = filename[-10:-4]
        with open('G:\\mros\\datasets\\mros-visit1-dataset-0.3.0.csv','r') as readFile:
            reader = csv.reader(readFile)
            csv_data = list(reader)
            header = csv_data[0]
            id_list = [item[0].lower() for item in csv_data]
            index_var = [header.index(x) for x in label]
            index_id = id_list.index(file_id)
            lab_val = [csv_data[index_id][x] for x in index_var]
            return lab_val
    elif cohort == 'kassel':
        parts = filename.split('\\')
        event_path = parts[0] + '\\' + parts[1] + '\\' + parts[2] + '\\' + parts[3] + '\\' + parts[4] + '_events\\' + parts[5][:-4] + '.txt'
        POS = []
        DUR = []
        TYP = []
        with open(event_path,'r') as readFile:
            for line in readFile:
                event_info = line.split(',')
                TYPe = event_info[2][1:-2]
                if TYPe in label:
                    TYP.append(TYPe)
                    POS.append(event_info[0])
                    DUR.append(event_info[1])
        return (POS,DUR,TYP)
    else:
        return -1
    
    return

def load_psg_lights(filename, cohort):
    if cohort == 'cfs':
        return -1, -1
    elif cohort == 'mros-v1':
        label = ['poststtp', 'postendp', 'postlotp']
        lab_val = load_psg_labels(filename, label, cohort)
        t_start = time.mktime(time.strptime('1 Jan 00 ' + lab_val[0],'%d %b %y %H:%M:%S'))
        t_lights_off = time.mktime(time.strptime('1 Jan 00 ' + lab_val[2],'%d %b %y %H:%M:%S'))
        t_lights_on = time.mktime(time.strptime('1 Jan 00 ' + lab_val[1],'%d %b %y %H:%M:%S'))
        lights_off = t_lights_off - t_start
        lights_on = t_lights_on - t_start
        # Correct for change in date
        if lights_off < 0:
            lights_off += 60*60*24.0
        if lights_on < 0:
            lights_on += 60*60*24.0
        
        return lights_off, lights_on
    elif cohort == 'kassel':
        label = ['Lights_Off','Lights_On']
        lab_val = load_psg_labels(filename, label, cohort)
        idx_off = [i for i, x in enumerate(lab_val[2]) if label[0] in x]
        idx_on = [i for i, x in enumerate(lab_val[2]) if label[1] in x]
        return lab_val[0][idx_off[0]], lab_val[0][idx_on[0]] 
    else:
        return -1

def correct_units(unit_string):
    if unit_string == 'uV':
        g = 0.001
    elif unit_string == 'V':
        g = 1.0
    else:
        g = 1.0
    return g

def load_edf_file(filename, channels):
    # Load EDF file
    f = pyedflib.EdfReader(filename)
    # Channel labels
    channel_labels = f.getSignalLabels()
    # Sampling frequencies
    fss = f.getSampleFrequencies()
    # Pre-allocation of x
    x = []
    # Pre-allocation of data quantiles
    q_low, q_high = [], []
    # Extract channels
    for channel in channels:
        # Is channel referenced?
        if any([x in channel_alias[channel] for x in channel_labels]):
            channel_idx = channel_labels.index(next(filter(lambda i: i in channel_alias[channel],channel_labels)))
            # Gain factor
            if channel in channel_correct_units:
                g = correct_units(f.getPhysicalDimension(channel_idx))
            else:
                g = 1
            # Read signal
            sig = g*f.readSignal(channel_idx)
            
        # Else: reference channels
        elif any([x in unref_channel_alias[channel] for x in channel_labels]):
            channel_idx = channel_labels.index(next(filter(lambda i: i in unref_channel_alias[channel],channel_labels)))
            ref_idx = channel_labels.index(next(filter(lambda i: i in ref_channel_alias[channel],channel_labels)))
            # Gain factor
            if channel in channel_correct_units:
                g = correct_units(f.getPhysicalDimension(channel_idx))
                g_ref = correct_units(f.getPhysicalDimension(ref_idx))
            else:
                g = 1
                g_ref = 1
            # Assuming fs for signal and reference is identical
            sig = g*f.readSignal(channel_idx) - g_ref*f.readSignal(ref_idx)
            
        # Else empty
        else:
            sig = []
        
        # If not empty
        if len(sig) != 0:
            # Resampling
            fs = fss[channel_idx]
            if fs != des_fs[channel]:
                resample_frac = Fraction(des_fs[channel]/fs).limit_denominator(100)
                sig = resample_poly(sig, resample_frac.numerator, resample_frac.denominator)
            
            # Filter signals
            if hp_fs[channel] != 0:
                sig_filtered = psg_highpass_filter(sig, hp_fs[channel], des_fs[channel], order = 16)
            else:
                sig_filtered = sig
            # Scale signal
            sig_scaled, q5, q95 = rescale(sig_filtered, 'soft')
        else:
            sig_scaled, q5, q95 = sig, 0, 0

        
        x.append(sig_scaled)
        if channel in channel_correct_units:
            q_low.append(q5)
            q_high.append(q95)
    
    # Replace empty with zeros
    N = max([len(s) for s in x])
    for i, channel in enumerate(channels):
        if len(x[i]) == 0:
            x[i] = np.zeros(N)
        elif len(x[i]) != N:
            x[i] = np.append(x[i], np.zeros(N - len(x[i])))
    
    data = {'x': x, 'fs': [des_fs[x] for x in channels], 'channels': channels, 'q_low': q_low, 'q_high': q_high}
    f._close()
    return data

def plot_edf_data(data, channels, save_fig = False):
    x = data['x']
    fs = data['fs']
    n = len(x)
    fig, axs = plt.subplots(n, 1, figsize = (10,10))
    for i in range(n):
        N = len(x[i])
        t = np.arange(0,N,1) /fs[i]
        if channels[i] in ['C3','C4','EOGL','EOGR']:
            ylimits = [-3, 0, 3]
        elif channels[i] == 'ECG':
            ylimits = [-3, 0, 10]
        else:
            ylimits = [-10, 0, 10]
        axs[i].plot(t, x[i], linewidth = 0.25)
        axs[i].set_xlim(0, (N-1)/fs[i])
        axs[i].set_ylim(ylimits[0],ylimits[-1])
        axs[i].set_yticks(ylimits)
        axs[i].set_yticklabels(['']*3)
        axs[i].set_xticks([0, 60, 120, 180, 240, 300])
        if i == n-1:
            axs[i].set_xlabel('Time [s]')
        else:
            axs[i].set_xticklabels([])
        axs[i].set_ylabel(channels[i], rotation = 'horizontal', ha='right')
        axs[i].grid(True)

    fig.tight_layout()
    plt.show()
    if save_fig:
        fig.savefig("5min_signals.pdf", bbox_inches='tight', dpi=300)

    return