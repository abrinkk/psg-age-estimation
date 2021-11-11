'''
Inspired by https://github.com/Stanford-STAGES/stroke-deep-learning
'''

import numpy as np
from numpy.lib.function_base import _cov_dispatcher
import pyedflib
from fractions import Fraction
from scipy.signal import resample_poly, iirfilter, sosfiltfilt, sosfreqz
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import xmltodict
import copy

# Channel label dictionaries
channel_alias = {'C3': ['C3-A2', 'EEG C3-A2','EEG C3-A22', 'C3-M2', 'C3-avg', 'C3-x','EEG(sec)','EEG 2','EEG2','CENTRAL','C3M2','C3:M2','c3-a2'],
                 'C4': ['C4-A1', 'EEG C4-A1','EEG C4-A12', 'C4-M1', 'C4-avg', 'C4-x', 'EEG','CENTRAL','C4M1','C4:M1','c4-a1'],
                 'Chin': ['Chin1-Chin2', 'Chin EMG', 'EMG', 'CHIN EMG', 'Chin', 'CHIN','EMG Chin', 'EMG Aux12','EMG CHIN','chin','Chin-Gnd'],
                 'Leg': ['LLEG1-RLEG1','LAT-RAT','Leg 12', 'L/RAT','RAT-L'],
                 'Arm': ['ARM1-ARM2'],
                 'EOGL': ['EOGL','LEOG-M2','EEG LOC-A2','LEOG-M2','LOCM2','LOC-M2','LOCA2','LOC-A2','LOC/A2','LEOGx','LEOG-x', 'EOG(L)', 'E1M2','EOG LOC-A2','EOG LOC-A22','E1:M2','EOG1','EOGV','EOGV-A2','eogl-a2'],
                 'EOGR': ['EOGR', 'EEG ROC-A1','REOG-M1','REOGM1','ROCM1','ROC-M1','ROCA1','ROC-A1','ROC/A1','ROCA2','REOGx','REOG-x', 'EOG(R)', 'E2M1','E2M2','EOG ROC-A1','EOG ROC-A2','EOG ROC-A22','E2:M1','EOG2','EOGH','EOGH-A1','eogr-a1'],
                 'ECG': ['ECG','EKG1-EKG2', 'ECG ECGL-ECGR','EKG1AVG','EKG1EKG2','LLEG1EKG2','EKG', 'ECG 2', 'ECG2', 'ECG I2', 'ECG II', 'ECG IIHF','ECG EKG','ecg'],
                 'Airflow': ['Airflow','AIRFLOW','OralTherm','Oral Thermistor','NasaslTherm','NasalTherm','Nasal Therm','Nasal/OralTherm','Resp FLOW','FLOW','Flow','FLOW2','AIR-flow','Flow Patient','Flow Patient2','Flow Patient3','Flow Aux4', 'NEW AIR', 'NEWAIR', 'New Air', 'new A/F', 'AIRFLOW', 'NasalOr','NasOr', 'NasOr2','Air Nasal','nasal','Nasal-Gnd'],
                 'NasalP': ['NasalPres', 'Nasal Pressure', 'Nasal','NasalP','NASAL PRES','NASAL PRESSURE','Cannula Flow', 'CannulaFlow', 'Canulla','Cannula','Press Patient','Pressure','CannualFlow','Cannulaflow'],
                 'Abd': ['ABD', 'ABDO EFFORT', 'Abd', 'Abdomen','Abdominal','Resp ABD', 'ABDO RES', 'ABDM', 'ABDOMEN','Effort ABD', 'AbdDC','Resp Abdomen','abdomen','Abdomen-Gnd'],
                 'Chest': ['CHEST', 'Chest', 'Resp CHEST', 'THOR EFFORT','Thoracic','THOR', 'THOR RES', 'RIB CAGE','Effort THO','Chest1','ChestDC','Resp Thorax','thorax','Thorax-Gnd'],
                 'OSat': ['OSAT','SpO2', 'SPO2', 'SaO2','Sa02','SAO2','SA02','SaO2 SpO2','sao2','SAO2-Gnd'],
                 'Mic': ['Snore','Snore2','SNORE','SNOR', 'SOUND']}

unref_channel_alias = {'C3': ['C3','EEG C3-Ref','C3-Ref'], 
                       'C4': ['C4','EEG C4-Ref','C4-Ref'], 
                       'Chin': ['L Chin', 'LChin','LCHIN', 'Lchin', 'EMG/L', 'Chin2', 'CHIN2','EMG2','Chin2 EMG','EMG #2','EMG Chin2', 'EMG Aux2'], 
                       'Leg': ['LLEG', 'Leg L', 'L Leg','Leg/L' , 'LAT','LAT2','EMG LAT1-LAT2','LLEG1-LLEG2','LLeg1-LLeg2','LAT1-LAT2','L-LEG1','L-LEG2','LegsL-Leg1','L-Legs','Leg 1','Lleg','Lleg1','Lleg2','LLeg1','LLeg2','LLEG1','LLEG2','L-Leg2','LLeg3','LLeg4','LA1-LA2','PLMl','PLMl.','Leg Ltibial','TIBV','EMG TIBV','tibl','TIBV-Gnd'],
                       'Arm': ['ARM L','ArmL','EMG ArmL', 'Arms-L'],
                       'EOGL': ['LOC', 'E1', 'E-1', 'E1 (LEOG)', 'EOG1','L-EOG','EEG EOGV-Ref','EOGV-Ref'],
                       'EOGR': ['ROC', 'E2', 'E-2', 'E2 (REOG)', 'EOG2','R-EOG','EEG EOGH-Ref','EOGH-Ref'],
                       'ECG': ['ECGL', 'ECG L','ECG2', 'EKG #2', 'EKG2'],
                       'Airflow': [],
                       'NasalP': [],
                       'Abd': [],
                       'Chest': [],
                       'OSat': [],
                       'Mic': []}

ref_channel_alias = {'C3': ['A2', 'M2', 'EEG A2-Ref','A2-Ref'], 
                     'C4': ['A1', 'M1', 'EEG A1-Ref','A1-Ref'], 
                     'Chin': ['R Chin', 'RChin','RCHIN','Rchin', 'Chin1', 'EMG/R', 'CHIN1','EMG1','EMG #1', 'EMG Aux1'], 
                     'Leg': ['RLEG','Leg R','RAT2','R Leg','RLeg', 'Leg/R', 'RAT','RAT1','RAT2','EMG RAT1-RAT2','RAT1-RAT2','RLEG1-RLEG2','RLeg1-RLeg2','R-Legs','R-LEG 1','Rleg','Rleg1','Rleg2','RLeg1','RLeg2','RLEG1','RLEG2','R-LEG 2','R-Leg1','R-Leg2','Leg 2','RLeg5','RLeg6','PLMr','PLMr.','Leg Rtibial','TIBH','EMG TIBH','tibh','TIBH-Gnd'],
                     'Arm': ['ARM R','ArmR','EMG ArmR', 'Arms-R'],
                     'EOGL': ['A2','M2'],
                     'EOGR': ['A1','M1'],
                     'ECG': ['ECGR','ECG1','ECG R', 'EKG #1','EKG1', 'ECG1'],
                     'Airflow': [],
                     'NasalP': [],
                     'Abd': [],
                     'Chest': [],
                     'OSat': [],
                     'Mic': []}

# Channel pre-processing options
des_fs = {'C3': 128, 'C4': 128, 'Chin': 128, 'Leg': 128, 'Arm': 128, 'EOGL': 128, 'EOGR': 128, 'ECG': 128, 'Airflow': 128, 'NasalP': 128, 'Abd': 128, 'Chest': 128, 'OSat': 128, 'Mic': 128}
hp_fs = {'C3': [0.3, 45.0], 'C4': [0.3, 45.0], 'Chin': 10.0, 'Leg': 10.0, 'Arm': 10.0, 'EOGL': [0.3, 45.0], 'EOGR': [0.3, 45.0], 'ECG': 0.3, 'Airflow': [0.1, 15.0], 'NasalP': 0.1, 'Abd': [0.1, 15.0], 'Chest': [0.1, 15.0], 'OSat': 0, 'Mic': 10.0}

# Channels to correct units
channel_correct_units = ['C3','C4','Chin','Leg','Arm','EOGL','EOGR','ECG']

def psg_highpass(cutoff, fs, order=5, plot_opt = 0):
    """Construct a high pass or bandpass filter

    Args:
        cutoff (int or list[int]): Cut-off freuqncies [Hz]
        fs (int): signal sampling frequency
        order (int, optional): Filter order. Defaults to 5.
        plot_opt (int, optional): If set to 1 it plots filter characteristics. Defaults to 0.

    Returns:
        sos (iirfilter): system order specifications for filter
    """
    nyq = 0.5 * fs
    if isinstance(cutoff, list):
        normal_cutoff = [x / nyq for x in cutoff]
        btype = 'bandpass'
    else:
        normal_cutoff = cutoff / nyq
        btype = 'highpass'

    sos = iirfilter(order, normal_cutoff, rp = 1, rs = 40, btype=btype, analog=False, output='sos', ftype='ellip')
    if plot_opt == 1:
        w, h = sosfreqz(sos, 2**12)
        plt.plot(w*fs/(2*np.pi), 20 * np.log10(abs(h)))
        plt.title('Filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        if btype == 'highpass':
            plt.xlim(0, cutoff*2)
            plt.axvline(cutoff, color='green') # cutoff frequency
        else:
            plt.xlim(0, cutoff[-1] + (cutoff[-1] - cutoff[0])*0.5)
            plt.axvline(cutoff[0], color='green') # cutoff frequency
            plt.axvline(cutoff[-1], color='green') # cutoff frequency
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.show()
    return sos

def psg_highpass_filter(data, cutoff, fs, order=5, plot_opt = 0):
    """Constructs and applies a high-pass or bandpass filter

    Args:
        data (numpy.array): polysomnography data
        cutoff (int or list[int]): Cut-off freuqncies [Hz]
        fs (int): signal sampling frequency
        order (int, optional): Filter order. Defaults to 5.
        plot_opt (int, optional): If set to 1 it plots filter characteristics. Defaults to 0.

    Returns:
        y (numpy.array): filtered polysomnography data
    """
    sos = psg_highpass(cutoff, fs, order=order, plot_opt=plot_opt)
    y = sosfiltfilt(sos, data)
    return y

def psg_resample(sig, channel, fs, method):
    """Resamples a signal

    Args:
        sig (np.array): Input signal to be resampled
        channel (str): channel type
        fs (int): signal sampling frequency
        method (str): Resampling method. Choose one of ['poly', 'linear'] for polyphase iir filter or linear interpolation

    Returns:
        sig (np.array): Input signal after resampling
    """
    if method == 'poly':
        resample_frac = Fraction(des_fs[channel]/fs).limit_denominator(100)
        sig = resample_poly(sig, resample_frac.numerator, resample_frac.denominator)
    elif method == 'linear':
        t = np.arange(0, len(sig)*(1/fs), 1/fs)
        resample_f = interpolate.interp1d(t, sig, bounds_error=False, fill_value='extrapolate')
        t_new = np.arange(0, len(sig)*(1/fs), 1/des_fs[channel])
        sig = resample_f(t_new)
    return sig

def rescale(x, mode):
    """Rescale function to normalize data

    Args:
        x (numpy.array): input data to be resclaed
        mode (str): mode of rescaling to choose ['hard', 'soft', 'standardize', 'osat']

    Returns:
        x_s (numpy.array): rescaled input data
        q5 (float): 5th data percentile of 'standardize' mode is used
        q95 (float): 95th data percentile of 'standardize' mode is used
    """
    eps = 1e-10
    if mode == 'hard':
        x_s = 2*(x - min(x))/(max(x) - min(x) + eps) - 1
    elif mode == 'soft':
        q5 = np.percentile(x, 5)
        q95 = np.percentile(x, 95)
        x_s = 2*(x - q5)/(q95 - q5 + eps) - 1
        return x_s, q5, q95
    elif mode == 'standardize':
        scaler = StandardScaler()
        x_s = scaler.fit_transform(x)
    elif mode == 'osat':
        if max(x) > 1.0:
            x = np.array(x / 100.0)
        x[x < 0.6] = 0.6
        x_s = 2*(x - 0.6)/(1.0 - 0.6) - 1
    else:
        x_s = x
    return x_s, 0, 0

def get_label_names(labels, cohort):
    """Get label names used i various dataset files from cohorts

    Args:
        labels (list[str]): Input labels to translate
        cohort (str): Cohort name

    Returns:
        labels (list[str]): Translated labels to read
    """
    if cohort == 'cfs':
        labels = ['SEX' if x == 'sex' else x for x in labels]
        labels = ['ESSSCOR' if x == 'ess' else x for x in labels]
    elif cohort == 'mros-v1':
        labels = ['vsage1' if x == 'age' else x for x in labels]
        labels = ['gender' if x == 'sex' else x for x in labels]
        labels = ['hwbmi' if x == 'bmi' else x for x in labels]
        labels = ['epepwort' if x == 'ess' else x for x in labels]
    elif cohort == 'shhs-v1':
        labels = ['age_s1' if x == 'age' else x for x in labels]
        labels = ['bmi_s1' if x == 'bmi' else x for x in labels]
        labels = ['gender' if x == 'sex' else x for x in labels]
        labels = ['ESS_s1' if x == 'ess' else x for x in labels]
    elif cohort == 'wsc':
        labels = ['SEX' if x == 'sex' else x for x in labels]
        labels = ['doze_sc' if x == 'ess' else x for x in labels]
    elif cohort == 'ssc':
        labels = ['age_float' if x == 'age' else x for x in labels]
        labels = ['gender' if x == 'sex' else x for x in labels]
    elif cohort == 'sof':
        labels = ['V8AGE' if x == 'age' else x for x in labels]
        labels = ['V8BMI' if x == 'bmi' else x for x in labels]
        labels = ['gender' if x == 'sex' else x for x in labels]
    elif cohort == 'hpap':
        labels = ['gender' if x == 'sex' else x for x in labels]
    return labels

def load_psg_labels(filename, label, cohort, label_path):
    """Loads specified labels from file for a given cohort

    Args:
        filename (str): filename for polysomnography file
        label (list[str]): labels to read
        cohort (str): Cohort name
        label_path (str): path to dataset file

    Returns:
        lab_val (list[float]): Value of labels
    """
    label = get_label_names(label, cohort)
    if cohort == 'cfs':
        file_id = filename[-10:-4]
        with open(label_path, 'r') as readFile:
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
        with open(label_path, 'r') as readFile:
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
    elif cohort == 'wsc':
        file_id = filename[-18:-11]
        df = pd.read_excel(label_path, sheet_name='data_all')
        id_list = df.SUBJ_ID + '_' + df.VISIT_NUMBER.astype(str)
        # if not in database return empty
        if sum(id_list == file_id) != 1:
            return ['' for x in label]
        lab_val = [df.loc[id_list == file_id][x].values[0] for x in label]
        # Modify values
        for i in range(len(label)):
            if label[i] == 'SEX':
                lab_val[i] = 1 if lab_val[i] == 'M' else 0
        return lab_val
    elif cohort == 'mros-v1':
        file_id = filename[-10:-4]
        with open(label_path, 'r') as readFile:
            reader = csv.reader(readFile)
            csv_data = list(reader)
            header = csv_data[0]
            id_list = [item[0].lower() for item in csv_data]
            index_var = [header.index(x) for x in label]
            index_id = id_list.index(file_id)
            lab_val = [csv_data[index_id][x] for x in index_var]
            lab_val = [1 if label[i] == 'gender' else lab_val[i] for i in range(len(label))]
            return lab_val
    elif cohort == 'stages':
        file_id = os.path.basename(filename)[:9]
        df = pd.read_excel(label_path, sheet_name='Sheet1')
        id_list = df.s_code
        # if not in database return empty
        if sum(id_list == file_id) != 1:
            return ['' for x in label]
        lab_val = [df.loc[id_list == file_id][x].values[0] for x in label]
        # Modify values
        for i in range(len(label)):
            if label[i] == 'sex':
                lab_val[i] = 1 if lab_val[i] == 'M' else 0
        return lab_val
    elif cohort == 'ssc':
        file_id = int(os.path.basename(filename)[4:-6])
        df = pd.read_excel(label_path, sheet_name='ssc')
        id_list = df.patid
        # if not in database return empty
        if sum(id_list == file_id) != 1:
            return ['' for x in label]
        lab_val = [df.loc[id_list == file_id][x].values[0] for x in label]
        # Modify values
        for i in range(len(label)):
            if label[i] == 'gender':
                lab_val[i] = 1 if lab_val[i] == 'M' else 0
        return lab_val
    elif cohort == 'sof':
        file_id = int(os.path.basename(filename)[-9:-4])
        df = pd.read_csv(label_path)
        id_list = df.sofid
        # if not in database return empty
        if sum(id_list == file_id) != 1:
            return ['' for x in label]
        lab_val = [df.loc[id_list == file_id][x].values[0] for x in label]
        # Modify values
        for i in range(len(label)):
            if label[i] == 'gender':
                lab_val[i] = 0
        return lab_val
    elif cohort == 'hpap':
        file_id = int(os.path.basename(filename)[-11:-4])
        df = pd.read_csv(label_path)
        id_list = df.nsrrid
        # if not in database return empty
        if sum(id_list == file_id) != 1:
            return ['' for x in label]
        lab_val = [df.loc[id_list == file_id][x].values[0] for x in label]
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
    elif cohort == 'unknown':
        lab_val = []
        for lab in label:
            if lab == 'sex':
                lab_val.append(0.0)
            else:
                lab_val.append(200.0)
        return lab_val
    else:
        return -1
    
    return

def load_ssc(ssc_path, filename, cohort, csv_file = None):
    """Reads hypnogram for a given polysomnography from a specified cohort

    Args:
        ssc_path (str): Path to hypnograms
        filename (str): filename of polysomnography
        cohort (str): Cohort name
        csv_file (csv file, optional): read csv file with hypnogram (not used). Defaults to None.

    Returns:
        ssc (list[int]): Sleep stage scoring (hypnogram) {0: wake, 1: NREM 1, 2: NREM 2, 3: NREM 3, 5: REM}
    """
    if cohort == 'cfs' or cohort == 'shhs-v1' or cohort == 'mros-v1' or cohort == 'sof' or cohort == 'hpap':
        file_id = os.path.basename(filename)[:-4]
        path_ssc_file = os.path.join(ssc_path, file_id + '-profusion.xml')
        if not os.path.isfile(path_ssc_file):
            return -1
        with open(path_ssc_file) as f_xml:
            ssc_dict=xmltodict.parse(f_xml.read())
            ssc_raw = [int(x) for x in ssc_dict['CMPStudyConfig']['SleepStages']['SleepStage']]
            ssc = []
            for x in ssc_raw:
                if x == 0:
                    ssc.append(1)
                elif x == 1:
                    ssc.append(-1)
                elif x == 2:
                    ssc.append(-2)
                elif x == 3:
                    ssc.append(-3)
                elif x == 5:
                    ssc.append(0)
        return ssc
    elif cohort == 'wsc':
        file_id = filename[-18:-4]
        path_ssc_file = os.path.join(ssc_path, file_id + '.STA')
        if not os.path.isfile(path_ssc_file):
            return -1
        df = pd.read_csv(path_ssc_file, delimiter='\t', header=None)
        if len(df.columns) != 3:
            return -1
        ssc_raw = np.array(df.iloc[:, 1])
        ssc = copy.deepcopy(ssc_raw)
        ssc[ssc_raw == 7] = 1
        ssc[ssc_raw == 0] = 1
        ssc[ssc_raw == 1] = -1
        ssc[ssc_raw == 2] = -2
        ssc[ssc_raw == 3] = -3
        ssc[ssc_raw == 5] = 0
        return ssc
    elif cohort == 'ssc':
        file_id = filename[-14:-4]
        path_ssc_file = os.path.join(ssc_path, file_id + '.STA')
        if not os.path.isfile(path_ssc_file):
            return -1
        df = pd.read_csv(path_ssc_file, delim_whitespace=True, header=None)
        if len(df.columns) != 2:
            return -1
        ssc_raw = np.array(df.iloc[:, 1])
        ssc = copy.deepcopy(ssc_raw)
        ssc[ssc_raw == 7] = 1
        ssc[ssc_raw == 0] = 1
        ssc[ssc_raw == 1] = -1
        ssc[ssc_raw == 2] = -2
        ssc[ssc_raw == 3] = -3
        ssc[ssc_raw == 5] = 0
        return ssc
    elif cohort == 'stages':
        file_id = os.path.basename(filename)[:-4]
        path_ssc_file = os.path.join(ssc_path, file_id + '.csv')
        #path_ssc_file = csv_file
        if not os.path.isfile(path_ssc_file):
            return -1
        df = pd.read_csv(path_ssc_file, delimiter=',', usecols=[0,1,2])
        ssc_dur = list(df.iloc[:, 1])
        ssc_raw = list(df.iloc[:, 2])
        ssc = []
        is_ssc_event = False
        for (dur, event) in zip(ssc_dur, ssc_raw):
            # Check for nan (missing column)
            if dur != dur:
                continue

            # Edit false dur
            if dur == 0:
                n_stage = 1
            elif dur == 2592000:
                n_stage = 1
            else:
                n_stage = int(dur // 30)
            
            if event == ' Wake':
                num_stage = 1
                is_ssc_event = True
            elif event == ' UnknownStage':
                num_stage = 1
                is_ssc_event = True
            elif event == ' Stage1':
                num_stage = -1
                is_ssc_event = True
            elif event == ' Stage2':
                num_stage = -2
                is_ssc_event = True
            elif event == ' Stage3':
                num_stage = -3
                is_ssc_event = True
            elif event == ' REM':
                num_stage = 0
                is_ssc_event = True
            else:
                is_ssc_event = False
            if is_ssc_event:
                for i in range(n_stage):
                    ssc.append(num_stage)

        return ssc
    elif cohort == 'unknown':
        ssc = [-1]*(2*60*10)
        return ssc
    else:
        return -1

def load_psg_lights(filename, cohort):
    """Reads lights off/on annotations (not used)

    Args:
        filename (str): polysomnography path
        cohort (str): Cohort name

    Returns:
        lights_off (float): lights off from start of polysomnography in seconds
        lights_on (float): lights on from start of polysomnography in seconds
    """
    if cohort == 'cfsDEPRECATED':
        return -1, -1
    elif cohort == 'mros-v1DEPRECATED':
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
    elif cohort == 'kasselDEPRECATED':
        label = ['Lights_Off','Lights_On']
        lab_val = load_psg_labels(filename, label, cohort)
        idx_off = [i for i, x in enumerate(lab_val[2]) if label[0] in x]
        idx_on = [i for i, x in enumerate(lab_val[2]) if label[1] in x]
        return lab_val[0][idx_off[0]], lab_val[0][idx_on[0]] 
    elif cohort == 'unknown':
        # ONLY WORKS FOR SUBDIR STRUCTURE
        lights_filename = os.path.join(os.path.dirname(filename), 'lights.txt')
        if not os.path.exists(lights_filename):
            return -1, -1
        else:
            df = pd.read_csv(lights_filename)
            return df['Lights_off'].values[0], df['Lights_on'].values[0]
    else:
        return -1

def correct_units(unit_string):
    """Get gain factor for signal unit

    Args:
        unit_string (str): Measurement unit

    Returns:
        g (float): gain factor
    """
    if unit_string == 'uV':
        g = 0.001
    elif unit_string == 'V':
        g = 1.0
    else:
        g = 1.0
    return g

def load_edf_file(filename, channels):
    """Loads an edf file

    Args:
        filename (str): path to edf file
        channels (list[str]): list of channels to read

    Returns:
        data (dict): a data dict with:
            {'x': data as a numpy.array, 
             'fs': frequency of signals, 
             'channels': channels that was read, 
             'q_low': 5th percentile for rescaling, 
             'q_high': 95th percentile for rescaling}
    """
    # Check if EDF exists
    if not os.path.isfile(filename):
        return -1
    # Load EDF file
    try:
        f = pyedflib.EdfReader(filename)
    except Exception as e:
        print(e)
        return -1
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
        elif any([x in unref_channel_alias[channel] for x in channel_labels]) and any([x in ref_channel_alias[channel] for x in channel_labels]):
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
        # Else: only ref channel
        elif any([x in unref_channel_alias[channel] for x in channel_labels]):
            channel_idx = channel_labels.index(next(filter(lambda i: i in unref_channel_alias[channel],channel_labels)))
            if channel in channel_correct_units:
                g = correct_units(f.getPhysicalDimension(channel_idx))
            else:
                g = 1
            # Read signal
            sig = g*f.readSignal(channel_idx)

        # Else empty
        else:
            sig = []
        
        # If not empty
        if len(sig) != 0:
            # Resampling
            fs = fss[channel_idx]
            if fs != des_fs[channel]:
                resample_method = 'linear' if channel is 'OSat' else 'poly'
                sig = psg_resample(sig, channel, fs, resample_method)
            
            # Filter signals
            if hp_fs[channel] != 0:
                sig_filtered = psg_highpass_filter(sig, hp_fs[channel], des_fs[channel], order = 16)
            else:
                sig_filtered = sig
            # Scale signal
            scale_method = 'osat' if channel is 'OSat' else 'soft'
            sig_scaled, q5, q95 = rescale(sig_filtered, scale_method)
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

if __name__ == '__main__':
    # View filter
    _ = psg_highpass([0.1, 15], 128, order=32, plot_opt=1)