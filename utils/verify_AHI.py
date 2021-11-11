# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:45:09 2020

@author: umaer
"""

import pandas as pd
import numpy as np
import datetime as dt
import glob
import zipfile
import copy
pd.options.mode.chained_assignment = None


def verify_ahi(files_path):
    
    patient_info = []
    i = 0
    files = glob.glob(files_path + '*.zip')
    for file in files:
        zf = zipfile.ZipFile(file, 'r')
        subject_code = zf.namelist()[0][:-4]
        file_name = subject_code + '.csv'
        df = pd.read_csv(zf.open(file_name), error_bad_lines=False, usecols=[0,1,2])
        s = pd.Series(df.Event)
                
        if df.empty:
            continue
        
        if s.isin([' LightsOff']).sum() == 1:
            idx_loff = s[s == ' LightsOff'].index[0]
        else: 
            idx_loff = 0
        
        if s.isin([' LightsOn']).sum() == 1:
            idx_lon = s[s == ' LightsOn'].index[0]
        else:
            idx_lon = len(s)-1
    
        s = s[idx_loff:idx_lon+1]
        df = df.loc[idx_loff:idx_lon+1]
            
        stages = df.loc[df['Event'].isin([' Wake',' Stage1', ' Stage2', ' Stage3', ' REM'])]
        stages['Duration (seconds)'] = stages['Duration (seconds)'].replace(2592000, 30)
        stages['Duration (seconds)'] = stages['Duration (seconds)'].replace(0, 30)      

        count = 0
        after_sl = 0
        sleep_latency = 0
        waso = 0
        for row in stages.iterrows():
            if row[1]['Event'] == ' Wake':
                count += row[1]['Duration (seconds)']
            elif after_sl == 0:
                sleep_latency = copy.copy(count)
                after_sl = 1
                count = 0
            else:
                waso = copy.copy(count)

        W_stages = stages.loc[df['Event'].isin([' Wake'])]
        N1_stages = stages.loc[df['Event'].isin([' Stage1'])]
        N2_stages = stages.loc[df['Event'].isin([' Stage2'])]
        N3_stages = stages.loc[df['Event'].isin([' Stage3'])]
        REM_stages = stages.loc[df['Event'].isin([' REM'])]
        
        n1t = N1_stages['Duration (seconds)'].sum()
        n2t = N2_stages['Duration (seconds)'].sum()
        n3t = N3_stages['Duration (seconds)'].sum()
        remt = REM_stages['Duration (seconds)'].sum()
        
        sleep_time = n1t + n2t + n3t + remt

        se = sleep_time / (W_stages['Duration (seconds)'].sum() + sleep_time)
        
        n1p = n1t / sleep_time
        n2p = n2t / sleep_time
        n3p = n3t / sleep_time
        remp = remt / sleep_time

        events = pd.Index(df['Event'])
        
        if ' ObstructiveApnea' in events: 
            obs_idx = events.get_loc(' ObstructiveApnea')
            obs_events = df.iloc[obs_idx]
            n_obs = len(obs_events)
            obs_events_dur = obs_events['Duration (seconds)']
            mean_obs_events_dur = obs_events_dur.mean()
        else:
            mean_obs_events_dur = np.nan
            n_obs = 0
            
        if ' CentralApnea' in events: 
            cen_idx = events.get_loc(' CentralApnea')
            cen_events = df.iloc[cen_idx]
            n_cen = len(cen_events)
            cen_events_dur = cen_events['Duration (seconds)']
            mean_cen_events_dur = cen_events_dur.mean()
        else:
            mean_cen_events_dur = np.nan
            n_cen = 0
            
        if ' MixedApnea' in events:
            mix_idx = events.get_loc(' MixedApnea')
            mix_events = df.iloc[mix_idx]
            n_mix = len(mix_events)
            mix_events_dur = mix_events['Duration (seconds)']
            mean_mix_events_dur = mix_events_dur.mean()
        else:
            mean_mix_events_dur = np.nan
            n_mix = 0
                        
        if ' Hypopnea' in events:
            hyp_idx = events.get_loc(' Hypopnea')
            hyp_events = df.iloc[hyp_idx]
            n_hyp = len(hyp_events)
            hyp_events_dur = hyp_events['Duration (seconds)']
            mean_hyp_events_dur = hyp_events_dur.mean()
        else:
            mean_hyp_events_dur = np.nan
            n_hyp = 0
            
        if ' Desaturation' in events:
            desat_idx = events.get_loc(' Desaturation')
            desat_events = df.iloc[desat_idx]
            n_desat = len(desat_events)
            desat_events_dur = desat_events['Duration (seconds)']
            mean_desat_events_dur = desat_events_dur.mean()
        else:
            mean_desat_events_dur = np.nan
            n_desat = 0
            
        if ' RERA' in events:
            rera_idx = events.get_loc(' RERA')
            rera_events = df.iloc[rera_idx]
            n_rera = len(rera_events)
            rera_events_dur = rera_events['Duration (seconds)']
            mean_rera_events_dur = rera_events_dur.mean()
        else:
            mean_rera_events_dur = np.nan
            n_rera = 0
            
        ahi = (n_obs + n_mix + n_hyp)/(sleep_time/3600)
    
        summary = {'s_code': subject_code, 'n_obs': n_obs, 'n_cen': n_cen, 'n_mix': n_mix,
                   'n_hyp': n_hyp, 'n_desat': n_desat, 'n_rera': n_rera, 
                   'obs_dur': mean_obs_events_dur, 'cen_dur': mean_cen_events_dur, 
                   'mix_dur': mean_mix_events_dur, 'hyp_dur': mean_hyp_events_dur, 
                   'desat_dur': mean_desat_events_dur,'rera_dur': mean_rera_events_dur, 
                   'sleep_time': sleep_time, 'ahi': ahi, 
                   'n1t': n1t, 'n2t': n2t, 'n3t': n3t, 'remt': remt, 
                   'n1p': n1p, 'n2p': n2p, 'n3p': n3p, 'remp': remp,
                   'waso': waso, 'sleep_latency': sleep_latency, 'se': se}
        patient_info.append(summary)
        i+= 1
        print('Patient {} out of {}'.format(i, len(files)))
        
    df_events = pd.DataFrame(patient_info)
    df_events = df_events.dropna(subset=['ahi'])
    df_events = df_events[df_events['ahi'] != np.inf]
    df_events = df_events[df_events['ahi'] < 200]
    
    #df_events.to_excel('H:\\STAGES\\PatientDemographicsAll_test.xlsx', float_format='%.5f')
    df_events.to_excel('/home/users/abk26/SleepAge/Scripts/data/PatientDemographicsAll.xlsx', float_format='%.5f')
        
    return df_events

    
#files_path = 'H:\\STAGES\\polysomnograms\\'
files_path = '/oak/stanford/groups/mignot/psg/STAGES/deid/'
demographics = verify_ahi(files_path)
