# Setup imports
import os
import numpy as np
import pandas as pd
import copy

from utils.get_h5_data import get_h5_ssc

from config import Config

# Config
config = Config()
config.epoch_size = int(5.0*128*60)
config.pretrain_dir = "H:\\nAge\\test_mortality"

# Filepaths
filepath = os.path.join(config.pretrain_dir)
filenames_all = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]

# preallocate metrics
patient_info = []
i = 0

for record in filenames_all:
    ssc = get_h5_ssc(os.path.join(filepath, record))

    count = 0
    after_sl = 0
    sleep_latency = 0
    waso = 0
    for stage in ssc:
        if stage == 1:
            count += 30
        elif after_sl == 0:
            sleep_latency = copy.copy(count)
            after_sl = 1
            count = 0
        else:
            waso = copy.copy(count)

    n1t = sum(ssc == -1) * 30
    n2t = sum(ssc == -2) * 30
    n3t = sum(ssc == -3) * 30
    remt = sum(ssc == 0) * 30
    wt = sum(ssc == 1) * 30

    tst = n1t + n2t + n3t + remt
    se = tst/(tst + wt)
    n1p = n1t/tst
    n2p = n2t/tst
    n3p = n3t/tst
    remp = remt/tst
    summary = {'record': record, 'sleep_time': tst, 
                   'n1t': n1t, 'n2t': n2t, 'n3t': n3t, 'remt': remt, 
                   'n1p': n1p, 'n2p': n2p, 'n3p': n3p, 'remp': remp,
                   'waso': waso, 'sleep_latency': sleep_latency, 'se': se}
    patient_info.append(summary)
    i+= 1
    print('Patient {} out of {}'.format(i, len(filenames_all)))

    df_ssc = pd.DataFrame(patient_info)
    
    df_ssc.to_excel('H:\\nAge\\ssc_stats_test_mortality.xlsx', float_format='%.5f')
    #df_ssc.to_excel('H:\\nAge\\ssc_stats_test_extra.xlsx', float_format='%.5f')
    #df_ssc.to_excel('/home/users/abk26/SleepAge/Scripts/data/ssc_stats.xlsx', float_format='%.5f')