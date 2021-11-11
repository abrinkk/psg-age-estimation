import argparse
import subprocess
import os
import pandas as pd

from config import Config

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser(description='Estimate age from polysomnograms with all models.')
    # parser.add_argument('--input_folder', type=str, default='C:\\Users\\andre\\Dropbox\\Phd\\SleepAge\\Scripts\\data\\age_experiment_edf\\',
    #                     help='folder with edf files to preprocess.')
    # parser.add_argument('--output_folder', type=str, default='C:\\Users\\andre\\Dropbox\\Phd\\SleepAge\\Scripts\\data\\age_experiment_h5\\',
    #                     help='folder for preprocessed h5 files')
    parser.add_argument('--input_folder', type=str, default='G:\\GlostrupRBD\\',
                        help='folder with edf files to preprocess.')
    parser.add_argument('--output_folder', type=str, default='H:\\RBD_Age\\h5_dcsm\\',
                        help='folder for preprocessed h5 files')
    parser.add_argument('--sub_folders', type=str, default='y',
                        help='folder for preprocessed h5 files')
    args = parser.parse_args()
    edf_folder = args.input_folder
    h5_input_folder = args.output_folder
    h5_folder = os.path.dirname(os.path.normpath(args.output_folder))
    split_name = os.path.basename(os.path.normpath(args.output_folder))
    sub_folders = 'sub' if args.sub_folders == 'y' else 'unknown'

    # Preprocessing
    subprocess.call(['python', 'psg2h5.py', '--input_folder', edf_folder, 
                                            '--output_folder', h5_folder, 
                                            '--cohort', sub_folders, 
                                            '--split_name', split_name])

    # Age estimation
    only_eeg = [1, 2, 3, 4]
    model_name = ['eeg5', 'eegeogemg5', 'ecg5', 'resp5']
    for i in range(len(model_name)):
        subprocess.call(['python', 'age_estimate.py', '--input_folder', h5_input_folder, 
                                                      '--pre_hyperparam', '1e-3', '1e-5', '0.75', '0.1', '1', '32', '5', '0', str(only_eeg[i]), 
                                                      '--model_name', model_name[i]])

    # Collect estimates and write to csv
    config = Config()
    df_all = []
    for i in range(len(model_name)):
        model_dir = os.path.join(config.data_dir, 'model_' + model_name[i])
        df_path = os.path.join(model_dir, 'metrics_' + split_name + '.csv')
        df_i = pd.read_csv(df_path)
        df_all.append(df_i)

    df_combined = pd.DataFrame(df_i['record'])
    age_p_combined = []
    for i in range(len(model_name)):
        age_p_combined.append(df_all[i]['age_p'].values)
        df_combined.insert(i + 1, 'age_p_' + model_name[i], df_all[i]['age_p'].values, True)
    df_combined.insert(i+2, 'age_p_avg', sum(age_p_combined)/len(age_p_combined), True)
    df_combined.to_csv(os.path.join(h5_folder, 'age_pred_ ' + split_name + '.csv'), index=False)