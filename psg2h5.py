import h5py
import numpy as np
import os
import argparse
import psg_reader
from config import Config
from zipfile import ZipFile

class psg_writer():
    def __init__(self, input_folder, output_folder, channels, labels, cohort, n = -1, split = [0.7, 0.1, 0.2], split_name = 'all', overwrite = 0, num_missing_max = 2, light = True):
        """A class to write a folder of edf polysomnography files to h5 files

        Args:
            input_folder (str): path to edf files
            output_folder (str): path to h5 files
            channels (list[str]): list of channels to extract
            labels (list[str]): list of labels to extract
            cohort (str): Cohort name
            n (int, optional): number of files to extract from folder (n=-1 extracts all). Defaults to -1.
            split (list, optional): split of training, validation, and test (not used). Defaults to [0.7, 0.1, 0.2].
            split_name (str, optional): name of subfolder to extract to. Defaults to 'all'.
            overwrite (int, optional): if 1 then it overwrites h5 files. Defaults to 0.
            num_missing_max (int, optional): if more than this number of channels are missing then it is skipped 
                (by default channels are zero-padded if missing). Defaults to 2.
            light (int, optional): To use lights off/on (not used). Defaults to 0.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.channels = channels
        self.labels = labels
        self.split = split
        self.cohort = cohort
        self.overwrite = overwrite
        self.num_missing_max = num_missing_max
        self.light = light
        self.n = n
        self.config = Config()
        self.extension = ".edf"
        self.split_name = split_name
        self.check_subfolders = False
        self.check_subfolders_ssc = False
        if self.cohort == 'cfs':
            self.ssc_path = self.config.cfs_ssc_path
        elif self.cohort == 'mros-v1':
            self.ssc_path = self.config.mros_ssc_path
        elif self.cohort == 'shhs-v1':
            self.ssc_path = self.config.shhs_ssc_path
        elif self.cohort == 'wsc':
            self.ssc_path = self.config.wsc_ssc_path
        elif self.cohort == 'stages':
            self.ssc_path = self.config.stages_ssc_path
        elif self.cohort == 'ssc':
            self.ssc_path = self.config.ssc_ssc_path
        elif self.cohort == 'sof':
            self.ssc_path = self.config.sof_ssc_path
        elif self.cohort == 'hpap':
            self.ssc_path = self.config.hpap_ssc_path
        else:
            self.ssc_path = ''

        # Zip-files (deprecated)
        if self.cohort == 'stages':
            self.extension = ".edf"

        # Check subfolders
        if self.cohort == 'stages':
            self.check_subfolders = True
            self.check_subfolders_ssc = True
            subfolders = ('MAYO', 'STNF', 'STLK', 'MSTR', 'MSNF', 'MSMI', 'GSDV', 'GS', 'BOGN', 'MSQW', 'MSTH')

        # Check subfolders
        if self.cohort == 'sub':
            self.check_subfolders = True
            self.check_subfolders_ssc = True
            subfolders = [d for d in os.listdir(self.input_folder) if os.path.isdir(os.path.join(self.input_folder, d))]
            self.cohort = 'unknown'

        # PSGs to convert
        if self.check_subfolders:
            self.filenames = list()
            self.subfolders = list()
            for (dirpath, _, dirfile) in os.walk(input_folder):
                if not isinstance(dirfile, list):
                    dirfile = [dirfile]
                for f in dirfile:
                    if os.path.isfile(os.path.join(dirpath, f)) and os.path.split(dirpath)[-1] in subfolders and f.lower().endswith(self.extension):
                        self.subfolders.append(os.path.split(dirpath)[-1])
                        self.filenames.append(f)
        else:
            self.filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(self.extension)]
        
        
        # Shuffle
        np.random.seed(seed = 0)
        #idx_file_shuffle = np.arange(len(self.filenames))
        #self.filenames = list(np.array(self.filenames)[idx_file_shuffle])
        #if self.check_subfolders:
        #    self.subfolders = list(np.array(self.subfolders)[idx_file_shuffle])
        #np.random.shuffle(self.filenames)
        
        # If not overwrite, then list those that does not exist
        filenames_out = [f for f in os.listdir(os.path.join(output_folder, self.split_name)) if os.path.isfile(os.path.join(output_folder, self.split_name, f))]
        # filenames_out = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
        # filenames_out.extend([f for f in os.listdir(os.path.join(output_folder, 'train')) if os.path.isfile(os.path.join(output_folder,'train', f))])
        # filenames_out.extend([f for f in os.listdir(os.path.join(output_folder, 'val')) if os.path.isfile(os.path.join(output_folder,'val', f))])
        # filenames_out.extend([f for f in os.listdir(os.path.join(output_folder, 'test')) if os.path.isfile(os.path.join(output_folder,'test', f))])


        if overwrite == 0:
            if self.check_subfolders:
                self.subfolders = [x for idx, x in enumerate(self.subfolders) if self.filenames[idx][:-4] not in map(lambda x: x[:-5], filenames_out)]
            self.filenames = list(filter(lambda x: x[:-4] not in list(map(lambda x: x[:-5], filenames_out)), self.filenames))
        # Number of files to convert
        if self.n == -1:
            self.n = len(self.filenames)

    
    def write_psg_h5(self, filename, output_filename):
        """write one edf file to h5 format

        Args:
            filename (str): path to edf file
            output_filename (str): path to h5 file
        """
        # Check subfolders
        if self.check_subfolders_ssc:
            if self.cohort == 'stages':
                ssc_path = os.path.split(filename)[0]
            else:
                ssc_path = self.ssc_path
        else:
            ssc_path = self.ssc_path
        
        # read psg
        if filename.endswith(".zip"):
            with ZipFile(filename, 'r') as zip_f:
                for f in zip_f.namelist():
                    if f.endswith(".edf"):
                        edf_file = zip_f.extract(f, self.config.tmp_dir)
                        try:
                            data = psg_reader.load_edf_file(edf_file, self.channels)
                            os.remove(edf_file)
                        except:
                            os.remove(edf_file)
                            return
                    if f.endswith(".csv"):
                        csv_file = zip_f.extract(f, self.config.tmp_dir)
                        try:
                            ssc = psg_reader.load_ssc(self.ssc_path, filename, self.cohort, csv_file)
                            os.remove(csv_file)
                        except:
                            os.remove(csv_file)
                            return
        else:
            data = psg_reader.load_edf_file(filename, self.channels)
            ssc = psg_reader.load_ssc(ssc_path, filename, self.cohort)
        if data == -1 or np.isscalar(ssc):
            # Exclude if error
            return
        elif data['x'][0].shape[0] < data['fs'][0]*60*60*3:
            # Exclude data with total recording time less than 3 hours
            return
        fs = data['fs']
        sig = data['x']
        q_low = data['q_low']
        q_high = data['q_high']
        sig = np.array(sig)
        
        # Select lights off - lights on (Legacy code)
        if self.light == 1:
            lights = psg_reader.load_psg_lights(filename, self.cohort)
            if lights[0] != -1 and lights[1] != -1:
                sig = sig[:, int(lights[0]*fs[0]):min([int(lights[1]*fs[0]), len(sig[0])])]
        
        # read labels
        if self.cohort == 'cfs':
            label_path = self.config.cfs_ds_path
        elif self.cohort == 'mros-v1':
            label_path = self.config.mros_ds_path
        elif self.cohort == 'shhs-v1':
            label_path = self.config.shhs_ds_path
        elif self.cohort == 'wsc':
            label_path = self.config.wsc_ds_path
        elif self.cohort == 'stages':
            label_path = self.config.stages_ds_path
        elif self.cohort == 'ssc':
            label_path = self.config.ssc_ds_path
        elif self.cohort == 'sof':
            label_path = self.config.sof_ds_path
        elif self.cohort == 'hpap':
            label_path = self.config.hpap_ds_path
        else:
            label_path = ''
        lab_val = psg_reader.load_psg_labels(filename, self.labels, self.cohort, label_path)
        
        # If labels are missing, then dont save and return
        if any([1 if i == '' else 0 for i in lab_val]):
            return

        # If labels are nan, then dont save and return
        if any([1 if i != i else 0 for i in lab_val]):
            return

        # If signals are nan, then dont save and return
        if (sig != sig).any():
            return

        # If too many signals are missing, then dont save and return
        if np.sum(np.count_nonzero(sig, 1) == 0) > self.num_missing_max:
            return

        # If age is zero, then dont save and return
        if lab_val[0] == 0:
            return

        # If not number values, dont save and return
        for lab_val_el in lab_val:
            try:
                float(lab_val_el)   
            except ValueError:
                return

        # Save data as h5 files
        with h5py.File(output_filename, "w") as f:
            # Save PSG
            f.create_dataset("PSG", data = sig, dtype='f4', chunks = (len(self.channels), 128*60*5))
            f.create_dataset("SSC", data = ssc, dtype='f4')
            # Save labels
            for i, l in enumerate(self.labels):
                f.attrs[l] = float(lab_val[i])
            # Save data quantiles
            f.attrs['q_low'] = q_low
            f.attrs['q_high'] = q_high
        return

    def psg_folder_to_h5(self):
        """Extracts all h5 files
        """
        # if self.use_split_list:
            # Read in lists
            # df_train = pd.read_csv(self.config.list_split_train, delimiter=',')
            # df_val = pd.read_csv(self.config.list_split_val, delimiter=',')
            # df_test = pd.read_csv(self.config.list_split_test, delimiter=',')
            # list_train = list(df_train.iloc[:, 4])
            # list_train = [x.lower() for x in list_train]
            # list_val = list(df_val.iloc[:, 4])
            # list_val = [x.lower() for x in list_val]
            # list_test = list(df_test.iloc[:,4])
            # list_test = [x.lower() for x in list_test]

        # Iterate files
        for i in range(self.n):
            
            if self.check_subfolders:
                filename = os.path.join(self.input_folder, self.subfolders[i], self.filenames[i])
            else:
                filename = os.path.join(self.input_folder, self.filenames[i])

            # if self.use_split_list:
            #     # Match to style of list
            #     if self.cohort == 'cfs':
            #         filelistname = os.path.basename(filename)[:-4].lower()
            #     elif self.cohort == 'mros-v1':
            #         filelistname = os.path.basename(filename)[:-4].lower()
            #     elif self.cohort == 'shhs-v1':
            #         filelistname = os.path.basename(filename)[:-4].lower()
            #     elif self.cohort == 'wsc':
            #         filelistname = os.path.basename(filename)[:7].lower()
            #     elif self.cohort == 'stages':
            #         filelistname = os.path.basename(filename).lower()
            #     elif self.cohort == 'ssc':
            #         filelistname = os.path.basename(filename)[:-4].lower()

            #     if filelistname in list_train:
            #         split_name = 'train'
            #     elif filelistname in list_val:
            #         split_name = 'val'
            #     elif filelistname in list_test:
            #         split_name = 'test'
            #     else:
            #         split_name = 'skip'
            # else:
            #     if i >= (self.split[0] + self.split[1])*self.n:
            #         split_name = 'test'
            #     elif i >= self.split[0]*self.n:
            #         split_name = 'val'
            #     else:
            #         split_name = 'train'
            if self.cohort == 'unknown':
                output_filename = os.path.join(self.output_folder, self.split_name, self.subfolders[i] + '.hdf5')
                #output_filename = os.path.join(self.output_folder, self.split_name, self.filenames[i][:-4] + '.hdf5')
            else:
                output_filename = os.path.join(self.output_folder, self.split_name, self.filenames[i][:-4] + '.hdf5')
            # if split_name != 'skip':
            self.write_psg_h5(filename, output_filename)
            if i % 10 == 0:
                print('PSG {} of {}'.format(i + 1, self.n))

def main(args):
    psg_h5_writer = psg_writer(args.input_folder, args.output_folder, args.channels, args.labels, args.cohort, n = args.n, split = args.split, split_name=args.split_name, overwrite = args.overwrite, num_missing_max = args.max_missing_channel)
    psg_h5_writer.psg_folder_to_h5()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Age Estimation from Polysomnograms.')
    parser.add_argument('--input_folder', type=str, default='H:\\STAGES\\polysomnograms\\',
                        help='folder with edf files to preprocess.')
    parser.add_argument('--output_folder', type=str, default='H:\\nAge\\',
                        help='folder for preprocessed h5 files')
    parser.add_argument('--channels', type=list, default=['C3','C4','EOGL','EOGR','ECG','Chin','Leg','Airflow','NasalP','Abd','Chest','OSat'],
                        help='To train preivously pretrained model')
    parser.add_argument('--labels', type=list, default=['age','bmi','sex'],
                        help='Save/overwrite model F features')
    parser.add_argument('--cohort', type=str, default='stages',
                        help='To train model.')
    parser.add_argument('--n', type=int, default=-1,
                        help='Number of files to preprocess')
    parser.add_argument('--split', nargs=3, type=float, default=[0.0, 0.0, 1.0],
                        help='train/val/test split')
    parser.add_argument('--split_name', type=str, default='all',
                        help='split name')
    parser.add_argument('--overwrite', type=bool, default=False,
                        help='overwrite previously written h5 files')
    parser.add_argument('--max_missing_channel', type=int, default=2,
                        help='number of channels allowed to be missing')

    args = parser.parse_args()
    print(args)
    main(args)
    
    #input_folder = 'G:\\cfs\\polysomnography\\edfs\\'
    #cohort = 'cfs'
    #input_folder = 'H:\\shhs\\polysomnography\\edfs\\shhs1\\'
    #cohort = 'shhs-v1'
    #input_folder = 'G:\\wsc\\polysomnography\\edfs\\'
    #cohort = 'wsc'
    #output_folder = 'H:\\nAge\\'
    #channels = ['C3','C4','EOGL','EOGR','ECG','Chin','Leg','Airflow','NasalP','Abd','Chest','OSat']
    #labels = ['age','bmi','sex','ess']
    #psg_h5_writer = psg_writer(input_folder, output_folder, channels, labels, cohort, n = -1, split = [0.25, 0.05, 0.7], overwrite = 0)
    #psg_h5_writer = psg_writer(input_folder, output_folder, channels, labels, cohort, n = -1, split = [1.0, 0.0, 0.0], overwrite = 0)
    
