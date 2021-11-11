import h5py
import numpy as np
import os
import psg_reader

class psg_writer():
    def __init__(self, input_folder, output_folder, channels, labels, cohort, n = -1, split = [0.7, 0.1, 0.2], overwrite = 0, light = 0):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.channels = channels
        self.labels = labels
        self.split = split
        self.cohort = cohort
        self.overwrite = overwrite
        self.light = light
        self.n = n
        
        # PSGs to convert
        self.filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
        # Shuffle
        np.random.seed(seed = 0)
        np.random.shuffle(self.filenames)
        # If not overwrite, then list those that does not exist
        filenames_out = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
        filenames_out.extend([f for f in os.listdir(os.path.join(output_folder, 'train')) if os.path.isfile(os.path.join(output_folder,'train', f))])
        filenames_out.extend([f for f in os.listdir(os.path.join(output_folder, 'val')) if os.path.isfile(os.path.join(output_folder,'val', f))])
        filenames_out.extend([f for f in os.listdir(os.path.join(output_folder, 'test')) if os.path.isfile(os.path.join(output_folder,'test', f))])

        if overwrite == 0:
            self.filenames = list(filter(lambda x: x[:-4] not in list(map(lambda x: x[:-5], filenames_out)), self.filenames))
        # Number of files to convert
        if self.n == -1:
            self.n = len(self.filenames)

    
    def write_psg_h5(self, filename, output_filename):
        
        # read psg
        data = psg_reader.load_edf_file(filename, self.channels)
        if data == -1:
            return
        fs = data['fs']
        sig = data['x']
        q_low = data['q_low']
        q_high = data['q_high']
        sig = np.array(sig)
        
        # Select lights off - lights on
        if self.light == 1:
            lights = psg_reader.load_psg_lights(filename, self.cohort)
            sig = sig[:,int(lights[0]*fs[0]):min([int(lights[1]*fs[0]),len(sig[0])])]
        
        # read labels
        lab_val = psg_reader.load_psg_labels(filename, self.labels, self.cohort)
        
        # If labels are missing, then dont save and return
        if any([1 if i == '' else 0 for i in lab_val]):
            return

        # Save data as h5 files
        with h5py.File(output_filename, "w") as f:
            # Save PSG
            f.create_dataset("PSG", data = sig, dtype='f4', chunks = (len(self.channels), 128*60*5))
            # Save labels
            for i, l in enumerate(self.labels):
                f.attrs[l] = float(lab_val[i])
            # Save data quantiles
            f.attrs['q_low'] = q_low
            f.attrs['q_high'] = q_high
        return

    def psg_folder_to_h5(self):
        # Iterate files
        for i in range(self.n):
            if i >= (self.split[0] + self.split[1])*self.n:
                split_name = 'test\\'
            elif i >= self.split[0]*self.n:
                split_name = 'val\\'
            else:
                split_name = 'train\\'
                
            filename = self.input_folder + self.filenames[i]
            output_filename = self.output_folder + split_name + self.filenames[i][:-4] + '.hdf5'
            self.write_psg_h5(filename, output_filename)
            if i % 10 == 0:
                print('PSG {} of {}'.format(i + 1, self.n))

if __name__ == "__main__":
    # input_folder = 'G:\\cfs\\polysomnography\\edfs\\'
    # cohort = 'cfs'
    input_folder = 'H:\\shhs\\polysomnography\\edfs\\shhs1\\'
    cohort = 'shhs-v1'
    output_folder = 'H:\\nAge\\'
    channels = ['C3','C4','EOGL','EOGR','ECG','Chin','Leg','Airflow','NasalP','Abd','Chest','OSat']
    labels = ['age','bmi','sex','ess']
    psg_h5_writer = psg_writer(input_folder, output_folder, channels, labels, cohort, n = -1, split = [0.70, 0.10, 0.2], overwrite = 0)
    psg_h5_writer.psg_folder_to_h5()
