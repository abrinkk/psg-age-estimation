import h5py
import numpy as np
import os

from config import Config

def main():
    
    # Config
    config = Config()
    config.F_train_dir = "H:\\nAge\\test_mortality_F_com5"

    # folders
    in_feat_paths = [os.path.join(config.data_dir, 'test_mortality_F_eeg5'), 
                  os.path.join(config.data_dir, 'test_mortality_F_eegeogemg5'), 
                  os.path.join(config.data_dir, 'test_mortality_F_ecg5'), 
                  os.path.join(config.data_dir, 'test_mortality_F_resp5')]

    # Get record paths
    filenames_all = [f for f in os.listdir(in_feat_paths[0]) if os.path.isfile(os.path.join(in_feat_paths[0], f))]
    #print(filenames_all)

    for record in filenames_all:
        # Get features
        feat = []
        for in_feat_path in in_feat_paths:
            in_filename = os.path.join(in_feat_path, record)
            with h5py.File(in_filename, "r") as f:
                # Load features
                feat.append(np.array(f['PSG']))
                # Load attributes
                attrs = {}
                for k, v in f.attrs.items():
                    attrs[k] = v

        # Save features
        out_filename = os.path.join(config.F_train_dir, record)
        with h5py.File(out_filename, "w") as f:
            # Add datasets
            f.create_dataset("PSG", data=np.concatenate(feat, 1), dtype='f4')
            # Attributes
            for key_a, v in attrs.items():
                f.attrs[key_a] = v

if __name__ == "__main__":
    main()