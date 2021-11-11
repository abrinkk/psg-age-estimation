from zipfile import ZipFile
import pyedflib
import os

file_name = 'H:\\STAGES\\polysomnograms\\STNF00172_1_EDFAndScore.zip'
tmp_folder = 'H:\\nAge\\tmp\\'

with ZipFile(file_name, 'r') as zip_f:
    for f in zip_f.namelist():
        if f.endswith(".edf"):
            edf_file = zip_f.extract(f, tmp_folder)
            with pyedflib.EdfReader(edf_file) as data:
                dir(data)
            os.remove(edf_file)
a = 1