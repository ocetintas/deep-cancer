import os
import numpy as np
import shutil
import pickle

suffix = "3_"   # Suffix to add foler names in order to prevent name collisions

npz_path = "Dataset/npz_data/"
dir_list = os.listdir(npz_path)
dir_list.sort()
zero_count = 0
zero_list = []
print("Sorting ended")
for dir in dir_list:
    print(dir)
    open_np = np.load(npz_path+dir+"/Data_0001.npz")
    data = open_np["data"]
    if (np.all(data == np.zeros_like(data))):
        zero_count += 1
        zero_list.append(dir)
        shutil.rmtree(npz_path+dir)
    else:
        os.rename(npz_path + dir, npz_path + suffix + dir)
print(zero_count, " Folders has been deleted!")
print("Deleted folders: ", zero_list)
print(len(dir_list))
