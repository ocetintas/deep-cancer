import os
import numpy as np
import pickle
import random
import torch

import time

class BatchLoader:

    def __init__(self, device, data_path="../neural_net/Dataset/npz_data/", batch_size=32):
        self.device = device
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_dir, self.validation_dir, self.test_dir, self.data_shape = self.file_path()
        self.mini_batches = []
        self.val_batches = []
        self.test_batches = []
        # Normalization of the ground truth labels
        self.minmax_dict = {
            "Dw": (3e-4, 13e-4),
            "rho": (5e-3, 13e-2),
            "icx": (0.4, 0.8),
            "icy": (0.4, 0.8),
            "icz": (0.2, 0.6),
            "uth": (0.3, 1.0),
            "Tend": (50.0, 1000.0)
        }

    def file_path(self):
        # Save all data directories
        # Train dir
        train_dir = os.listdir(self.data_path+"train/")
        train_dir = [self.data_path + "train/" + d + "/Data_0001.npz" for d in train_dir]

        # Validation dir
        validation_dir = os.listdir(self.data_path+"validation/")
        validation_dir = [self.data_path + "validation/" + d + "/Data_0001.npz" for d in validation_dir]

        # Test dir
        test_dir = os.listdir(self.data_path+"test/")
        test_dir = [self.data_path + "test/" + d + "/Data_0001.npz" for d in test_dir]

        # Shape of the input
        file = np.load(train_dir[0])
        data = file["thr_data"]

        return train_dir, validation_dir, test_dir, (data.shape[0], data.shape[1], data.shape[2])
    def create_train_batches(self):
        mini_batches = []
        random.shuffle(self.train_dir)
        n_minibatch = int(len(self.train_dir) / self.batch_size)
        for i in range(n_minibatch):
            mini_batches.append(self.train_dir[i*self.batch_size:(i+1)*self.batch_size])

        if len(self.train_dir) % self.batch_size != 0:
            mini_batches.append(self.train_dir[(i+1)*self.batch_size:])

        self.mini_batches = mini_batches

    def create_testval_batches(self):
        val_batches = []
        test_batches = []

        val_n = int(len(self.validation_dir) / self.batch_size)
        test_n = int(len(self.test_dir) / self.batch_size)

        for i in range(val_n):
            val_batches.append(self.validation_dir[i*self.batch_size:(i+1)*self.batch_size])
        if len(self.validation_dir) % self.batch_size != 0:
            val_batches.append(self.validation_dir[(i + 1) * self.batch_size:])

        for i in range(test_n):
            test_batches.append(self.test_dir[i*self.batch_size:(i+1)*self.batch_size])
        if len(self.test_dir) % self.batch_size != 0:
            test_batches.append(self.test_dir[(i + 1) * self.batch_size:])

        self.val_batches = val_batches
        self.test_batches = test_batches

    def get_XY(self, dir_list):
        # X and Y are PyTorch Tensors
        X = torch.zeros((len(dir_list), 1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))     # NCDHW
        Y = torch.zeros((len(dir_list), 7))

        # Create X and Y
        for i, d in enumerate(dir_list):
            # X
            file = np.load(d)
            data = file["thr_data"]
            data_torch = torch.from_numpy(data)
            X[i, 0, :] = data_torch.permute(3, 0, 1, 2)  # Change dimensions into Pytorch NCDHW format

            # Y
            path, _ = os.path.split(d)
            with open(path + "/parameter_tag.pkl", "rb") as f:
                data = pickle.load(f)
            self.normalize_data(data)
            y = torch.tensor([data["Dw"], data["rho"], data["icx"], data["icy"], data["icz"], data["uth"],
                          data["Tend"]]).view(1, -1)
            Y[i, :] = y

        return X.to(self.device), Y.to(self.device)

    def normalize_data(self, data):
        for key, value in data.items():
            data[key] = (data[key] - self.minmax_dict[key][0]) / (self.minmax_dict[key][1] - self.minmax_dict[key][0])

if __name__ == "__main__":
    loader = BatchLoader(device="cpu")
    loader.create_testval_batches()
    X, Y = loader.get_XY(loader.test_batches[0])
    print(X.shape, Y.shape)