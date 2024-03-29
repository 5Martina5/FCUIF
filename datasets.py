import os, sys, random
import numpy as np 
import scipy.io as sio

import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, X_qc,view):

        self.X_qc = X_qc
        self.view_size = view

    def __getitem__(self, index):
        current_x_qc = self.X_qc[index]
        client_p = dict()
        feature_dim = 10*self.view_size
        # permutation
        for viewIndex in range(self.view_size):
            P_index = random.sample(range(len(index)), len(index))
            P = np.eye(len(index)).astype('float32')
            P = P[:, P_index]
            client_p[viewIndex] = P
            current_x_qc[:, viewIndex * feature_dim: (viewIndex + 1) * feature_dim] = \
                current_x_qc[:, viewIndex * feature_dim: (viewIndex + 1) * feature_dim][P_index]

        return current_x_qc, client_p

    def __len__(self):
        # return the total size of data
        return  self.X_qc.shape[0]


class Data_Sampler(object):
    """Custom Sampler is required. This sampler prepares batch by passing list of
    data indices instead of running over individual index as in pytorch sampler"""
    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)
        else:
            self.sampler = SequentialSampler(pairs)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = [batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
