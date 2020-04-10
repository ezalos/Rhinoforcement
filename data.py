from node import node
from sklearn import preprocessing
from torch.utils import data
import numpy as np
import copy
import torch


class datapoint():
    def __init__(self, state, policy, value = None):
        self.S = state     #unmodified state
        self.P = policy                    #numpy array
        self.V = value                     #scalar

    def display(self):
        print("State: ")
        self.S.display()
        print("Policy: ", self.P)
        print("Value: ", self.V)
        print("")


class Dataseto(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data = None):
        if (data == None):
            self.data = []
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        S = self.data[index].S
        P = self.data[index].P
        V = self.data[index].V

        return S, P, V
    
    def add_point(self, state, policy):
        '''
            converts state and policy to tensors and adds them
        '''
        self.data.append(datapoint(torch.from_numpy(state.encode_board()).float(), torch.from_numpy(policy).float()))
        return (len(self.data) - 1)
