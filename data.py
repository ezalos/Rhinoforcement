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
    def __init__(self,data = None):
        if (data == None):
            self.data = []
        else:
            self.data = data
        self.dict = {}

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
        a = self.dict.get(state.stringify())
        if (a == None):
            self.data.append(datapoint(torch.from_numpy(state.encode_board()).float(), torch.from_numpy(policy).float()))
            self.dict[state.stringify()] = len(self.data) - 1
        else:
            self.data[a] = datapoint(torch.from_numpy(state.encode_board()).float(), torch.from_numpy(policy).float())
            return (a)
        return (len(self.data) - 1)

    def reset(self):
        self.data = []
        self.dict = {}
class DataHandler(data.Dataset):
    def __init__(self, args): #optimize handling by popping from training set on removal instead of rebuilding
        self.batch_size = args.batch_size
        self.maxHistory = args.maxHistory
        self.datasets = []
        #self.trainingSet = Dataseto()
        self.current = 0

    def add_dataset(self, dataset):
        if (self.current >= self.maxHistory - 1):
            self.datasets.pop(0)
            self.current -= 1
        self.datasets.append(dataset)
        self.current += 1

    def get_training_set(self):
        out = []
        for d in self.datasets:
            if (d != None):
                out.extend(d.data)
        return Dataseto(out)

    def get_data_loader(self):
        return torch.utils.data.DataLoader(self.get_training_set(), batch_size=self.batch_size, shuffle=True, num_workers=1)
