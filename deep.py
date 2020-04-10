#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import numpy as np

class board_data(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return np.int64(self.X[idx].transpose(2,0,1)), self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*6*7, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6*7*32, 7)

    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 3*6*7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 6*7*32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v

class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(10):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()

    def forward(self,s):
        s = self.conv(s)
        for block in range(10):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def cross_entropy(self, pred, soft_targets):
        logsoftmax = torch.nn.LogSoftmax()
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
    
    def forward(self, y_value, value, y_policy, policy):
        value = value.view(-1,1)
        y_value = y_value.view(-1,1)
        a = self.MSE(value.float(), y_value.float())
        b = self.cross_entropy(policy, y_policy)
        return (a + b)

from save_load import *

class Training():
    def __init__(self, DNN = ConnectNet()):
        #                           Min         Default     Max
        #self.num_epochs = 5
        self.total_epochs = 10      #5          10          15
        self.num_classes = 10
        self.batch_size = 5         #32         64          96
        self.learning_rate = 0.001  #0.001      0.005       0.01
        self.total_loss_epoch = []
        self.show = 0
        self.cache_directory = "DNN/"
        self.save_rate = 10
        self.version = 0
        self.initialize(DNN)

    def initialize(self, DNN):
        self.DNN = DNN
        self.optimizer = optim.Adam(self.DNN.parameters(), lr=self.learning_rate, betas=(0.8, 0.999))
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50,100,150,200,250,300,400], gamma=0.77)
        self.criterion = AlphaLoss()

    def backprop(self):
        #Backprop and perform Adam optimization
        self.optimizer.step()#updates the value of x using the gradient x.grad
        self.optimizer.zero_grad()#clears x.grad for every parameter x in the optimizer

    def forward_pass(self, data):
        #Get data ready
        #self.DNN.convert_state(data.S)
        S, policy, value = data
        #Run forward pass
        #policy_pred, value_pred = self.DNN.run()
        policy_pred, value_pred = self.DNN(S)
        if self.show:
            data.display()
            print("V:    ", value)
            print("V_y:  ", value_pred)
            print("P:    ", policy)
            print("P_y:  ", policy_pred)
        self.loss = self.criterion(value_pred[:,0], value, policy_pred, policy) / self.total_step
        self.loss.backward()#computes dloss/dx for every parameter x 

    def keep_track_of_numbers(self, i, epoch):
        self.total_loss += self.loss.item()#Loss is the sum of differencies for v & v_y
        self.loss_list.append(self.loss.item())
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            .format(epoch + 1, self.total_epochs, i + 1, self.total_step, self.loss.item()))

    def train(self, dataset):
        print("\n\nTRAINING DNN ", self.version)
        for epoch in range(self.total_epochs):
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
            self.total_step = len(trainloader)
            self.total_loss = 0.0
            self.loss_list = []
            for i, data in enumerate(trainloader, 0):#should be a fraction of data set of size batch_size
                self.forward_pass(data)
                self.keep_track_of_numbers(i, epoch)
            self.backprop()
            self.scheduler.step()#it change the learning rate
            self.total_loss_epoch.append(self.total_loss)
            print("\tEpoch ", epoch, "\tTotal loss:    ", self.total_loss)
        self.version += 1
        self.graph_data()
        if self.version % self.save_rate == 0:
            self.save()

    def save(self):
        file_name = self.get_name()
        save_cache(self, self.cache_directory + file_name)

    def load(self):
        file_to_load = get_list_of_files(self.cache_directory)
        if file_to_load != None:
            print("Loading...")
            cache = load_cache(self.cache_directory + file_to_load)
            if cache != None:
                self.version = cache.version
                self.total_epochs = cache.total_epochs
                self.batch_size = cache.batch_size
                self.learning_rate = cache.learning_rate
                self.total_loss_epoch = cache.total_loss_epoch
                self.initialize(cache.DNN)
                print("Success!")
            else:
                print("Failure!")
        else:
            print("Loading canceled")

    def get_name(self, on=True):
        file_name = ""
        if on:
            file_name += "V" + str(self.version) 
        file_name += "_E" + str(self.total_epochs) 
        file_name += "_B" + str(self.batch_size)
        file_name += "_LR" + str(self.learning_rate)
        return file_name

    def graph_data(self):
        self.path = "./graphs/"
        plt.plot(self.total_loss_epoch)
        plt.savefig(self.path + "All_epochs_" + self.get_name(False) + ".png")
        plt.clf()
        plt.plot(self.total_loss_epoch[0::self.total_epochs])
        plt.plot(self.total_loss_epoch[self.total_epochs - 1::self.total_epochs])
        plt.savefig(self.path + "First_and_Last_epochs_" + self.get_name(False) + ".png")
        plt.clf()
        for one in range(self.total_epochs):
            plt.plot(self.total_loss_epoch[one::self.total_epochs])
        plt.savefig(self.path + "All_by_epochs_" + self.get_name(False) + ".png")
        plt.clf()
















