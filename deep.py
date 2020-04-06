#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")
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
        for block in range(5):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()

    def forward(self,s):
        s = self.conv(s)
        for block in range(5):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                  (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error

class Deep_Neural_Net():
    def __init__(self):
        self.temp = 1
        self.deep_neural_net = ConnectNet()
        self.policy = None
        self.value = None

    def convert_state(self, state):
        encoded_s = state.encode_board();
        encoded_s = encoded_s.transpose(2,0,1)
        encoded_s = torch.from_numpy(encoded_s).float()#.cuda()
        self.encoded_state = encoded_s

    def run(self):
        policy, value = self.deep_neural_net(self.encoded_state)
        self.policy = policy.detach().cpu().numpy().reshape(-1);
        self.value = value.item()
        return policy, value

class Training():
    def __init__(self, DNN):
        #self.num_epochs = 5
        self.total_epochs = 5
        self.num_classes = 10
        self.batch_size = 100
        self.learning_rate = 0.001
        self.total_loss_epoch = []
        self.initialize(DNN)

    def initialize(self, DNN):
        self.DNN = DNN
        self.optimizer = optim.Adam(self.DNN.deep_neural_net.parameters(), lr=learning_rate, betas=(0.8, 0.999))
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,400], gamma=0.77)
        self.criterion = AlphaLoss()

    def backprop(self):
        #Backprop and perform Adam optimization
        self.optimizer.zero_grad()#clears x.grad for every parameter x in the optimizer
        self.loss.backward()#computes dloss/dx for every parameter x 
        self.optimizer.step()#updates the value of x using the gradient x.grad

    def forward_pass(self, data):
        #Get data ready
        data.display()
        self.DDN.convert_state(data.S)
        value = torch.from_numpy(data.V).float()
        policy = torch.from_numpy(data.P).float()
        #Run forward pass
        policy_pred, value_pred = self.DNN.deep_neural_net()
        print("V:    ", value)
        print("V_y:  ", value_pred)
        print("P:    ", policy)
        print("P_y:  ", policy_pred)
        self.loss = self.criterion(value_pred[:,0], value, policy_pred, policy)

    def keep_track_of_numbers(self, i):
        self.total_loss += self.loss.item()#Loss is the sum of differencies for v & v_y
        self.loss_list.append(self.loss.item())
        #if (i + 1) % self.batch_size == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            .format(epoch + 1, self.total_epochs, i + 1, self.total_step, self.loss.item()))

    def train(self, dataset):
        print("\n\nTRAINING DNN")
        for epoch in range(self.total_epoch):
            self.total_loss = 0.0
            self.total_step = len(dataset.data)
            self.loss_list = []
            for data, i in enumerate(dataset.data):#should be a fraction of data set of size batch_size
                self.forward_pass(data)
                self.backprop()
                self.keep_track_of_numbers(i)
            scheduler.step()#it change the learning rate
            self.total_loss_epoch.append(self.total_loss)
            print(self.total_loss_epoch)


















