#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib

import numpy as np
from state import state

LAYERS = 64
RESBLOCKS = 7
K_SIZE = 3
RESIDUAL_TOWER_SIZE = 3

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(LAYERS, LAYERS, K_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(LAYERS)
        self.conv2 = nn.Conv2d(LAYERS, LAYERS, K_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(LAYERS)
    
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = F.relu(x)
        return (x)

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(3, LAYERS, K_SIZE, padding=1)
        self.bn = nn.BatchNorm2d(LAYERS)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  
        x = F.relu(x)
        return (x)

class HeadBlock(nn.Module):
    def __init__(self):
        super(HeadBlock, self).__init__()
        self.conv1 = nn.Conv2d(LAYERS, 1, 1, padding=0) #value head
        self.bn1 = nn.BatchNorm2d(1)
        self.lin1 = nn.Linear(6 * 7, 7)
        #self.lin2 = nn.Linear(42, 20)
        self.lin3 = nn.Linear(7, 1)
        self.conv2 = nn.Conv2d(LAYERS, 2, 1, padding=0)
        self.bn2 = nn.BatchNorm2d(2)
        self.lin4 = nn.Linear(2 * 7 * 6, 7)

    def forward(self, x):
        v = self.conv1(x)
        v = self.bn1(v)
        v = F.relu(v)
        v = v.view(-1, 6 * 7)

        v = self.lin1(v)
        v = F.relu(v)
        v = self.lin3(v)

        v = torch.tanh(v)

        p = self.conv2(x)
        p = self.bn2(p)
        p = F.relu(p)
        p = p.view(-1, 2 * 6 * 7)
        p = self.lin4(p)

        return p, v

class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(RESIDUAL_TOWER_SIZE):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = HeadBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(RESIDUAL_TOWER_SIZE):
            s = getattr(self, "res_%i" % block)(s)
        p, v = self.outblock(s)
        return p, v

    def evaluate(self, unencoded_s):
        s = torch.from_numpy(unencoded_s.encode_board()).float()
        t = s.new_empty(1, 3, 6, 7)  ## create batch of size 1
        t[0] = s
        return (self.forward(t))

class NetHandler():
    def __init__(self, net, args):
        self.net = net
        self.args = args

    def cross_entropy(self, pred, soft_targets):
        logsoftmax = torch.nn.LogSoftmax()
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


    def loss(self, P, V, PGT, VGT):
        MSEloss = torch.nn.MSELoss()
        a = MSEloss(V.float(), VGT.float())
        b = MSEloss(P.float(), PGT.float()) * 7
        return (b + a)    

    def train(self, trainloader):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = self.net
        net.to(device)
        net.train()
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) ##wtf is this the right one ??

        for epoch in range(self.args.Epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                S, PGT, VGT = data[0].to(device), data[1].to(device), data[2].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                P, V = net(S)
                losses = self.loss(P, V, PGT, VGT)
                losses.backward()
                optimizer.step()
                # print statistics
                running_loss += losses.item()
            print('[%d, %5d] 100 * loss: %.3f' % (epoch + 1, i + 1, running_loss / len(trainloader)))
        print('Finished Training')

#
#class ConvBlock(nn.Module):
#    def __init__(self):
#        super(ConvBlock, self).__init__()
#        self.action_size = 7
#        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
#        self.bn1 = nn.BatchNorm2d(128)
#
#    def forward(self, s):
#        s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
#        s = F.relu(self.bn1(self.conv1(s)))
#        return s
#
#class ResBlock(nn.Module):
#    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
#        super(ResBlock, self).__init__()
#        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
#                     padding=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                     padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(planes)
#
#    def forward(self, x):
#        residual = x
#        out = self.conv1(x)
#        out = F.relu(self.bn1(out))
#        out = self.conv2(out)
#        out = self.bn2(out)
#        out += residual
#        out = F.relu(out)
#        return out
#    
#class OutBlock(nn.Module):
#    def __init__(self):
#        super(OutBlock, self).__init__()
#        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
#        self.bn = nn.BatchNorm2d(3)
#        self.fc1 = nn.Linear(3*6*7, 32)
#        self.fc2 = nn.Linear(32, 1)
#        
#        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
#        self.bn1 = nn.BatchNorm2d(32)
#        self.logsoftmax = nn.LogSoftmax(dim=1)
#        self.fc = nn.Linear(6*7*32, 7)
#    
#    def forward(self,s):
#        v = F.relu(self.bn(self.conv(s))) # value head
#        v = v.view(-1, 3*6*7)  # batch_size X channel X height X width
#        v = F.relu(self.fc1(v))
#        v = torch.tanh(self.fc2(v))
#        
#        p = F.relu(self.bn1(self.conv1(s))) # policy head
#        p = p.view(-1, 6*7*32)
#        p = self.fc(p)
#        p = self.logsoftmax(p).exp()
#        return p, v
#    













    
