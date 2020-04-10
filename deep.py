#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib

import numpy as np


LAYERS = 64
RESBLOCKS = 7
K_SIZE = 3
RESIDUAL_TOWER_SIZE = 7

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
        v = v.view(-1, 6 * 7)

        v = F.relu(v)
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

    def train(self, data):
        print("\n\nTRAINING DNN")
        data.display()
        self.convert_state(data.S)
        value = torch.from_numpy(data.V).float()
        policy = torch.from_numpy(data.P).float()

        policy_pred, value_pred = self.run()
        
        print("V:    ", value)
        print("V_y:  ", value_pred)
        print("P:    ", policy)
        print("P_y:  ", policy_pred)

        criterion = AlphaLoss()
        loss = criterion(value_pred[:,0], value, policy_pred, policy)
        #loss = loss/args.gradient_acc_steps
        loss.backward()
        print("loss: ", loss)
        #if (epoch % args.gradient_acc_steps) == 0:
        #    optimizer.step()
        #    optimizer.zero_grad()
        #total_loss += loss.item()




















    
