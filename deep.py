#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from ARGS import DotDict
from ARGS import ARGS
import numpy as np
from state import state


LAYERS = 64
RESBLOCKS = 7
K_SIZE = 3
RESIDUAL_TOWER_SIZE = 3

def cross_entropy_loss(input, target):
    loss = 0
    for i in range(7):
        loss = loss + (input[i] * torch.log(target[i]))
    return (-loss)

def cross_entropy_loss_batch(input, target):
    loss = 0
    for i in range(7):
        loss = loss + (input[i] * torch.log(target[i]))
    return (-loss)

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
        self.conv = nn.Conv2d(3, LAYERS, K_SIZE, padding=1)
        self.bn = nn.BatchNorm2d(LAYERS)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  
        x = F.relu(x)
        return (x)


class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(LAYERS, 1, kernel_size=1) #value head
        self.bn1 = nn.BatchNorm2d(1)
        self.lin1 = nn.Linear(6 * 7, 256)
        self.lin2 = nn.Linear(256, 1)

    def forward(self, x):
        v = self.conv1(x)
        v = self.bn1(v)
        v = F.relu(v)
        v = v.view(-1, 6 * 7)

        v = self.lin1(v)
        v = F.relu(v)
        v = self.lin2(v)

        v = torch.tanh(v)

        return v

class PolicyHead(nn.Module):
    def __init__(self):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(LAYERS, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.lin = nn.Linear(6 * 7, 7)

    def forward(self, x):

        p = self.conv(x)
        p = self.bn(p)
        p = F.relu(p)
        p = p.view(-1, 6 * 7)
        p = self.lin(p)
        p = self.logsoftmax(p).exp()

        return p

class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(RESIDUAL_TOWER_SIZE):
            setattr(self, "res_%i" % block,ResBlock())
        self.PolicyHead = PolicyHead()
        self.ValueHead = ValueHead()
        self.Value_loss = nn.MSELoss()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(RESIDUAL_TOWER_SIZE):
            s = getattr(self, "res_%i" % block)(s)
        p = self.PolicyHead(s)
        v = self.ValueHead(s)
        return p, v

    def VLoss(self, V, target):
        return(self.Value_loss(V, target))

    def PLoss(self, P, target):
        return (cross_entropy_loss(P, target))

    def evaluate(self, unencoded_s):
        s = torch.from_numpy(unencoded_s.encode_board()).float()
        t = s.new_empty(1, 3, 6, 7)  ## create batch of size 1
        t[0] = s
        return (self.forward(t))

    def evaluate_encoded(self, s):
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

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* (1e-8 + y_policy.float()).float().log()), 1)
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
        loss = self.criterion(value_pred[:,0], value, policy_pred, policy)

    def train(self, dataset):
        print("\n\nTRAINING DNN")
        #update_size = len(train_loader)//10
        for epoch in range(self.total_epoch):
            total_loss = 0.0
            #losses_per_batch = []
            total_step = len(dataset.data)
            loss_list = []
            acc_list = []
            for data, i in enumerate(dataset.data):#should be a fraction of data set of size batch_size
                self.forward_pass(data)
                self.backprop()
                #Track numbers
                total_loss += loss.item()#Loss is the sum of differencies for v & v_y

                #total = labels.size(0)
                #_, predicted = torch.max(outputs.data, 1)
                #correct = (predicted == labels).sum().item()
                #acc_list.append(correct / total)#Accuracy is the % of good answers
                if (i + 1) % self.batch_size == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                  (correct / total) * 100))

            scheduler.step()#it change the learning rate
        #should print total update here


    def evaluate(self, unencoded_s):
        s = torch.from_numpy(unencoded_s.encode_board()).float()
        t = s.new_empty(1, 3, 6, 7)  ## create batch of size 1
        t[0] = s
        return (self.forward(t))


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













