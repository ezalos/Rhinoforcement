from MCTS import MCTS
from state import state
import copy
import numpy
from deep import ConnectNet
from data import Dataseto
from ARGS import DotDict
from ARGS import ARGS
from deep import NetHandler
import torch.optim as optim
import torch




class Trainer():
    def __init__(self, net = ConnectNet(), args = ARGS, dataset = Dataseto(), MC = 0):
        if (MC == 0):
            self.MCTS = MCTS(net, args)
        else:
            MC = MCTS
        self.dataset = dataset
        self.args = args
        self.net = net
        self.netHandler = NetHandler(net, args)

    def createSet(self):
        i = 0
        while len(self.dataset) < 1000:
            self.MCTS.self_play(dataset=self.dataset, root=state())
            i += 1
            print(i)
        print("LENNN: ", len(self.dataset))
        

    def train(self):
        net = self.net
        net.train()
        trainLoader = torch.utils.data.DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=1)
        self.netHandler.train(trainLoader)

    def arena_fightooo(self, nn1, nn2):
        pass

    def execute(self):
        self.createSet()
        self.train()


def train_net(net, dataset):
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    trainLoader = torch.utils.data.DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=2)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

t = Trainer()
t.execute()