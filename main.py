# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ezalos <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/25 11:40:52 by ezalos            #+#    #+#              #
#    Updated: 2020/04/02 16:09:26 by ezalos           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#!/usr/bin/env python

from datetime import datetime
from time import sleep
from state import state
from MCTS import MCTS
from node import node
import random
import copy

#from MCTS_c4 import run_MCTS
#from train_c4 import train_connectnet
#from evaluator_c4 import evaluate_nets
#from argparse import ArgumentParser
import logging
import pickle

from data import dataset
import sys
from time import sleep
from listy import ft_progress
import time
from data import Dataseto
import torch.optim as optim
import torch 

cache = "cache_MCTS_tree"

def save_state(s_object, file_name = cache):
    print("Save cache in ", file_name)
    with open(file_name, 'wb') as my_cache:
        pickle.dump(s_object, my_cache)
        if (type(s_object) == type(MCTS())):
            print("Successful, tree size = ", s_object.size)


def load_state(file_name = cache):
    print("Load cache from ", file_name)
    with open(file_name, 'rb') as my_cache:
        my_obj = pickle.load(my_cache)
    if (type(my_obj) == type(node())):
        print("Successful")
    return my_obj


if __name__ == "__main__":
    from deep import ConnectNet #MAYBE TO FLOAT
    datasett = Dataseto()

    jo = MCTS()
    root = jo.tree_root
    for i in range(100):
        start = time.time()
        jo.self_play_new_game()
        if (i == 50):
            jo.dataset = datasett
        #jo.current_node.state.display()
        #jo.tree_root.print_n_floor(jo.tree_root, 0)
        print(time.time() - start)
    
    save_state(jo.tree_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ConnectNet()
    net.to(device)
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    MSEloss = torch.nn.MSELoss()
    CrossLoss = torch.nn.CrossEntropyLoss()
    print("LOADED")
    def cross_entropy(pred, soft_targets):
        logsoftmax = torch.nn.LogSoftmax()
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

    def loss(P, V, PGT, VGT):
        a = 2 * MSEloss(V.float(), VGT.float())
        b = cross_entropy(P, PGT)
        return (a + b)
    
    print("NETTED")
    for weshwesh in range(10):
        trainiloader = torch.utils.data.DataLoader(datasett, batch_size=8, shuffle=True, num_workers=1)
        for epoch in range(10):  # loop over the dataset multiple times

            running_loss = 0.0
            SUP = None
            for i, data in enumerate(trainiloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                S, PGT, VGT = data[0].to(device), data[1].to(device), data[2].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                P, V = net(S)
                losses = loss(P, V, PGT, VGT) / 8
                losses.backward()
                optimizer.step()
                # print statistics
                running_loss += losses.item()
            print('[%d, %5d] 100 * loss: %.3f' % (epoch + 1, i + 1, running_loss  *100 / i))
                

    print('Finished Training')
    torch.cuda.empty_cache()
        #save_state(jo.tree_root)