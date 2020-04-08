# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ezalos <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/25 11:40:52 by ezalos            #+#    #+#              #
#    Updated: 2020/04/08 11:59:46 by ezalos           ###   ########.fr        #
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
from deep import Training
import time


cache = "cache_MCTS_tree"

def save_state(s_object, file_name = cache):
    print("Save cache in ", file_name)
    with open(file_name, 'wb') as my_cache:
        pickle.dump(s_object, my_cache)
        if (type(s_object) == type(MCTS())):
            print("Successful, tree size = ", s_object.size)

def dirty_save(s_object, file_name):
    pickle.dump(s_object, open(file_name, 'wb'))
    print("Save of cache successful, tree size = ", s_object.size)

def load_state(file_name = cache):
    print("Load cache from ", file_name)
    with open(file_name, 'rb') as my_cache:
        my_obj = pickle.load(my_cache)
    if (type(my_obj) == type(node())):
        print("Successful")
    return my_obj


if __name__ == "__main__":
    try:
        jo = MCTS()
        jo.dnn = load_state().DNN
        print("DNN version: ", jo.dnn.version)
    except:
        jo = MCTS()
        print("No dnn")
#        print("New MCTS")
#        jo = MCTS()
#    iterations = 1
#    try:
#        dataset = load_state("cache_dataset")
#        for data in dataset.data:
#            jo.dnn.train(data)
#        dataset.data = []
#    except:
#        print("New dataset")
#        dataset = dataset()
#    print("How much times ", iterations, " should be run ?")
#    how = input()
#    try:
#        how = int(how)
#    except:
#        how = 0
#    k = 0
#    while k < how:
#        save_dnn = jo.dnn
#        jo = MCTS(node(), dataset)
#        jo.dnn = save_dnn
#        for i in ft_progress(range(iterations)):
#            jo.self_play_new_game()
#        jo.display()
#        save_state(dataset, "cache_dataset")
#        for data in dataset.data:
#            jo.dnn.train(data)
#        dataset.data = []
#        save_state(jo, cache)
#
#        k += 1
#    jo.play_vs_MCTS()
    #tree = load_state()
#    jo = MCTS()
    train = Training(jo.dnn)
    for _ in range(10):
        for _ in range(5):
            start = time.time()
            jo.self_play_new_game()
            print(time.time() - start)
        train.initialize(jo.dnn)
        train.train(jo.dataset)
        tmp = jo.dnn
        jo.current_node = node()
        jo.root = jo.current_node
        jo.tree_root = jo.root
        jo.size = 1
        jo.dataset = dataset()
        jo.dnn = tmp
        save_state(train, cache)
    #save_state(jo.tree_root)
