#!/usr/bin/env python

from datetime import datetime
from time import sleep
from state import state
from MCTS import MCTS
from node import node
import random
import copy
import logging
from data import Dataseto
import sys
from time import sleep
from listy import ft_progress
from deep import Training
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    jo = MCTS()
    train = Training(jo.dnn)
    train.load()
    jo.dnn = train.DNN
    while jo.size < 0:
        jo.self_play_new_game()
    for _ in range(50):
        print("Size: ", jo.size)
        start = time.time()
        for i in range(5):
            print(i + 1, "/", 5)
            jo.self_play_new_game()
        print("Game time:  ", time.time() - start)
        start = time.time()
        train.initialize(jo.dnn)
        train.train(jo.dataset)
        print("Train time: ", time.time() - start)
        jo.dataset = Dataseto()
