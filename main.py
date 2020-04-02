# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ezalos <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/25 11:40:52 by ezalos            #+#    #+#              #
#    Updated: 2020/04/02 12:53:38 by ezalos           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#!/usr/bin/env python

from datetime import datetime
from time import sleep
from state import state
from MCTS import MCTS
import random
import copy

#from MCTS_c4 import run_MCTS
#from train_c4 import train_connectnet
#from evaluator_c4 import evaluate_nets
#from argparse import ArgumentParser
import logging
import pickle

import sys
from time import sleep
from listy import ft_progress
import time


cache = "cache_MCTS_Tree"

def save_state(s_object, file_name = cache):
    print("Save cache in ", file_name)
    with open(file_name, 'wb') as my_cache:
        pickle.dump(s_object, my_cache)
        print("Successful, tree size = ", s_object.size)

def load_state(file_name = cache):
    print("Load cache from ", file_name)
    with open(file_name, 'rb') as my_cache:
        my_obj = pickle.load(my_cache)
    print("Successful, tree size = ", my_obj.size)
    return my_obj

def one_turn(my_board):
    #print("\033[0;0H")
    actions = my_board.actions()
    move = random.randint(0, len(actions) - 1)
    play = actions[move]
    my_board.drop_piece(play)
    my_board.display()
    sleep(0.01)

def time_one_game(self, jo):
    start = time.time()    
    jo.self_play_one_game()
    print(time.time() - start)
    start = time.time()    
    jo.self_play_one_game()
    print(time.time() - start)
    start = time.time()    
    jo.self_play_one_game()
    print(time.time() - start)
    start = time.time()    
    jo.self_play_one_game()
    print(time.time() - start)
    start = time.time()    
    jo.self_play_one_game()
    print(time.time() - start)
    start = time.time()    
    jo.self_play_one_game()
    print(time.time() - start)
    start = time.time()    
    jo.self_play_one_game()
    print(time.time() - start)

if __name__ == "__main__":
    nb = 0
    while nb:
        nb -= 1
        my_board = state()
        turn = 0
        while my_board.check_winner() is not True and turn < 42:
            #input()
            print("\n\n\nTurn :\t", turn)
            one_turn(my_board)
            turn = turn + 1
        print("Simulations left : ", nb, "    ")
    try:
        jo = load_state()
    except:
        jo = MCTS()
    iterations = 2
    print("How much times ", iterations, " should be run ?")
    how = input()
    try:
        how = int(how)
    except:
        how = 0
    while how:
        for i in ft_progress(range(iterations)):
            jo.self_play_one_game()
        jo.display()
        save_state(jo)
        how -= 1
    jo.play_vs_MCTS()
