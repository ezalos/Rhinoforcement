# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ezalos <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/25 11:40:52 by ezalos            #+#    #+#              #
#    Updated: 2020/03/31 17:52:17 by ezalos           ###   ########.fr        #
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

from data import dataset
import sys
from time import sleep
import time

start_time = time.time()

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def ft_progress(listy):
	total = len(listy)
	unit = int(total / 20)
	for i in listy:
#       if (i == 0):
#            start_time = time.time()
		elapsed_time = ((time.time()) - (start_time))
		eta = ((elapsed_time) * (total - i) / (i + 1))
		print("ETA: ", ' ' if (eta // 10) < 1 else '', str(truncate(eta, 2)) + "s ",
		"[ " if i < (total // 10) else "[", str((((i + 1) * 100)// total)) + '%' + ']' +
		'[' + (i // unit) * '=' + '>' + ((total - i - 1) // unit) * ' ' + '] ' +
		str(i + 1) + '/' + str(total) +
		" | elapsed time " + str(truncate(elapsed_time, 2)) + "s  ",
		end="\r")
		yield i
	print("")

cache = "cache_MCTS_tree"

def save_state(s_object, file_name = cache):
    with open(file_name, 'wb') as my_cache:
        pickle.dump(s_object, my_cache)
        print("Save of cache successful, tree size = ", s_object.size)

def dirty_save(s_object, file_name):
    pickle.dump(s_object, open(file_name, 'wb'))
    print("Save of cache successful, tree size = ", s_object.size)

def load_state(file_name = cache):
    with open(file_name, 'rb') as my_cache:
        my_obj = pickle.load(my_cache)
    print("Load of cache successful, tree size = ", my_obj.size)
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
    try:
        jo = load_state()
    except:
        jo = MCTS()
    dataset = dataset()
    iterations = 25
    print("How much times ", iterations, " should be run ?")
    how = input()
    try:
        how = int(how)
    except:
        how = 0
    k = 0
    while k < how:
        for i in ft_progress(range(iterations)):
            jo.self_play_one_game(dataset)
        jo.display()
        save_state(jo, cache)
        k += 1
#    jo.play_vs_MCTS()
