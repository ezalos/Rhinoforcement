# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ezalos <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/25 11:40:52 by ezalos            #+#    #+#              #
#    Updated: 2020/03/25 11:41:13 by ezalos           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#!/usr/bin/env python

from datetime import datetime
from time import sleep
from game import state
from board_visu import term_visu
from board_visu import print_state
from tree import MCTS
import random

#from MCTS_c4 import run_MCTS
#from train_c4 import train_connectnet
#from evaluator_c4 import evaluate_nets
#from argparse import ArgumentParser
import logging



def one_turn(my_board):
    #print("\033[0;0H")
    actions = my_board.actions()
    move = random.randint(0, len(actions) - 1)
    play = actions[move]
    my_board.drop_piece(play)
    term_visu(my_board.board, my_board.last_move)
    sleep(0.01)

if __name__ == "__main__":
    nb = 1
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
        #sleep(3)

    jo = MCTS()
    for i in range (5000):
        jo.play()