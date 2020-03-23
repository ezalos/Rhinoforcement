from datetime import datetime
from time import sleep
from game import board
from board_visu import view_board, term_visu
import random

#from MCTS_c4 import run_MCTS
#from train_c4 import train_connectnet
#from evaluator_c4 import evaluate_nets
#from argparse import ArgumentParser
import logging



def one_turn(my_board):
    print("\033[0;0H")
    actions = my_board.actions()
    move = random.randint(0, len(actions) - 1)
    play = actions[move]
    my_board.drop_piece(play)
    term_visu(my_board.current_board)
    sleep(0.01)

if __name__ == "__main__":
    nb = 100
    while nb:
        nb -= 1
        my_board = board()
        turn = 0
        while my_board.check_winner() is not True:
            print("\n\n\nTurn :\t", turn)
            one_turn(my_board)
            turn = turn + 1
        print("Simulations left : ", nb, "    ")
