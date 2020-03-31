#!/usr/bin/env python

import numpy as np
import copy
from color import *

MAX_ROWS = 6
MAX_COLS = 7


class state():
    def __init__(self):
        self.init_board = np.zeros([MAX_ROWS, MAX_COLS]).astype(str)
        self.init_board[self.init_board == "0.0"] = " "
        self.player = "X"
        self.board = self.init_board
        self.last_move = [0,0]
        self.turn = 0
        self.victory = ''

    def drop_piece(self, column):
        if self.victory != '' :
            return "Game Over"
        elif self.board[0, column] != " ":
            return "Invalid move"
        else:
            row = MAX_ROWS - 1
            while " " != self.board[row, column]:
                row -= 1
        self.board[row, column] = self.player
        self.last_move = [row, column]
        self.turn += 1
        self.check_winner()
        self.player = "X" if self.player == "O" else "O"

    def check_line(self, y, x):
        player = self.player
        row = self.last_move[0]
        col = self.last_move[1]
        if 0:
            if y == 1 and x == 0:
                print("|")
            elif y == 0 and x == 1:
                print("_")
            elif y == 1 and x == 1:
                print("/")
            elif y == -1 and x == 1:
                print("\\")
        count = 0
        found = 0
        for i in range(0, 4):
            if 0 <= ((i * x) + row) and ((i * x) + row) < MAX_ROWS:
                if 0 <= ((i * y) + col) and ((i * y) + col) < MAX_COLS:
                    if player == self.board[row + (x * i), col + (y * i)]:
                        count += 1
                        found = 1
                    elif found:
                        break
        for i in range(-1, -4, -1):
            if 0 <= (row + (i * x)) and ((i * x) + row) < MAX_ROWS:
                if 0 <= ((i * y) + col) and ((i * y) + col) < MAX_COLS:
                    if player == self.board[row + (x * i), col + (y * i)]:
                        count += 1
                    elif found:
                        break
        if 0:
            print("Count : ", count)
        if count >= 4:
            self.victory = player
            return True
        return False

    def check_winner(self):
        if self.turn >= 42:
            self.victory = '.'
            return False
        elif self.check_line(1, 0):
            return True
        elif self.check_line(0, 1):
            return True
        elif self.check_line(1, 1):
            return True
        elif self.check_line(-1, 1):
            return True
        else:
            return False

    def stringify(self):
        return (self.board.tostring())
    
    def actions(self): # returns all possible moves
        acts = []
        for col in range(MAX_COLS):
            if self.board[0, col] == " ":
                acts.append(col)
        return acts
    
    def create_child_state(self, action):
        child_state = copy.deepcopy(self)
        child_state.drop_piece(action)
        return (child_state)

    def display(self):
        board = self.board
        move = self.last_move
        for rows in range(MAX_ROWS):
            for cols in range(MAX_COLS):
                spot = board[rows, cols]
                if   cols == move[1] and rows == move[0]:
                   print(UNDERLINE, end="")
                if spot == 'X':
                    print(RED + 'X' + RESET, end="")
                elif spot == 'O':
                    print(BLUE + 'O' + RESET, end="")
                else:
                    print('.', end="")
                print(' ', end="")
            print('\n', end="")
        print('\n', end="")



