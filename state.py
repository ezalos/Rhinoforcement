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
        self.last_move = [-1,-1]
        self.turn = 0
        self.victory = ''


    def is_game_over(self):
        '''
            returns 1, 0
            assumes self.victory has been updated (done everytime we drop_piece)
        '''
        if self.victory == ".":
            return (0.0000000001)
        elif self.victory == "X":
            return (1.0)
        elif self.victory == "O":
            return (1.0)
        else:
            return (0.0)

    def do_action(self, column):
        '''
            changes player, turn and victory
        '''
        if self.victory != '' :
            print("Game Over")
        elif self.board[0, column] != " ":
            print("Invalid move")
            print(column)
        else:
            row = MAX_ROWS - 1
            while " " != self.board[row, column]:
                row -= 1
        self.board[row, column] = self.player
        self.last_move = [row, column]
        self.turn += 1
        self.check_winner()
        self.player = "X" if self.player == "O" else "O"

    def undrop_piece(self):
        if self.last_move[0] != -1:
            self.board[self.last_move[0]][self.last_move[1]] = " "
            self.turn -= 1
            self.player = "X" if self.player == "O" else "O"
        else:
            print("No memory of last move")

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
        if self.last_move[0] == -1:
            for row in MAX_ROWS:
                for col in MAX_COLS:
                    self.last_move = [row, col]
                    if self.check_line(1, 0):
                        return True
                    elif self.check_line(0, 1):
                        return True
                    elif self.check_line(1, 1):
                        return True
                    elif self.check_line(-1, 1):
                        return True
            self.last_move = [-1, -1]
        else:
            if self.check_line(1, 0):
                return True
            elif self.check_line(0, 1):
                return True
            elif self.check_line(1, 1):
                return True
            elif self.check_line(-1, 1):
                return True
        if self.turn >= 42:
            self.victory = "."
        return False

    def get_reward(self):
        '''
            returns 1, 0
            assumes self.victory has been updated (done everytime we drop_piece)
        '''
        if self.victory == ".":
            return (0)
        elif self.victory == "X":
            return (1)
        elif self.victory == "O":
            return (1)
        else:
            return None

    def actions(self):
        '''
            returns array of possible actions
        '''
        acts = []
        for col in range(MAX_COLS):
            if self.board[0, col] == " ":
                acts.append(col)
        return acts

    def valid_moves_mask(self):
        valid = np.zeros([MAX_COLS])
        for col in range(MAX_COLS):
            if self.board[0, col] == " ":
                valid[col] = 1
        return (valid)

    def reset(self):
        self.player = "X"
        self.last_move = [-1,-1]
        self.turn = 0
        self.victory = ''
        for row in range(MAX_ROWS): ## replace by init board ?
            for col in range(MAX_COLS):
                self.board[row][col] = " "

    def copy(self, other):
        '''
            copies all attributes of other into self
        '''
        self.player = other.player
        self.last_move[0] = other.last_move[0]
        self.last_move[1] = other.last_move[1]
        self.turn = other.turn
        self.victory = other.victory
        for row in range(MAX_ROWS):
            for col in range(MAX_COLS):
                self.board[row][col] = other.board[row][col]

    def encode_board(self):
        encoded = np.zeros([3, MAX_ROWS, MAX_COLS]).astype(float)
        player_conv = {"O":0, "X":1}
        for row in range(MAX_ROWS):
            for col in range(MAX_COLS):
                pos = self.board[row, col]
                encoded[2, row, col] = player_conv[self.player]
                if pos != " ":
                    encoded[player_conv[pos], row, col] = 1.0
        return encoded
    
    def decode_board(self, encoded):
        self.reset()
        player_conv = {0:"O", 1:"X"}
        for row in range(MAX_ROWS):
            for col in range(MAX_COLS):
                for player in range(2):
                    pos = encoded[row, col, player]
                    if pos == 1:
                        self.board[row, col] = player_conv[player]
                        self.turn += 1
        self.player = player_conv[encoded[0,0,2]]
        self.check_winner()

    def display(self):
        board = self.board
        move = self.last_move
        print("Turn", YELLOW, self.turn - 1, RESET, "for ", end="")
        if self.player == "X":
            print(BLUE + 'O' + RESET, end="")
        else:
            print(RED + 'X' + RESET, end="")
        print("")
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
                    print('.' + RESET, end="")
                print(' ', end="")
            print('\n', end="")
        print("0 1 2 3 4 5 6")
        if (self.victory != ''):
            print("Victory: ", self.victory)
        print('\n', end="")

    def stringify(self):
        return (str(self.last_move) + np.array_repr(self.board) + self.player)