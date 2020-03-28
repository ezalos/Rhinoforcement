# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    game.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ezalos <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/25 11:40:41 by ezalos            #+#    #+#              #
#    Updated: 2020/03/28 14:35:10 by ezalos           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#!/usr/bin/env python

import numpy as np
import copy

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
        self.player = "X" if self.player == "O" else "O"
        self.check_winner()

    def check_line(self, y, x):
        player = self.player
        row = self.last_move[0]
        col = self.last_move[1]
        count = 0
        
        for i in range(0, 4):
            if 0 <= ((i * x) + row) and ((i * x) + row) < MAX_ROWS:
                if 0 <= ((i * y) + col) and ((i * y) + col) < MAX_COLS:
                    if player == self.board[row + (x * i), col + (y * i)]:
                        count += 1
                    else:
                        break
        for i in range(-1, -4, -1):
            if 0 <= (row + (i * x)) and ((i * x) + row) < MAX_ROWS:
                if 0 <= ((i * y) + col) and ((i * y) + col) < MAX_COLS:
                    if player == self.board[row + (x * i), col + (y * i)]:
                        count += 1
                    else:
                        break
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

