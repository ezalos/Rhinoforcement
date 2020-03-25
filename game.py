# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    game.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ezalos <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/25 11:40:41 by ezalos            #+#    #+#              #
#    Updated: 2020/03/25 11:41:25 by ezalos           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#!/usr/bin/env python

import numpy as np
import copy

class state():
    def __init__(self):
        self.init_board = np.zeros([6,7]).astype(str)
        self.init_board[self.init_board == "0.0"] = " "
        self.player = 0
        self.board = self.init_board
        self.last_move = [0,0]
        self.turn = 0

    def drop_piece(self, column):
        if self.board[0, column] != " ":
            return "Invalid move"
        else:
            self.turn += 1
            row = 0; pos = " "
            while (pos == " "):
                if row == 6:
                    row += 1
                    break
                pos = self.board[row, column]
                row += 1
            if self.player == 0:
                self.board[row-2, column] = "O"
                self.last_move = [row-2, column]
                self.player = 1
            elif self.player == 1:
                self.board[row-2, column] = "X"
                self.last_move = [row-2, column]
                self.player = 0
    
    def check_line(self, y, x):
        player = ("O" if self.player == 1 else "X")
        row = self.last_move[0]
        col = self.last_move[1]
        #print("Where : ", self.last_move)
        #print(y, x)
        count = 0
        
        for i in range(0, 4):
            #print (i, end="")
            if 0 <= ((i * x) + row) and ((i * x) + row) < 6:
                if 0 <= ((i * y) + col) and ((i * y) + col) < 7:
                    if player == self.board[row + (x * i), col + (y * i)]:
                        count += 1
                    else:
                        break
        for i in range(-1, -4, -1):
            #print (i, end="")
            if 0 <= (row + (i * x)) and ((i * x) + row) < 6:
                if 0 <= ((i * y) + col) and ((i * y) + col) < 7:
                    if player == self.board[row + (x * i), col + (y * i)]:
                        count += 1
                    else:
                        break
        #print("Count: ", count)
        if count >= 4:
            return True
        return False

    def check_winner(self):
        if self.check_line(1, 0):
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
        for col in range(7):
            if self.board[0, col] == " ":
                acts.append(col)
        return acts
    
    def create_child_state(self, action):
        child_state = copy.deepcopy(self)
        child_state.drop_piece(action)
        return (child_state)

