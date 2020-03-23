#!/usr/bin/env python

import numpy as np

class board():
    def __init__(self):
        self.init_board = np.zeros([6,7]).astype(str)
        self.init_board[self.init_board == "0.0"] = " "
        self.player = 0
        self.current_board = self.init_board

    # ALSO CHANGES PLAYER
    def drop_piece(self, column):
        if self.current_board[0, column] != " ":
            return "Invalid move"
        else:
            row = 0
            while (row < 6):
                if (row == 5):
                    break
                if (self.current_board[row, column] != " "):
                    row = row - 1
                    break
                row = row + 1
                if self.player == 0:
                    self.current_board[row, column] = "O"
                    self.player = 1
                elif self.player == 1:
                    self.current_board[row, column] = "X"
                    self.player = 0
                    
    # returns True or false carefull uses self.player
    def check_winner(self, row, col):
        '''
        returns ( self.player is the winner )
        '''
        player = ("O" if self.player == 0 else "X")
        count = 0
        row2 = row
        col2 = col
        row = 0
        while (row < 6):
            if (self.current_board[row, col] != player):
                count = 0
            else:
                count += 1
            row += 1
        if (count >= 4):
            return (True)

        row = row2
        count = 0
        col = 0
        while (col < 7):
            if (self.current_board[row, col] != player):
                count = 0
            else:
                count += 1
            row += 1
        if (count >= 4):
            return (True)
        col = col2
        count = 0
        while (col > 0 and row > 0):
            col -= 1
            row -= 1
        while (row < 6 and col < 7):
            if (self.current_board[row, col] != player):
                count = 0
            else:
                count += 1
            row += 1
            col += 1
        if (count >= 4):
            return (True)
        count = 0
        col = col2
        row = row2
        while (row > 0 and col < 7):
            col += 1
            row -= 1
        while (row < 6 and col > 0):
            if (self.current_board[row, col] != player):
                count = 0
            else:
                count += 1
                row += 1
                col -= 1
        if (count >= 4):
            return (True)
        return (false)

    def check_winner(self):
        if self.player == 1:
            for row in range(6):
                for col in range(7):
                    if self.current_board[row, col] != " ":
                        # rows
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col] == "O" and \
                                self.current_board[row + 2, col] == "O" and self.current_board[row + 3, col] == "O":
                                #print("row")
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row, col + 1] == "O" and \
                                self.current_board[row, col + 2] == "O" and self.current_board[row, col + 3] == "O":
                                #print("col")
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col + 1] == "O" and \
                                self.current_board[row + 2, col + 2] == "O" and self.current_board[row + 3, col + 3] == "O":
                                #print("\\")
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col - 1] == "O" and \
                                self.current_board[row + 2, col - 2] == "O" and self.current_board[row + 3, col - 3] == "O"\
                                and (col-3) >= 0:
                                #print("/")
                                return True
                        except IndexError:
                            next
        if self.player == 0:
            for row in range(6):
                for col in range(7):
                    if self.current_board[row, col] != " ":
                        # rows
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col] == "X" and \
                                self.current_board[row + 2, col] == "X" and self.current_board[row + 3, col] == "X":
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row, col + 1] == "X" and \
                                self.current_board[row, col + 2] == "X" and self.current_board[row, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col + 1] == "X" and \
                                self.current_board[row + 2, col + 2] == "X" and self.current_board[row + 3, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col - 1] == "X" and \
                                self.current_board[row + 2, col - 2] == "X" and self.current_board[row + 3, col - 3] == "X"\
                                and (col-3) >= 0:
                                return True
                        except IndexError:
                            next
    def actions(self): # returns all possible moves
        acts = []
        for col in range(7):
            if self.current_board[0, col] == " ":
                acts.append(col)
        return acts
            