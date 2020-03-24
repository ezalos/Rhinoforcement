#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd
import numpy as np

PURPLE = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def term_visu(board, move):
    for cols in range(len(board)):
        for rows in range(len(board[0])):
            spot = board[cols][rows]
            if cols == move[0] and rows == move[1]:
                print(UNDERLINE, end="")
            if spot == 'X':
                print(RED + 'X' + RESET, end="")
            elif spot == 'O':
                print(PURPLE + 'O' + RESET, end="")
            else:
                print('.', end="")
            print(' ', end="")
        print('\n', end="")
    print('\n', end="")

def view_board(np_data, fmt='{:s}', bkg_colors=['pink', 'pink']):
    data = pd.DataFrame(np_data, columns=['0','1','2','3','4','5','6'])
    fig, ax = plt.subplots(figsize=[7,7])
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])
    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    for (i,j), val in np.ndenumerate(data):
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = bkg_colors[idx]

        tb.add_cell(i, j, width, height, text=fmt.format(val), 
                    loc='center', facecolor=color)

    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text=label, loc='right', 
                    edgecolor='none', facecolor='none')

    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    tb.set_fontsize(24)
    ax.add_table(tb)
    return fig

