import numpy as np
import math
from state import state
from node import node
import copy
import random

# isterminal to be added
PURPLE = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

class tree():
    
    def __init__(self):
        self.root = node()
        #self.hash = [{}] * 43
        #self.hash[0][self.root.state.stringify()] = self.root

    def print_n_floor(self, node=None, limit=1, deepness=0):
        max = len(str(self.root.visits))
        if (node == None):
            node = self.root
        best_UCB1 = -100000000000
        for action in node.actions :
            danger = node.children.get(action)
            if danger != None:
                new_UCB1 = danger.UCB1()
                if (new_UCB1 > best_UCB1):
                    best_UCB1 = new_UCB1
                    best_action = action
        for act in node.state.actions():
#            print("act", act)
            child = node.children[act]
            if deepness < 2 or act == node.state.actions()[0]:
                print("    " * deepness, end="")
            if act == best_action:
                print(UNDERLINE, end="")
            if deepness % 2 == 1:
                print(PURPLE, end="")
            else:
                print(RED, end="")
            print(act, "-->", end="")
            if (child != None):
                child.display(max)
                if deepness < 2:
                    print("")
                elif act != node.state.actions()[-1]:
                    print(" | ", end="")
                if deepness < limit:
                    self.print_n_floor(child, limit, deepness + 1)
            else:
                print("  NONE", RESET)
        if deepness >= 2:
            print("")

    def display(self):
        self.print_n_floor()

    def get_node_from_state(self, state):
        return self.hash[state.board.turn].get(state.stringify())
    
    def add_child_to_hash_and_parent(self, child_state, action, parent):
        existing_child = self.get_node_from_state(child_state)
        if (existing_child == None): # .key ?
            child = node(child_state, parent)
            self.hash[child_state.board.turn][child_state.stringify()] = child
            parent.children[action] = child
            self.size += 1
        else:
            parent.children[action] = existing_child

    def expand_hash(self, node, action):
        child_state = node.state.create_child_state(action)
        self.add_child_to_hash_and_parent(child_state, action, node)
