import numpy as np
import math
from state import state
from node import node
import copy
import random
from board_visu import print_state


# isterminal to be added


class tree():
    
    def __init__(self):
        self.root = node()
        self.size = 1
        self.hash = [{}] * 43
        self.hash[0][self.root.state.stringify()] = self.root

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

    def add_child_to_parent(self, child_state, action, parent):
        child = node(child_state, parent)
        parent.add_child(action, child)
        self.size += 1
        return child

    def expand(self, node, action):
        child = node.create_child(action)
        self.add_child_to_parent(child, action, node)

    def expand_hash(self, node, action):
        child_state = node.state.create_child_state(action)
        self.add_child_to_hash_and_parent(child_state, action, node)

    def print_first_floor(self, node = None):
        if (node == None):
            node = self.root
        for a in node.state.actions():
            print(a)
            child = node.children.get(a)
            print(child)
            if (child != None):
                print("visits:", node.children.get(a).visits)
                print("wins:", node.children.get(a).total_reward)
            print(" ")

    def print_n_floor(self, node=None, limit=1, deepness=0):
        PURPLE = '\033[95m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        max = len(str(self.root.visits))
        if (node == None):
            node = self.root
        best_UCB1 = -100000000000
        for action in node.actions :
            new_UCB1 = node.children.get(action).UCB1()
            if (new_UCB1 > best_UCB1):
                best_UCB1 = new_UCB1
                best_action = action
        for act in node.state.actions():
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
                print(" " * (max - len(str(child.total_reward))), child.total_reward, "/", child.visits, " " * (max - len(str(child.visits))), end="")
                print("=", str(child.UCB1())[:4], RESET, end="")
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

