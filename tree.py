import numpy as np
import copy
import board

class tree():
    def __init__(self, root, policy):
        self.root = root
        self.policy = policy

# isterminal to be added
class node():
    def __init__(self, parent = None):
        self.state = board()
        self.daddy = parent
        self.visits = 0
        self.totalReward = 0
        self.children = {}
        self.actions = []

    def create_child(self, action)
        childState = copy.deepcopy(self.state)
        childState.drop_piece(action)
        self.children[action] = node(childState, self)

    def add_child(self, action, child):
        self.children[action] = child
    
    def possible_actions(self):
        out = []
        for i in range(7):
            if (self.state.current_board[0, i] == " "):
                out.append(i)
        return (out)




class MCTS():
    def __init__(self):
        self.yo = 1
        self.tree = {}
    


    def expand(self, node):
        for a in (node.possible_actions()):
            existing_child = tree.


