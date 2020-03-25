import numpy as np
import copy
import board

class tree():
    def __init__(self, root, policy):
        self.root = root
        self.policy = policy

# isterminal to be added
class node():
    def __init__(self, state = None, parent = None):
        self.state = state
        self.daddy = parent
        self.visits = 0
        self.totalReward = 0
        self.children = {}
        self.actions = []
        if (self.state == None):
            self.state = board()

    def create_child(self, action):
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
        root = board()
        self.tree[root.state] = root
    


    def expand(self, node):
        for a in (node.possible_actions()):
            existing_child = tree.


