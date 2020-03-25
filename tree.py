import numpy as np
from game import state
import copy

# isterminal to be added
class node():
    def __init__(self, state = state(), parent = None):
        self.state = state
        self.daddy = parent
        self.visits = 0
        self.totalReward = 0
        self.children = {}
        #self.actions = []
        if (self.state == None):
            self.state = state()

    def create_child(self, action):
        childState = copy.deepcopy(self.state)
        childState.drop_piece(action)
        self.children[action] = node(childState, self)

    def add_child(self, action, child):
        self.children[action] = child


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

    def expand(self, node):
        for action in (node.state.actions()):
            child_state = node.state.create_child_state(action)
            self.add_child_to_hash_and_parent(child_state, action, node)


class MCTS():
    def __init__(self, tree = tree()):
        self.tree = tree
        self.current_node = self.tree.root

    def policy(self):
        pass

    def select(self):
        # update .daddy in selected child
        pass

    def expand(self):
        pass
        self.tree.expand()

    def simulate(self):
        pass

    def backpropagate(self):
        pass

    def explore(self):
        pass

    def exploit(self):
        pass
    


                


