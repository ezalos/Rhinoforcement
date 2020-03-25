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
        self.actions = self.state.actions()
        if (self.state == None):
            self.state = state()

    def create_child(self, action):
        childState = self.state.create_child_state(action) 
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


class MCTS():
    def __init__(self, tree = tree()):
        self.tree = tree
        self.current_node = self.tree.root

    def policy(self):
        pass

    def select(self):
        # update .daddy in selected child
        pass

    def expand(self, action):
        self.tree.expand(self.current_node, action)

    def one_game(self, node):
        board = copy.deepcopy(node.state)
        while board.victory is '':
            actions = board.actions()
            move = random.randint(0, len(actions) - 1)
            play = actions[move]
            board.drop_piece(play)
        if self.victory == "." :
            vic = 0
        elif self.victory == "X" :
            vic = 1
        elif (-1 if self.victory == "O")
            vic = -1   
        return vic  

    def simulate(self):
        pass

    def backpropagate(self, node, cacahuetas):
        while node.daddy is not None:
            node.totalReward += cacahuetas
            node.visits += 1
            node = node.daddy
            

    def explore(self):
        pass

    def exploit(self):
        pass
    


                


