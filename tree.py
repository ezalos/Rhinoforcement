import numpy as np
import math
from game import state
import copy
import random
from board_visu import print_state


# isterminal to be added
class node():
    def __init__(self, state = state(), parent = None):
        self.state = state # WE CAN REMOVE THE STATE AS THE PATH HOLDS IT !!!
        self.daddy = parent
        self.visits = 0
        self.total_reward = 0
        self.children = {}
        self.actions = self.state.actions()
        self.is_fully_expanded = 0 #TO UPDATE
        if (self.state == None):
            self.state = state()

    def create_child(self, action):
        childState = self.state.create_child_state(action)
        self.children[action] = node(childState, self)
        if len(self.actions) == len(self.children):
            self.is_fully_expanded = 1

    def add_child(self, action, child):
        self.children[action] = child
        if len(self.actions) == len(self.children):
            self.is_fully_expanded = 1

    def UCB1(self):
        return (self.total_reward / self.visits) + math.sqrt(2) * math.sqrt(math.log(self.daddy.visits / self.visits))

    def winrate(self):
        return (self.total_reward / self.visits)
    
    def unexplored_actions(self):
        unexplored_moves = []
        for a in self.actions :
            if (self.children.get(a) == None): ## need to check it actually returns none when lacking an entry
                unexplored_moves.append(a)
        return (unexplored_moves)

    def random_unexplored_action(self):
        act = self.unexplored_actions()
        return (act[random.randint(0, len(act) - 1)])

    def play_move_keep_board(self, action):
        existing_child = self.children.get(action)
        self.state.drop_piece(action)
        if (existing_child == None):
            self.children[action] = node(self.state, self)
            if (len(self.children == len(self.actions))):
                self.is_fully_expanded = 1
        else:
            self.children.get(action).state = self.state

    def create_child_keep_board(self, action):
        self.state.drop_piece(action)
        self.children[action] = node(self.state, self)
        if (len(self.children) == len(self.actions)):
            self.is_fully_expanded = 1


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


class MCTS():
    def __init__(self, tree = tree()):
        self.tree = tree
        self.current_node = self.tree.root
        self.size = 0

    def policy(self):
        pass

    def select(self):
        '''
        returns the action leading to the state with the highest UCB score
        '''
        best_action = self.current_node.actions[0]
        best_UCB1 = self.current_node.children.get(best_action).UCB1()
        new_UCB1 = 0
        for action in self.current_node.actions :
            new_UCB1 = self.current_node.children.get(action).UCB1()
            if (new_UCB1 > best_UCB1):
                best_UCB1 = new_UCB1
                best_action = action
        return (best_action)

    def select_greedy(self):
        '''
        returns the action leading to the state with the highest UCB score
        '''
        best_action = self.current_node.actions[0]
        best_winrate = self.current_node.children.get(best_action).winrate()
        new_winrate = 0
        for action in self.current_node.actions :
            new_winrate = self.current_node.children.get(action).winrate()
            if (new_winrate > best_winrate):
                best_winrate = new_winrate
                best_action = action
        return (best_action)

    def selection(self):
        while (self.current_node.is_fully_expanded):
            action = self.select()
            self.play_action(action)

    def expand(self):
        '''
        picks a move among those never played, and plays the move. Creating the corresponding child node.
        '''
        if (self.current_node.is_fully_expanded):
            print("youre trying to expand a fully expanded node and this should never print")
        action = self.current_node.random_unexplored_action()
        self.current_node.create_child_keep_board(action)
        self.play_action(action)
        self.size += 1

    def one_game(self, node):
        board = copy.deepcopy(node.state)
        while board.victory is '':
            actions = board.actions()
            move = random.randint(0, len(actions) - 1)
            play = actions[move]
            board.drop_piece(play)
        if board.victory == ".":
            vic = 0
        elif board.victory == "X":
            vic = 1
        elif board.victory == "O":
            vic = -1   
        return vic  

    def simulate(self):
        return (self.one_game(self.current_node))

    def backpropagate(self, node, cacahuetas):
        while node is not None:
            if node.state.turn % 2 == 1:
                node.total_reward += cacahuetas 
            else:
                node.total_reward -= cacahuetas
            node.visits += 1
            node = node.daddy
          
    def explore(self):
        pass

    def exploit(self):
        pass
    
    def play_action(self, action):
        self.current_node.play_move_keep_board(action)
        self.current_node = self.current_node.children.get(action)

    def play(self):
        self.current_node = self.tree.root
        self.current_node.state = state()
        self.selection()
        self.expand()
        cacahueta = self.simulate()
        self.backpropagate(self.current_node, cacahueta)

    def choose_move(self)   :
        if (self.current_node.is_fully_expanded == 1):
            self.play_action(self.select_greedy())
        else:
            self.expand()
    
    def play_vs_MCTS(self):
        self.current_node = self.tree.root
        self.current_node.state = state()
        while self.current_node.state.victory is '':
            self.choose_move()
            print_state(self.current_node.state)
            self.play_action(int(input()))
            print_state(self.current_node.state)

                


