from state import state
from deep import Deep_Neural_Net
import random
import math
import copy
from color import *

class node():
    def __init__(self, state = state(), parent = None):
        self.state = state # WE CAN REMOVE THE STATE AS THE PATH HOLDS IT !!!
        self.daddy = parent
        self.visits = 0
        self.total_reward = 0
        self.children = {}
        self.actions = self.state.actions()
        self.is_fully_expanded = False #TO UPDATE
        if (self.state == None):
            self.state = state()

    def add_child(self, action, child = None):
        '''
            adds existing node to the children dictionary of self. (with action as its key)
            if no node is given as argument, will create new node instance (sharing ref to board)
        '''
        if (child == None):
            child = node(self.state, self)
        self.children[action] = child
        if len(self.actions) == len(self.children):
            self.is_fully_expanded = True

    def expand(self):
        '''
            Expands node. Creates all child nodes and adds them to dictionary.
            USE ONLY ON NODES WITH NO CHILDREN, existing children will be replaced.
            return number of nodes added to tree
        '''
        count = 0
        last_move_tmp = copy.deepcopy(self.state.last_move)
        victory_tmp = copy.copy(self.state.victory) #undeep check
        for act in self.actions:
            self.state.drop_piece(act)
            child = node(self.state, self)
            self.children[act] = child
            self.state.undrop_piece()
            self.state.last_move = copy.copy(last_move_tmp)
            self.state.victory = copy.copy(victory_tmp)
            count += 1
        self.is_fully_expanded = True
        return (count)

    def UCB1(self):
        '''
            returns UCB1 value of current node (will crash if run on root of tree)
        '''
        if (self.visits == 0):
            return (1234) #arbitrary big number here
        sqrt_log_of_visits = math.sqrt(math.log(self.daddy.visits) / self.visits)
        reward_visits = (self.total_reward / self.visits)
        c_explo = math.sqrt(2)
        #c_explo = 2
        result = reward_visits + c_explo * sqrt_log_of_visits
        return result

    def PUCT(self, DNN):
        PUCT = []
        C = 1
        #DNN = Deep_Neural_Net()
        DNN.convert_state(self.state)
        DNN.run()
        for act in self.actions:
            child = self.children.get(act)
            if child != None:
                Q = child.total_reward
                P = DNN.policy[act]
                N = math.sqrt(self.visits) / (1 + child.visits)
                PUCT.append([Q + (C * P * N), act])
        best_puct = -1234567890
        pos = -1
        for i in range(len(PUCT)):
            if PUCT[i][0] > best_puct:
                best_puct = PUCT[i][0]
                pos = PUCT[i][1]
        return pos

    def winrate(self):
        return (self.total_reward / self.visits)
    
    def unexplored_actions(self):
        unexplored_moves = []
        for a in self.actions :
            if (self.children.get(a) == None):
                unexplored_moves.append(a)
        return (unexplored_moves)

    def random_unexplored_action(self):
        act = self.unexplored_actions()
        return (act[random.randint(0, len(act) - 1)])

    def play_move_keep_board(self, action):
        '''
            will do the action and pass a pointer/reference to the board to the corresponding
            child node. The child node will be created if necessary
        '''
        existing_child = self.children.get(action)
        self.state.drop_piece(action)
        if (existing_child == None):
            self.children[action] = node(self.state, self)
            if (len(self.children) == len(self.actions)):
                self.is_fully_expanded = True
        else:
            self.children.get(action).state = self.state

    def create_child_keep_board(self, action):
        self.state.drop_piece(action)
        self.children[action] = node(self.state, self)
        if (len(self.children) == len(self.actions)):
            self.is_fully_expanded = True
                
    def display(self, max_nb_size = 5):
        print(" " * (max_nb_size - len(str(self.total_reward))), end="")
        print(self.total_reward, "/", self.visits, end="")
        print(" " * (max_nb_size - len(str(self.visits))), end="")
        print("=", str(self.UCB1())[:7], RESET, end="")
