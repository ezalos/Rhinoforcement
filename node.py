from state import state
from deep import Deep_Neural_Net
import random
import math
import copy
from color import *

class node():
    def __init__(self, state = state(), parent = None):
        self.state = state
        self.daddy = parent
        self.visits = 0
        self.total_reward = 0
        self.player = self.state.player
        self.Q = None
        self.P = None
        self.children = {}
        self.P = None
        self.actions = self.state.actions()
        self.is_fully_expanded = (len(self.actions) == len(self.children))
        self.is_terminal = 1 if (self.state.victory != '') else 0# or (len(self.actions) == 0)

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

    def do_action(self, action):
        '''
            returns the child node corresponding to the action
        '''
        self.state.do_action(action)
        ret = self.children.get(action)
        if (ret == None):
            print("trying to access a non existing node THIS SHOULD NEVER PRINT")

    def UCB1(self):
        '''
            returns UCB1 value of current node (will crash if run on root of tree)
        '''
        if (self.visits == 0):
            return (1234) #arbitrary big number here
        c_explo = math.sqrt(2)
        return ((self.total_reward / self.visits) + c_explo * math.sqrt(math.log(self.daddy.visits) / self.visits))

    def PUCT(self, DNN):
        PUCT = []
        C = 1
        #DNN = Deep_Neural_Net()
        DNN.convert_state(self.state)
        DNN.run()
        for act in self.actions:
            child = self.children.get(act)
            if child != None:
                Q = child.total_reward / (1 + child.visits) #should be this
                Q = child.total_reward #but this seems better
                if self.P == None:
                    self.P = DNN.policy[act]
                N = math.sqrt(self.visits) / (1 + child.visits)
                PUCT.append([Q + (C * self.P * N), act])
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
        #print("=", str(self.PUCT())[:7], RESET, end="")
        print(RESET, end="")

    def print_n_floor(self, node = None, limit=1, deepness=0):
        if node == None:
            node = self
        max = len(str(self.visits))
        best_UCB1 = -100000000000
        best_action = -1 #quick fix
        for action in self.actions :
            danger = self.children.get(action)
            if danger != None:
                new_UCB1 = danger.UCB1()
                if (new_UCB1 > best_UCB1):
                    best_UCB1 = new_UCB1
                    best_action = action
        for act in self.actions:
            child = self.children.get(act)
            if deepness < 2 or act == self.actions[0]:
                print("    " * deepness, end="")
            if act == best_action:
                print(UNDERLINE, end="")
            if deepness % 2 == 1:
                print(BLUE, end="")
            else:
                print(RED, end="")
            print(act, "-->", end="")
            if (child != None):
                child.display(max)
                if deepness < 2:
                    print("")
                elif act != self.state.actions()[-1]:
                    print(" | ", end="")
                if deepness < limit:
                    self.print_n_floor(child, limit, deepness + 1)
            else:
                print("  NONE", RESET)
        if deepness >= 2:
            print("")
