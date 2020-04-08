from state import state
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
        self.PUCT_ = None
        self.children = {}
        self.P = None
        self.P_update = 0
        self.actions = self.state.actions()
        self.unexplored_babies = len(self.actions)
        self.is_fully_expanded = (len(self.actions) == len(self.children))
        self.is_terminal = (self.state.victory != '') # or (len(self.actions) == 0)

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
            self.state.do_action(act)
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
            does action
        '''
        if (self.unexplored_babies > 0):
            self.unexplored_babies -= 1
        self.state.do_action(action)

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
        run = 0
        C = 1
        #DNN = Deep_Neural_Net()
        for act in self.actions:
            child = self.children.get(act)
            if child != None:
                Q = child.total_reward / (1 + child.visits) #should be this
                Q = child.total_reward #but this seems better
                N = math.sqrt(self.visits) / (1 + child.visits)
                if child.P == None or self.P_update < DNN.version:
                    self.P_update = DNN.version
                    if not run:
                        DNN.convert_state(self.state)
                        DNN.run()
                        run = 1
                    child.P = DNN.policy[act]
                    child.PUCT_ = (Q + (C * child.P * N))
                PUCT.append(Q + (C * child.P * N))
            else:
                PUCT.append(0)
        #print("Puct: ", PUCT)
        return PUCT
        best_puct = -1234567890
        pos = -1
        for i in range(len(PUCT)):
            if PUCT[i][0] > best_puct:
                best_puct = PUCT[i][0]
                pos = PUCT[i][1]
        return pos
 
    def display(self, max_nb_size = 5, best_winrate=False, best_visits=False, best_puct=False, best_ucb1=False):
        print(" " * (max_nb_size - len(str(self.total_reward))), end="")
        if best_winrate:
            print(GREEN, end="")
        print(self.total_reward, end="")
        if self.state.player == "O":
            print(BLUE, end="")
        else:
            print(RED, end="")
        print(" / ", end="")
        if best_visits:
            print(GREEN, end="")
        elif self.state.player == "O":
            print(BLUE, end="")
        else:
            print(RED, end="")
        print(self.visits, end="")
        print(" " * (max_nb_size - len(str(self.visits))), end="")
        if best_puct:
            print(GREEN, end="")
        elif self.state.player == "O":
            print(BLUE, end="")
        else:
            print(RED, end="")
        print("   PUCT ", str(self.PUCT_)[:7], end="")
        if best_ucb1:
            print(GREEN, end="")
        elif self.state.player == "O":
            print(BLUE, end="")
        else:
            print(RED, end="")
        print("   UCB1 ", str(self.UCB1())[:7], RESET, end="")
        print(RESET, end="")

    def print_n_floor(self, node = None, limit=1, deepness=0):
        if node == None:
            node = self
        max = len(str(node.visits)) + 1
        best_UCB1 = -100000000000
        best_PUCT = -100000000000
        best_action_UCB1 = -1 #quick fixi
        best_action_PUCT = -1 #quick fixi
        max_visits = -1
        best_max_visits = -1
        best_winrate = -100000000000
        best_action_winrate = -1
        for action in node.actions :
            danger = node.children.get(action)
            if danger != None:
                new_UCB1 = danger.UCB1()
                if (danger.UCB1() > best_UCB1):
                    best_UCB1 = danger.UCB1()
                    best_action_UCB1 = action
                if (danger.PUCT_ > best_PUCT):
                    best_PUCT = danger.PUCT_
                    best_action_PUCT = action
                if (danger.visits > max_visits):
                    max_visits = danger.visits
                    best_max_visits = action
                if (danger.total_reward / danger.visits > best_winrate):
                    best_winrate = danger.total_reward / danger.visits
                    best_action_winrate = action
        for act in node.actions:
            child = node.children.get(act)
            if deepness < 2 or act == node.actions[0]:
                print("    " * deepness, end="")
            if act == best_max_visits:
                visits_high = True
            else:
                visits_high = False
            if act == best_action_winrate:
                winrate_high = True
            else:
                winrate_high = False
            if act == best_action_UCB1:
                #print(UNDERLINE, end="")
                ucb1_high = True
            else:
                ucb1_high = False
            if act == best_action_PUCT:
                #print(BOLD, end="")
                puct_high = True
            else:
                puct_high = False
            if node.state.player == "O":
                print(BLUE, end="")
            else:
                print(RED, end="")
            print(act, "-->", end="")
            if (child != None):
                child.display(max, winrate_high, visits_high, puct_high, ucb1_high)
                if deepness < 2:
                    print("")
                elif act != node.state.actions()[-1]:
                    print(" | ", end="")
                if deepness < limit:
                    node.print_n_floor(child, limit, deepness + 1)
            else:
                print("  NONE", RESET)
        if deepness >= 2:
            print("")
