from state import state
import random
import math

PURPLE = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

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
        sqrt_log_of_visits = math.sqrt(math.log(self.daddy.visits / self.visits))
        reward_visits = (self.total_reward / self.visits)
        #c_explo = math.sqrt(2)
        c_explo = 2
        result = reward_visits + c_explo * sqrt_log_of_visits
        return result

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
        '''
        will do the action and pass a pointer/reference to the board to the corresponding
        child node. The child node will be created if necessary
        '''
        existing_child = self.children.get(action)
        self.state.drop_piece(action)
        if (existing_child == None):
            self.children[action] = node(self.state, self)
            if (len(self.children) == len(self.actions)):
                self.is_fully_expanded = 1
        else:
            self.children.get(action).state = self.state

    def create_child_keep_board(self, action):
        self.state.drop_piece(action)
        self.children[action] = node(self.state, self)
        if (len(self.children) == len(self.actions)):
            self.is_fully_expanded = 1
                
    def display(self, max_nb_size = 5):
        print(" " * (max_nb_size - len(str(self.total_reward))), end="")
        print(self.total_reward, "/", self.visits, end="")
        print(" " * (max_nb_size - len(str(self.visits))), end="")
        print("=", str(self.UCB1())[:4], RESET, end="")
