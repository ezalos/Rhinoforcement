from tree import tree
from state import state
import copy
import random

class MCTS():

    iterations_per_turn = 100
    def __init__(self, tree = tree()):
        self.tree = tree
        self.current_node = self.tree.root
        self.size = 0

    def default_policy(self):
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
            #print("action: ", action, "best_action: ", best_action, "UCB: ", new_UCB1, "best: ", best_UCB1)
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
        picks a move among those never played, and PLAYS THE MOVE.
        Creating the corresponding child node.
        '''
        if (self.current_node.is_fully_expanded):
            print("youre trying to expand a fully expanded node and this should never print")
            return 
        action = self.current_node.random_unexplored_action()
        self.play_action(action)
        self.size += 1

    def get_cacahuetas(self, state = None):
        if state == None:
            state = self.current_node.state
        if state.victory == ".":
            vic = 0
        elif state.victory == "X":
            vic = 1
        elif state.victory == "O":
            vic = -1
        else:
            return None
        return vic  
    
    def simulate(self, node = None, f = lambda x : random.randint(0, len(x) - 1)):
        if node == None:
            node = self.current_node
        state = copy.deepcopy(node.state) # maybe remove this later
        while state.victory is '':
            actions = state.actions()
            move = f(actions)
            play = actions[move]
            state.drop_piece(play)
        return (self.get_cacahuetas(state))

    def backpropagate(self, node, cacahuetas):
        while node is not None:
            if node.state.turn % 2 == 1:
                node.total_reward += cacahuetas 
            else:
                node.total_reward -= cacahuetas
            node.visits += 1
            node = node.daddy
    
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
            #self.play_action(int(input()))
            print("AI play")
            print_state(self.current_node.state)
            self.play_action(int(input()))
            print_state(self.current_node.state)

    def iterate_then_choose_move(self):
        initial_node = self.current_node
        initial_state = copy.deepcopy(self.current_node.state)
        for i in range(self.iterations_per_turn):
            self.current_node = initial_node
            self.current_node.state = copy.deepcopy(initial_state)
            while (self.current_node.is_fully_expanded):
                action = self.select()
                self.play_action(action)
            if (self.current_node.state.victory != ''): #no fuckin clue how string comparisons work carefull untested
                self.backpropagate(self.current_node, self.get_cacahuetas())
            else:
                self.expand()
                self.backpropagate(self.current_node, self.simulate())
        self.current_node = initial_node
        self.current_node.state = copy.deepcopy(initial_state)
        return (self.select())

    def self_play(self):
        while (self.current_node.state.victory == ''):
            action = self.iterate_then_choose_move()
            self.play_action(action)
        self.backpropagate(self.current_node, self.get_cacahuetas()) # maybe double backprop
        
