from tree import tree
from state import state
import copy
import random

class MCTS():

    iterations_per_turn = 10
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

    def select_most_visits(self):
        '''
            returns the action leading to the state with the most visits
        '''
        best_action = self.current_node.actions[0]
        best_visits = self.current_node.children.get(best_action).visits
        new_visits = 0
        for action in self.current_node.actions :
            new_visits = self.current_node.children.get(action).visits
            #print("action: ", action, "best_action: ", best_action, "UCB: ", new_UCB1, "best: ", best_UCB1)
            if (new_visits > best_visits):
                best_visits = new_visits
                best_action = action
        return (best_action)
        
    def selection(self):
        while (self.current_node.is_fully_expanded == True and self.current_node.visits != 0 and self.current_node.state.victory is ''):
            self.play_action(self.select())

    def expand(self):
        '''
            create all children for self.current_node
            WILL DESTROY EXISTING CHILDREN
        '''
        self.current_node.expand()
        self.size += 7
    
    def simulate(self, node = None, f = lambda x : random.randint(0, len(x) - 1)):
        '''
            plays random moves until the game ends.
            returns the obtained reward.
        '''
        if node == None:
            node = self.current_node
        state = node.state # maybe remove this later
        while state.victory is '':
            actions = state.actions()
            move = f(actions)
            play = actions[move]
            state.drop_piece(play)
        return (state.get_reward())

    def backpropagate(self, node, cacahuetas):
        while node is not None:
            if node.state.turn % 2 == 1:
                node.total_reward += cacahuetas 
            else:
                node.total_reward -= cacahuetas
            node.visits += 1
            node = node.daddy
    
    def play_action(self, action):
        '''
            Does the action.
            the child node MUST ALREADY EXIST
        '''     
        self.current_node.state.drop_piece(action)
        self.current_node = self.current_node.children[action]
   
    def play(self):
        '''
            Plays from current node to loss or victory and backpropagates the result
        '''
        self.selection()
        if (self.current_node.visits != 0 and self.current_node.is_fully_expanded == False and self.current_node.state.victory is ''): # fishy feeling here
            self.expand()
            actions = self.current_node.actions
            self.play_action(actions[random.randint(0, len(actions) - 1)])
        cacahueta = self.simulate()
        self.backpropagate(self.current_node, cacahueta)

    def self_play_one_move(self):
        '''
            runs many games from current_node, chooses a move the plays it.
        '''
        initial_state = self.current_node.state
        initial_node = self.current_node
        for i in range(400):
            self.current_node = initial_node
            self.current_node.state = copy.deepcopy(initial_state)
            self.play()
        self.current_node = initial_node
        self.current_node.state = initial_state
        self.play_action(self.select_most_visits())

    def self_play_one_game(self):
        '''
            resets current node and state then plays a game vs itself
        '''
        self.current_node = self.tree.root
        self.current_node.state = state()
        while (self.current_node.state.victory is ''):
            self.self_play_one_move()

    def play_vs_MCTS(self):
        self.current_node = self.tree.root
        self.current_node.state = state()
        while self.current_node.state.victory is '':
            self.choose_move() #stupid
            #self.play_action(int(input()))
            print("AI play")
            self.current_node.state.display()
            if self.current_node.state.victory is '':
                self.play_action(int(input()))
                self.current_node.state.display()

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

    def self_play_old(self):
        self.current_node = self.tree.root
        self.current_node.state = state()
        while (self.current_node.state.victory == ''):
            action = self.iterate_then_choose_move()
            self.play_action(action)
        self.backpropagate(self.current_node, self.get_cacahuetas()) # maybe double backprop

    def display(self):
        print("Size MCTS = ", self.size)
        self.tree.display()


#    def get_cacahuetas(self, state = None):
#        if state == None:
#            state = self.current_node.state
#        if state.victory == ".":
#            vic = 0
#        elif state.victory == "X":
#            vic = 1
#        elif state.victory == "O":
#            vic = -1
#        else:
#            return None
#        return vic  


#    def play(self):
#        self.current_node = self.tree.root
#        self.current_node.state = state()
#        self.selection()
#        self.expand()
#        cacahueta = self.simulate()
#        self.backpropagate(self.current_node, cacahueta)
 