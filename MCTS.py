from tree import tree
from state import state
import copy
import random
import time
import numpy

class MCTS():

    iterations_per_turn = 10
    def __init__(self, tree = tree()):
        self.tree = tree
        self.current_node = self.tree.root
        self.size = 0

    def default_policy(self):
        pass

    def policy(self):
        policy = numpy.zeros([7])
        best_action = self.current_node.actions[0]
        best_UCB1 = self.current_node.children.get(best_action).UCB1()
        policy[best_action] = best_UCB1
        new_UCB1 = 0
        count_equal = 0
        for action in self.current_node.actions :
            new_UCB1 = self.current_node.children.get(action).UCB1()
            policy[action] = self.current_node.children.get(action).UCB1()
            if (new_UCB1 > best_UCB1):
                best_UCB1 = new_UCB1
                best_action = action
        for i in range(7):
            if (policy[i] != best_UCB1):
                policy[i] = 0.0
            if (policy[i] == best_UCB1):
                count_equal += 1
        for i in range(7):
            policy[i] = policy[i] / (best_UCB1 * count_equal)
        return (policy)

    def select(self):
        policy = self.policy()
        try:
            act = numpy.random.choice(7, 1, p = policy)[0]
        except:
            print(policy)
            act = 8
            while (act == 8):
                act = random.randint(0, 6)
                if (policy[act] == 0):
                    act = 8
        return (act)

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
            elif (new_UCB1 == best_UCB1):
                if (random.randint(0, 2) == 2):
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
            if (new_visits > best_visits):
                best_visits = new_visits
                best_action = action
        return (best_action)

    def play_action(self, action):
        '''
            Does the action.
            the child node MUST ALREADY EXIST
        '''     
        self.current_node.state.drop_piece(action)
        self.current_node = self.current_node.children.get(action)
        self.current_node.state = self.current_node.daddy.state

    def selection(self):
        while (self.current_node.is_fully_expanded and self.current_node.visits != 0 and self.current_node.state.victory == ''):
            self.play_action(self.select())

    def expand(self):
        '''
            create all children for self.current_node
            WILL DESTROY EXISTING CHILDREN
        '''
        self.size += self.current_node.expand()
    
    def simulate(self, node = None, f = lambda x : random.randint(0, len(x) - 1)):
        '''
            plays random moves until the game ends.
            returns the obtained reward.
        '''
        if node == None:
            node = self.current_node
        state = copy.deepcopy(node.state) # maybe remove this later
        while state.victory is '':
            actions = state.actions()
            move = f(actions)
            play = actions[move]
            state.drop_piece(play)
        return (state.get_reward())

    def backpropagate(self, node, cacahuetas):
        turn = node.state.turn
        while node is not None:
            if turn % 2 == 1:
                node.total_reward += cacahuetas 
            else:
                node.total_reward -= cacahuetas
            node.visits += 1
            turn -= 1
            node = node.daddy

    def play(self):
        '''
            Plays from current node to loss or victory and backpropagates the result
        '''
        self.selection()
        if (self.current_node.state.victory != ''):
            self.backpropagate(self.current_node, self.current_node.state.get_reward())
        else:
            if (self.current_node.visits == 0):
                self.backpropagate(self.current_node, self.simulate())
            elif (self.current_node.is_fully_expanded == False):
                self.expand()
                actions = self.current_node.actions
                self.play_action(actions[random.randint(0, len(actions) - 1)]) #implement winning move here !
                self.backpropagate(self.current_node, self.simulate())

    def self_play_one_move(self, iterations = 400):
        '''
            runs many games from current_node, chooses a move the plays it.
        '''
        initial_state = copy.deepcopy(self.current_node.state)
        initial_node = self.current_node
        for i in range(iterations):                                            # HERE WE DEFINE ITERATIONS PER TURN !!!!
            self.current_node = initial_node
            self.current_node.state.copy(initial_state)
            self.play()
        self.current_node = initial_node
        self.current_node.state.copy(initial_state)
        self.play_action(self.select_most_visits())

    def self_play_one_move_time(self, time_per_move = 1):
        '''
            runs many games from current_node, chooses a move the plays it.
        '''
        initial_state = copy.deepcopy(self.current_node.state)
        initial_node = self.current_node
        timer = time.time()
        while (time.time() < timer + time_per_move):                                            # HERE WE DEFINE ITERATIONS PER TURN !!!!
            self.current_node = initial_node
            self.current_node.state.copy(initial_state)
            self.play()
        self.current_node = initial_node
        self.current_node.state.copy(initial_state)
        self.play_action(self.select_most_visits())

    def self_play_one_game(self):
        '''
            resets current node and state then plays a game vs itself
        '''
        self.current_node = self.tree.root
        self.current_node.state.reset()
        while (self.current_node.state.victory is ''):
            self.self_play_one_move(400)

    def play_vs_MCTS(self):
        self.current_node = self.tree.root
        self.current_node.state.reset()
        while self.current_node.state.victory is '':
            self.self_play_one_move_time()
            self.current_node.state.display()
            if self.current_node.state.victory is '':
                self.play_action(int(input()))
                self.current_node.state.display()

    def display(self):
        print("Size MCTS = ", self.size)
        self.tree.display()