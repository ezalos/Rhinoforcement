from tree import tree
from state import state
import copy
import random
import time
from color import *
import numpy
from data import dataset
from data import datapoint


class MCTS():

    iterations_per_turn = 10
    def __init__(self, tree = tree()):
        self.tree = tree
        self.current_node = self.tree.root
        self.size = 0

    def default_policy(self):
        pass

    def policy_UCB1(self):
        policy = numpy.full([7], numpy.NINF, dtype=float) #arbitrary small number so it will not be the maximum
        for action in self.current_node.actions :
            policy[action] = self.current_node.children.get(action).UCB1()
        best_UCB1 = policy.max()
        policy[policy != best_UCB1] = 0.0
        policy = policy / (numpy.count_nonzero(policy) * best_UCB1)
        return(policy)

    def select_UCB1_policy(self):
        '''
            returns the action leading to the state with the highest UCB score.
            will pick a random action among the best if there are multiple
            3x SLOWER THAN SELECT HIGHEST UCBgi tpudh
        '''
        policy = self.policy_UCB1()
        try:
            act = numpy.random.choice(7, 1, p = policy)[0]
        except:
            act = 8
            while (act == 8):
                act = random.randint(0, 6)
                if (policy[act] == 0):
                    act = 8
        return (act)

    def select_highest_UCB1(self):
        '''
            returns the action leading to the state with the highest UCB score
        '''
        best_action = self.current_node.actions[0]
        best_UCB1 = self.current_node.children.get(best_action).UCB1()
        for action in self.current_node.actions :
            new_UCB1 = self.current_node.children.get(action).UCB1()
            if (new_UCB1 > best_UCB1):
                best_UCB1 = new_UCB1
                best_action = action
            elif (new_UCB1 == best_UCB1):
                if (random.randint(0, 2) == 2):
                    best_action = action
        return (best_action)

    def select(self):
        return (self.select_highest_UCB1())

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

    def backpropagate(self, cacahuetas, node = None):
        if (node == None):
            node = self.current_node
        if node.state.player == "0" :
            cacahuetas = (-cacahuetas)
        while node is not None:
            node.total_reward += cacahuetas
            node.visits += 1
            cacahuetas = (-cacahuetas)
            node = node.daddy

    def backpropagate_old(self, cacahuetas):
        node = self.current_node
        if node.state.player == "0" :
            cacahuetas = (-cacahuetas)
        while (node != self.tree.current_root):
            node.total_reward += cacahuetas
            node.visits += 1
            cacahuetas = (-cacahuetas)
            node = node.daddy
        node.total_reward += cacahuetas
        node.visits += 1

    def play(self):
        '''
            Plays from current node to loss or victory and backpropagates the result
        '''
        self.selection()
        if (self.current_node.state.victory != ''):
            self.backpropagate(self.current_node.state.get_reward())
        else:
            if (self.current_node.visits == 0):
                self.backpropagate(self.simulate())
            elif (self.current_node.is_fully_expanded == False):
                self.expand()
                actions = self.current_node.actions
                self.play_action(actions[random.randint(0, len(actions) - 1)]) #implement winning move here !
                self.backpropagate(self.simulate())

    def self_play_one_move(self, dataset, iterations = 400):
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
        dataset.add_point(self.current_node)
        chosen_action = self.select_most_visits()
        self.play_action(chosen_action)
        self.tree.current_root = self.current_node

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
        chosen_action = self.select_most_visits()
        self.play_action(chosen_action)
        self.tree.current_root = self.current_node

    def self_play_one_game(self, dataset):
        '''
            resets current node and state then plays a game vs itself
        '''
        self.current_node = self.tree.root
        self.tree.current_root = self.tree.root
        self.current_node.state.reset()
        while (self.current_node.state.victory is ''):
            self.self_play_one_move(dataset)
        dataset.add_value_to_set(self.current_node.state.get_reward(), self.current_node)

    def human_play_one_move(self):
        if self.current_node.state.victory is '':
            to_play = None
            while to_play == None:
                to_play = input("What should be your next move ?\n")
                if to_play == "cheat":
                    self.tree.print_n_floor(self.current_node, limit=0)
                    to_play = None
                else:
                    try:
                        to_play = int(to_play)
                        if to_play < 0 or to_play > 6:
                            to_play = None
                        elif self.current_node.state.board[0, to_play] != " ":
                            to_play = None
                            print("Invalid move")
                    except:
                        to_play = input("Exit game? [y/n]\n")
                        if to_play == "y":
                            return
                        else:
                            to_play = None
            self.play_action(to_play)

    def play_vs_MCTS(self):
        self.current_node = self.tree.root
        self.current_node.state.reset()
        while (True):
            play_as = input("Play first or second ? [1/2]\n")
            if play_as == "1":
                play_as = "X"; break
            elif play_as == "2":
                play_as = "O"; break
            else:
                print("I didn't get that.")
        while self.current_node.state.victory is '':
            self.current_node.state.display()
            if self.current_node.state.player == play_as:
                self.human_play_one_move()
            else:
                self.self_play_one_move_time()
            self.tree.print_n_floor(self.current_node.daddy, limit=0)
        self.current_node.state.display()

    def display(self):
        print("Size MCTS = ", self.size)
        self.tree.display()
