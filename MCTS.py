from state import state
from node import node
import copy
import random
import time
from color import *
from deep import Deep_Neural_Net
import numpy
from data import dataset
from data import datapoint
import numpy as np
import sklearn

DEBUG = 0
class MCTS():

    def __init__(self, node = node(), dataset = dataset(), tree_policy = None, rollout_policy = None):
        '''
            tree policy takes a node and returns an action, rollout_policy takes a node and retruns a value.
        '''
        self.current_node = node
        self.root = self.current_node
        self.tree_root = self.current_node
        self.size = 0
        self.dataset = dataset
        if (tree_policy != None):
            self.tree_policy = tree_policy
        else:
            self.tree_policy = lambda : self.select()
        if (rollout_policy != None):
            self.rollout_policy = rollout_policy
        else:
            self.rollout_policy = lambda : self.simulate()
        self.dnn = Deep_Neural_Net()

    def launch(self):
        self.current_node = self.tree_root
        self.current_node.state.reset()
        self.MCTS_to_reward()
        self.tree_root.print_n_floor(self.tree_root, 0)
    
    def MCTS_to_reward(self):
        node = self.current_node
        if (node.is_terminal): #game is finished
            node.visits += 1
            v = node.state.get_reward()
            node.total_reward += v
            #print("PLAYA: ", node.player, "winna: ", node.state.victory, "  v: ", v)
            return -v

        if (node.is_fully_expanded == False and node.visits == 0):  #first visit
            v = self.rollout_policy()
            #node.P = 1 #cheat
            node.visits += 1
            node.total_reward += v
            return -v

        if (node.is_fully_expanded == False and node.visits != 0):  #second visit
            self.expand_current_node()
            action = self.tree_policy()
            #safe_action = node.state.actions()[action]
            self.play_action(action)
            v = self.MCTS_to_reward()
            node.visits += 1
            node.total_reward += v
            return -v

        if (node.is_fully_expanded):
            action = self.tree_policy()
            #safe_action = node.state.actions()[action]
            self.play_action(action)
            v = self.MCTS_to_reward()
            node.visits += 1    #increase here or before PUCT evaluation ?
            node.total_reward += v
            return -v
        print(" YOOOOOO FUCKED UP BROOOO")

    def self_play(self, dataset = dataset(), iterations = 400): # DIRICHELET NMOISE
        self.root.state.display()
        if (self.root.is_terminal):
            return -(self.root.state.get_reward())
        initial_state = copy.deepcopy(self.root.state)

        for _ in range(iterations):
            self.current_node = self.root
            self.current_node.state.copy(initial_state)
            self.MCTS_to_reward()
        
        self.current_node = self.root
        self.current_node.state.copy(initial_state)
        #self.current_node.print_n_floor(limit=1)
        policy = self.policy_policy()
        dataset_index = dataset.add_point(state=self.root.state, policy=policy)
        action = np.random.choice(7, 1, p=policy)[0]
        #action = self.select_highest_visits()
        self.play_action(action)
        self.root = self.current_node
        v = self.self_play(dataset)
        dataset.data[dataset_index].V = np.array([v])
        return -v

    def play_one_move(self, iterations = 400):
        if (self.root.is_terminal):
            return -(self.root.state.get_reward())
        initial_state = copy.deepcopy(self.root.state)

        for _ in range(iterations):
            self.current_node = self.root
            self.current_node.state.copy(initial_state)
            self.MCTS_to_reward()
        
        self.current_node = self.root
        self.current_node.state.copy(initial_state)
        action = self.select_highest_visits()
        self.play_action(action)
        self.root = self.current_node

    def self_play_new_game(self):
        print("new game")
        self.root = self.tree_root
        self.current_node = self.root
        self.current_node.state.reset()
        self.self_play(self.dataset)
    
    def policy_policy(self, node = None): #IT FUCKED UP
        '''
            NO TEMPERATURE YET
            return policy vector based on visit numbers
            USES ROOT not current node !!!!
            state must correspond to node
        '''
        if node == None :
            node = self.root
        policy = np.zeros(7)
        for action in node.actions:
            policy[action] = node.children.get(action).visits
#        if (self.root.state.turn < 25): #DEFINE here
#            temperature = 1
#        else:
#            temperature = 1   #SHOULD BE 0.1
#        for idx in range(len(policy)):
#            policy[idx] = policy[idx]**(1/temperature)
        summ = sum(policy)
        for action in node.actions:
            policy[action] = policy[action] / summ
        return (policy)

    def simulate(self, node = None, f = lambda x : random.randint(0, len(x) - 1)):
        '''
            plays random moves until the game ends.
            returns the winner as a string
        '''
        if node == None:
            node = self.current_node
        state = copy.deepcopy(node.state) # maybe remove this later
        while state.victory is '':
            actions = state.actions()
            move = f(actions)
            play = actions[move]
            state.do_action(play)

        winner = state.victory
        if (winner == node.player):
            v = -1
        elif (winner == "."):
            v = 0
        else:
            v = 1
        return (v)

    def play_action(self, action):
        '''
            Does the action.
            Changes current node and the state
            the child node MUST ALREADY EXIST
        '''     
        self.current_node.do_action(action)
        self.current_node = self.current_node.children.get(action)
        self.current_node.state = self.current_node.daddy.state #should be removed later

    def policy_UCB1(self):
        policy = numpy.full([7], numpy.NINF, dtype=float)
        for action in self.current_node.actions :
            policy[action] = self.current_node.children.get(action).UCB1()
        best_UCB1 = policy.max()
        policy[policy != best_UCB1] = 0.0
        policy = policy / numpy.sum(policy)
        return(policy)
    
    def policy_PUCT(self):
        policy = np.array(self.current_node.PUCT(self.dnn))
        best_PUCT = policy.max()
        policy[policy != best_PUCT] = 0.0
        policy = policy / numpy.sum(policy)
        #print("Policy: ", policy)
        return(policy)

    def select_PUCT_policy(self):
        '''
            returns the action leading to the state with the highest UCB score.
            will pick a random action among the best if there are multiple
            3x SLOWER THAN SELECT HIGHEST UCB1
        '''
        policy = self.policy_PUCT()
        #print("Select: ", policy)
        try:
            act = numpy.random.choice(len(policy), 1, p=policy)[0]
        except:
            act = 8
            while (act == 8):
                act = random.randint(0, 6)
                if (policy[act] == 0):
                    act = 8
        #print("Act: ", act)
        safe_action = self.current_node.actions[act]
        return (safe_action)
    
    def select_UCB1_policy(self):
        '''
            returns the action leading to the state with the highest UCB score.
            will pick a random action among the best if there are multiple
            3x SLOWER THAN SELECT HIGHEST UCB1
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
        return self.select_PUCT_policy()
        if (self.current_node.unexplored_babies > 0):
            return (self.select_UCB1_policy())
        else:
            return (self.select_highest_UCB1())

    def expand_current_node(self):
        '''
            create all children for self.current_node
            WILL DESTROY EXISTING CHILDREN
        '''
        self.size += self.current_node.expand() ## ADD DNN INITIALIZATIONS BB

    def human_play_one_move(self):
        if self.current_node.state.victory is '':
            while True:
                to_play = input("What should be your next move ?\n")
                if to_play == "cheat":
                    self.current_node.print_n_floor(self.current_node, limit=0)
                    to_play = None
                elif to_play == "exit":
                    return None
                else:
                    try:    
                        if int(to_play) in self.current_node.state.actions():
                            to_play = int(to_play)
                            self.play_action(to_play)
                            break
                            return True
                        else:
                            print("Invalid move.")
                    except:
                        print("I didn't get that.")
        return True

    def play_vs_MCTS(self):
        self.current_node = self.tree_root
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
                if self.human_play_one_move() == None:
                    return
            else:
                self.root = self.current_node
                self.play_one_move()
            self.current_node.print_n_floor(self.current_node.daddy, limit=0)
        self.current_node.state.display()

    def display_game(self):
        self.current_node = self.tree_root
        self.current_node.state.reset()
        while self.current_node.state.victory is '':
            self.root = self.current_node
            self.play_one_move()
            self.root.state.display()

    def select_highest_visits(self):
        max = -1
        node = self.current_node
        best_a = None
        for act in self.current_node.actions:
            if (node.children[act].visits > max):
                max = node.children[act].visits
                best_a = act
        return (best_a)

    def display(self):
        print("Size MCTS = ", self.size)
        self.tree_root.display()
