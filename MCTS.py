from state import state
from node import node
import copy
import random
import time
from color import *
from deep import ConnectNet
import numpy
from data import datapoint
import numpy as np
import sklearn
import torch
import math
from data import Dataseto

DEBUG = 0
# need net.evalueate to return P and Q as numpy array
# make state.valid_moves_mask to return bool array / 1 0 array



class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

ARGS = DotDict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'cexplo': 2,
    'verbose': 0,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

class MCTSO():
    def __init__(self, net, args=ARGS):
        self.net = net
        self.args = args
        self.Qsa = {}       # Q values for s,a
        self.Nsa = {}       # visit count for edge s,a
        self.Ns = {}        # visit count board s
        self.Ps = {}        # policy (returned by neural net)

        self.Ts = {}        # stores is_terminal for board s
        self.Ms = {}        # stores game.valid_moves_mask for board s
    
    def get_policy(self, state, temperature = 1):
        s = state.stringify()
        out = np.zeros([7]) #DEFINE
        for act in state.actions():
            out[act] = self.Nsa[(s, act)] if (s, act) in self.Nsa else 0
        
        policy = np.zeros([7])
        if (temperature == 0):
            policy[out.argmax()] = 1
            return (policy)
        
        for act in state.actions():
            policy[act] = out[act]**(1.0 / temperature)
        summ = sum(policy)
        for act in state.actions():
            policy[act] = policy[act] / summ

        return policy

    def simulate(self, statee = None):
        '''
            plays random moves until the game ends.
            returns the winner as a string
        '''
        if statee == None:
            print("FAK NO state IN SIMULATOR")
        state = copy.deepcopy(statee) # maybe remove this later
        while state.victory is '':
            actions = state.actions()
            move = random.randint(0, len(actions) - 1)
            play = actions[move]
            state.do_action(play)

        winner = state.victory
        if (winner == statee.player): #check this bs
            v = -1.0
        elif (winner == "."):
            v = 0.000001
        else:
            v = 1.0
        return v

    def search_UCB1(self, state):
        state_string = state.stringify()
        if (state_string not in self.Ts): #terminal states
            self.Ts[state_string] = state.is_game_over()
        if (self.Ts[state_string] != 0):
            return (self.Ts[state_string])

        if (state_string not in self.Ps): #LEAF
            #self.Ps[state_string], value = self.net.evaluate(torch.from_numpy(state.encode_board()).float())
            valid_moves = state.valid_moves_mask()
            self.Ps[state_string] = valid_moves
            #self.Ps[state_string] = self.Ps[state_string] * valid_moves #verify product
            sum = np.sum(self.Ps[state_string])
            if (sum > 0): #here we assume elements of PS are positive !!
                self.Ps[state_string] = self.Ps[state_string] / sum
            else: #all valid moves have p = 0
                print("no valid moves in Ps[State]")
                self.Ps[state_string] = valid_moves
                self.Ps[state_string] /= np.sum(self.Ps[state_string])
        
            self.Ms[state_string] = valid_moves
            self.Ns[state_string] = 0
            value = self.simulate(state)
            return value


        #"Normal" node
        best_UCB = -123456789
        best_action = None

        for action in state.actions():
            if (state_string, action) in self.Qsa:
                #u = self.Qsa[(state_string, action)] + self.args.CPUCT*self.Ps[state_string][action]*math.sqrt(self.Ns[state_string])/(1+self.Nsa[(state_string,action)])
                u = self.Qsa[(state_string, action)] + self.args.cexplo * math.sqrt(math.log(self.Ns[state_string]) / self.Nsa[(state_string, action)])
            else:
                #u = self.args.cpuct*self.Ps[state_string][action]*math.sqrt(self.Ns[state_string] + 0.00000001) # Q could be initializerd by network here
                u = 123456
            if u > best_UCB:
                best_UCB = u
                best_action = action
        a = best_action
        
        state.do_action(a) #state is now next_state
        v = self.search(state)

        if ((state_string, a) in self.Qsa):
            self.Qsa[(state_string, a)] = (self.Qsa[(state_string, a)] * self.Nsa[(state_string, a)] + v) / (self.Nsa[(state_string, a)] + 1)
            self.Nsa[(state_string, a)] += 1
        else:
            self.Qsa[(state_string, a)] = v
            self.Nsa[(state_string, a)] = 1
        #print("Player: ", playa, "winner: ", winner, "v: ", v)
        self.Ns[state_string] += 1
        return -v

    def search(self, state):
        state_string = state.stringify()
        if (state_string not in self.Ts): #terminal states
            self.Ts[state_string] = state.is_game_over()
        if (self.Ts[state_string] != 0):
            return (self.Ts[state_string])

        if (state_string not in self.Ps): #LEAF
            #self.Ps[state_string], value = self.net.evaluate(torch.from_numpy(state.encode_board()).float())
            valid_moves = state.valid_moves_mask()
            self.Ps[state_string] = valid_moves
            #self.Ps[state_string] = self.Ps[state_string] * valid_moves #verify product
            sum = np.sum(self.Ps[state_string])
            if (sum > 0): #here we assume elements of PS are positive !!
                self.Ps[state_string] = self.Ps[state_string] / sum
            else: #all valid moves have p = 0
                print("no valid moves in Ps[State]")
                self.Ps[state_string] = valid_moves
                self.Ps[state_string] /= np.sum(self.Ps[state_string])
        
            self.Ms[state_string] = valid_moves
            self.Ns[state_string] = 0
            value = self.simulate(state)
            return value


        #"Normal" node
        best_UCB = -123456789
        best_action = None

        for action in state.actions():
            if (state_string, action) in self.Qsa:
                #u = self.Qsa[(state_string, action)] + self.args.CPUCT*self.Ps[state_string][action]*math.sqrt(self.Ns[state_string])/(1+self.Nsa[(state_string,action)])
                u = self.Qsa[(state_string, action)] + self.args.cexplo * math.sqrt(math.log(self.Ns[state_string]) / self.Nsa[(state_string, action)])
            else:
                #u = self.args.cpuct*self.Ps[state_string][action]*math.sqrt(self.Ns[state_string] + 0.00000001) # Q could be initializerd by network here
                u = 123456
            if u > best_UCB:
                best_UCB = u
                best_action = action
        a = best_action
        
        state.do_action(a) #state is now next_state
        v = self.search(state)

        if ((state_string, a) in self.Qsa):
            self.Qsa[(state_string, a)] = (self.Qsa[(state_string, a)] * self.Nsa[(state_string, a)] + v) / (self.Nsa[(state_string, a)] + 1)
            self.Nsa[(state_string, a)] += 1
        else:
            self.Qsa[(state_string, a)] = v
            self.Nsa[(state_string, a)] = 1
        #print("Player: ", playa, "winner: ", winner, "v: ", v)
        self.Ns[state_string] += 1
        return -v

    def self_play(self, dataset = Dataseto(), root = state(), iterations = 400, turn = 0): # DIRICHELET NMOISE
        s = root.stringify()
        if (s not in self.Ts): #terminal states
            self.Ts[s] = root.is_game_over()
        if (self.Ts[s] != 0):
            return (self.Ts[s])

        for _ in range(iterations):
            current_state = copy.deepcopy(root)
            self.search(current_state)

        temperature = 1
        if (turn > 12):
            temperature = 0.1
        policy = self.get_policy(root, temperature)
        dataset_index = dataset.add_point(state=root, policy=policy)
        action = np.random.choice(7, 1, p=policy)[0]
        root.do_action(action)
        v = self.self_play(dataset, root, iterations, turn + 1)

        dataset.data[dataset_index].V = torch.tensor([v])
        if ((s, action) in self.Qsa):
            self.Qsa[(s, action)] = (self.Qsa[(s, action)] * self.Nsa[(s, action)] + v) / (self.Nsa[(s, action)] + 1)
            self.Nsa[(s, action)] += 1
        else:
            self.Qsa[(s, action)] = v
            self.Nsa[(s, action)] = 1
        self.Ns[s] += 1
        return -v



class OLD():

    def __init__(self, node = node(), dataset = Dataseto(), tree_policy = None, rollout_policy = None):
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
        #self.dnn = Deep_Neural_Net()

    def launch(self):
        self.current_node = self.tree_root
        self.current_node.state.reset()
        self.MCTS_to_reward()
        self.tree_root.print_n_floor(self.tree_root, 0)
    
    def MCTS_to_reward(self):
        node = self.current_node
        print(node.state.stringify())
        if (node.is_terminal): #game is finished
            node.visits += 1
            v = node.state.get_reward()
            node.total_reward += v
            return -v

        if (node.is_fully_expanded == False and node.P == None):  #first visit
            v = self.rollout_policy()
            node.P = 1 #cheat
            node.visits += 1
            node.total_reward += v
            return -v

        if (node.is_fully_expanded == False and node.P != None):  #second visit
            self.expand_current_node()
            self.play_action(self.tree_policy())
            v = self.MCTS_to_reward()
            node.visits += 1
            node.total_reward += v
            return -v

        if (node.is_fully_expanded):
            action = self.tree_policy()
            self.play_action(action)
            v = self.MCTS_to_reward()
            node.visits += 1    #increase here or before PUCT evaluation ?
            node.total_reward += v
            return -v
        print(" YOOOOOO FUCKED UP BROOOO")

    def self_play(self, iterations = 400): # DIRICHELET NMOISE
        dataset = self.dataset
        if (self.root.is_terminal):
            return -(self.root.state.get_reward())
        initial_state = copy.deepcopy(self.root.state)

        for _ in range(iterations):
            self.current_node = self.root
            self.current_node.state.copy(initial_state)
            self.MCTS_to_reward()
        self.current_node = self.root
        self.current_node.state.copy(initial_state)
        policy = self.policy_policy()
        dataset_index = dataset.add_point(state=self.root.state, policy=policy)
        action = np.random.choice(7, 1, p=policy)[0]
        #action = self.select_highest_visits()
        self.play_action(action)
        self.root = self.current_node
        v = self.self_play()
        dataset.data[dataset_index].V = torch.tensor([v])
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
        self.self_play()
    
    def policy_policy(self, node = None): #IT FUCKED UP
        '''
            NO TEMPERATURE YET
            return policy vector based on visit numbers
            USES ROOT not current node !!!!
            state must correspond to node
        '''
        if node == None :
            node = self.root
        policy = np.zeros(7, dtype=float)
        for action in node.actions:
            policy[action] = node.children.get(action).visits
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
