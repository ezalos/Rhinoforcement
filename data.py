from node import node
from sklearn import preprocessing
import numpy as np
import copy

class datapoint():
    def __init__(self, state, policy, value = None):
        self.S = copy.deepcopy(state)      #unmodified state
        self.P = policy                    #numpy array
        self.V = value                     #scalar

    def display(self):
        print("State: ")
        self.S.display()
        print("Policy: ", self.P)
        print("Value: ", self.V)
        print("")

class dataset():
    def __init__(self):
        self.data = []
        self.size = 0

    def make_policy_vector_from_node(self, node):
        out = np.zeros([7], dtype= float)
        for act in node.actions :
            out[act] = node.children.get(act).visits
        out = out / np.sum(out)                                    #dunno untested 
        return (out)

    def add_point(self, state, policy):
        self.data.append(datapoint(state, policy))
        self.size += 1
        return (len(self.data) - 1)

    def display(self):
        for data in self.data:
            data.display()
