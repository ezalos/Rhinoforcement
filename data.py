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
        #self.size = 0

    def make_policy_vector_from_node(self, node):
        out = np.zeros([7], dtype= float)
        for act in node.actions :
            out[act] = node.children.get(act).visits
        out = out / np.sum(out)                                    #dunno untested 
        return (out)

    def add_point(self, state, policy):
        self.data.append(datapoint(state, policy))
        #self.size += 1
        return (len(self.data) - 1)

    def add_value_to_set(self, value, last_node): #if 1 game = 1 dataset then this is needlessly complicated
        i = 1
        while (last_node.daddy != None): #should be no need for +1 as final state will not be included in set
            self.data[-i].V = np.array([value])
            last_node = last_node.daddy
            i += 1

    def display(self):
        print(self.data)
        for data in self.data:
            data.display()
