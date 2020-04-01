from node import node
from sklearn import preprocessing
import numpy as np

class dataset():
    def _init(self):
        self.data = []

    def make_policy_vector_from_node(self, node):
        out = np.zeros([7], dtype= float)
        for act in node.actions :
            out[act] = node.children.get(act).visits.float()
        out = out / sum(out)                                    #dunno untested 
        return (out)

    