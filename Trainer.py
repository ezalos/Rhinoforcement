from MCTS import MCTSO
from MCTS import DotDict
from state import state
import copy
import numpy
from deep import NetHandler
from deep import ConnectNet
from data import Dataseto
from data import DataHandler
import time

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
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'maxHistory': 5,
    'batch_size': 8,
    'Episodes': 100,
    'Epochs': 10000
})

def self_play(MC):
    game = state()
    value = 0
    while (game.victory == ''):
        for _ in range(400):
            root = copy.deepcopy(game)
            v = MC.search(root)
        game.do_action(numpy.random.choice(7, 1, p = MC.get_policy(game))[0])
        game.display()
    return value

def printo(MC, state = state()):
    s = state.stringify()
    for i in range(7):
        print("Q: ", MC.Qsa[(s, i)])
        print("N: ", MC.Nsa[(s, i)])

#net = ConnectNet()
#jo = MCTSO(net)
#dd = Dataseto()
#v = 0
#for i in range(100):
#    start = time.time()
#    v += jo.self_play(dd)
#    print(time.time() - start)

class Trainer():
    def __init__(self, net= ConnectNet(), args = ARGS):
        self.MCTS = MCTSO(net, args)
        self.dataHandler = DataHandler(args)
        self.args = args
        self.net = net
        self.net.to()
        self.netHandler = NetHandler(net, args)

    def createSet(self):
        dataset = Dataseto()
        i = 0
        while len(dataset) < 1000:
            MCTS = MCTSO(self.net, self.args)
            MCTS.self_play(dataset=dataset, root=state())
            i += 1
            print(i)
        print("LENNN: ", len(dataset))
        
        self.dataHandler.add_dataset(dataset)

    def train(self):
        net = self.net
        net.train()
        trainLoader = self.dataHandler.get_data_loader()
        self.netHandler.train(trainLoader)

    def arena_fightooo(self, nn1, nn2):
        pass

    def execute(self):
        self.createSet()
        self.train()

t = Trainer()
t.execute()