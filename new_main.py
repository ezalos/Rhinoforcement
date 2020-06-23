from MCTS import MCTS
from data import Dataseto
from deep import ConnectNet
from state import state
import torch
from deep import cross_entropy_loss_batch
import numpy
#from blackfire import probe

#probe.initialize()
#probe.enable()



numpy.set_printoptions(linewidth=120)

net = ConnectNet()
datasett = Dataseto()

MC = MCTS(net)
for _ in range(1):
	MC.self_play(datasett)

trainloader = torch.utils.data.DataLoader(datasett, batch_size=10, shuffle=True, num_workers=2)
for i, data in enumerate(trainloader, 0):
    print(i)

#    print("\n\n\n\n\n  data[1]")
#    print(data[1])
#    print("\n\nnet out")
#    print(net.forward(data[0]))
    print("\n\n Ploss")
    print(cross_entropy_loss_batch(net.forward(data[0])[0], data[1]))
    print("single")
#    print(net.forward(data[0])[0][0])
#    print(data[1][0])
#    print(net.PLoss(net.forward(data[0])[0][0], data[1][0]))

#print(net.PLoss(net.evaluate_encoded(datasett[0][0])[0][0], datasett[0][1]))
#statee = state()
#statee.display()
#print(net.evaluate(statee)[0].size())
#print("YEERUAKJHBASd")

#probe.end()