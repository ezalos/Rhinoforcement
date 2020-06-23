from MCTS import MCTS
from data import Dataseto
from deep import ConnectNet
from state import state
import torch

#from blackfire import probe

#probe.initialize()
#probe.enable()




print("YdfgfgO")

net = ConnectNet()
datasett = Dataseto()

MC = MCTS(net)
for _ in range(1):
	MC.self_play(datasett)

trainloader = torch.utils.data.DataLoader(datasett, batch_size=10, shuffle=True, num_workers=2)
for i, data in enumerate(trainloader, 0):
    print(i)
    print(data)
    print(net.forward(data))

print(net.PLoss(net.evaluate_encoded(datasett[0][0])[0][0], datasett[0][1]))
#statee = state()
#statee.display()
#print(net.evaluate(statee)[0].size())
#print("YEERUAKJHBASd")

#probe.end()