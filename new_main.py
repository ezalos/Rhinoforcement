from MCTS import MCTS
from ARGS import ARGS
from data import Dataseto
from deep import ConnectNet
from deep import cross_entropy_loss_batch
from state import state
import torch
from deep import NetHandler
import numpy
from Trainer import Trainer
#from blackfire import probe

#probe.initialize()
#probe.enable()
torch.set_printoptions(linewidth=100, precision=2)
numpy.set_printoptions(linewidth=100, precision=2)


net = ConnectNet()
datasett = Dataseto()
testsett = Dataseto()
MC = MCTS(net)
NetHandler = NetHandler(net, ARGS)
NetHandler.train_init()

MC.self_play(datasett, root=state())
MC.self_play(testsett, root=state()) 
MC.self_play(datasett, root=state())
MC.self_play(testsett, root=state()) 
MC.self_play(datasett, root=state())
MC.self_play(testsett, root=state())


for i in range(1000):
    print(i)
    MC.self_play(datasett, root=state())
    MC.self_play(testsett, root=state())
    trainloader = torch.utils.data.DataLoader(datasett, batch_size=10, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testsett, batch_size=10, shuffle=True, num_workers=2)
    NetHandler.train(trainloader)
    NetHandler.test_error(testloader)
    if (i % 7 == 0):
        print("len",len(datasett))
        datasett.reset()
        testsett.reset()
        print("RESET")


#trainloader = torch.utils.data.DataLoader(datasett, batch_size=10, shuffle=True, num_workers=2)
#for i, data in enumerate(trainloader, 0):
#    print(data[1])
#    print("\n\nnet out")
#    print(net.forward(data[0]))
#    print("\n\n Ploss")
#     print(cross_entropy_loss_batch(net.forward(data[0])[0], data[1]))
#    print("single")
#    print(net.forward(data[0])[0][0])
#    print(data[1][0])
#    print(net.PLoss(net.forward(data[0])[0][0], data[1][0]))


#probe.end()