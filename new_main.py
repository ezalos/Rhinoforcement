from MCTS import MCTS
from data import Dataseto
from deep import ConnectNet
from state import state

print("YdfgfgO")

net = ConnectNet()
datasett = Dataseto()

MC = MCTS(net)
#for _ in range(5):
#	MC.self_play(datasett)
print("YEE")

statee = state()
print(net.evaluate(statee))
print("YEERUAKJHBASd")
