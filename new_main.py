from MCTS import MCTS
from data import Dataseto
from deep import ConnectNet

print("YdfgfgO")

net = ConnectNet()
datasett = Dataseto()

MC = MCTS(net)
print("YEE")
for _ in range(10):
	print("M")
	MC.self_play(datasett)
	print("YO")

MC.play_vs_MCTS()