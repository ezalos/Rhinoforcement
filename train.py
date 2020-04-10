
import torch

class patron():
    def __init__(self, dataset= None):
        self.dataset = dataset

    def dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=1)