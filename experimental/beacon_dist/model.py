
import torch

class Reconstructor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._bias = torch.nn.parameter.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        x.x += self._bias
        return x
