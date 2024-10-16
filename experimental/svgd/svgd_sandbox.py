

import matplotlib.pyplot as plt
import math
from common.torch import torch

from typing import NamedTuple


class RadialExp(NamedTuple):
    center: torch.Tensor
    scale: float

# The observation model returns the log probabilility of a particle
class ObservationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._components = [
            (1.0, RadialExp(center=torch.tensor([3, 4]), scale=1)),
            (0.5, RadialExp(center=torch.tensor([6, 7]), scale=1)),
            (2.0, RadialExp(center=torch.tensor([8, 4]), scale=1))
        ]

    def forward(self, x):
        # x is a b x d tensor
        b, d = x.shape
        assert d == 2
        log_prob_terms = torch.zeros(b, len(self._components))
        for i, (weight, comp) in enumerate(self._components):

            log_prob_terms[:, i] = (
                math.log(weight) -
                torch.sum(torch.abs(x - comp.center), dim=1) / comp.scale)
        return torch.logsumexp(log_prob_terms, 1)




def main():
    obs_model = ObservationModel()
    xs, ys = torch.meshgrid([torch.linspace(0, 10, 300)]*2, indexing='ij')
    eval_pts = torch.stack([xs, ys], dim=-1)
    eval_pts = eval_pts.reshape(-1, 2)
    log_probs = obs_model(eval_pts)
    log_probs = log_probs.reshape(*xs.shape)
    probs = torch.exp(log_probs)

    plt.figure()
    plt.contour(xs, ys, torch.exp(log_probs))
    plt.colorbar()
    plt.title(f'max: {torch.max(probs)} min: {torch.min(probs)}')
    plt.savefig('/tmp/plot.png')


if __name__ == "__main__":
    main()
