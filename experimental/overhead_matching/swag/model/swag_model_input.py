
import common.torch.load_torch_deps
import torch
from dataclasses import dataclass


@dataclass
class ModelInput:
    image: torch.Tensor
    metadata: list[dict]

    def to(self, *args, **kwargs):
        return ModelInput(
            image=self.image.to(*args, **kwargs),
            metadata=self.metadata)
