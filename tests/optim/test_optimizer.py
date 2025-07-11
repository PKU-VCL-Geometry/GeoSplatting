import numpy as np
import torch

from rfstudio.graphics import Splats
from rfstudio.optim import ModuleOptimizers, Optimizer

if __name__ == '__main__':
    fields = ["means", "scales", "quats", "shs", "colors", "opacities"]
    gaussians = Splats.random(10, sh_degree=3, random_scale=1, requires_grad=True)
    optimizers = ModuleOptimizers(
        mixed_precision = False,
        optim_dict = {
            name: Optimizer(
                category=torch.optim.Adam,
                modules=gaussians.as_module(field_name=name),
                lr=1e-1
            )
            for name in fields
        }
    )

    def get_loss(x):
        loss = 0
        for name in fields:
            loss = loss + ((getattr(x, name) - 1) ** 2).mean()
        return loss

    for i in range(15):
        optimizers.zero_grad()
        loss = get_loss(gaussians)
        optimizers.backward(loss)
        optimizers.step()

    with torch.no_grad():
        indices = torch.tensor([0, 1, 2]).view(3, 1)
        new_gaussians = Splats.cat([
            gaussians[indices.flatten()],
            gaussians[indices.flatten()].split(2).flatten(),
        ], dim=0)
        gaussians.swap_(new_gaussians.requires_grad_())
        indices = torch.cat([
            indices,
            -indices.new_ones(np.prod(gaussians.shape) - indices.shape[0], indices.shape[1])
        ], dim=0)
    optimizers.mutate_params(indices=indices)
    loss = get_loss(gaussians).item()

    for i in range(15):
        optimizers.zero_grad()
        loss = get_loss(gaussians)
        optimizers.backward(loss)
        optimizers.step()
