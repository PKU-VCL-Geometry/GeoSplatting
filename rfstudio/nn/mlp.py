from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, TypeAlias

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from rfstudio.utils.decorator import lazy

from .module import Module

activation_dict: Dict[ActivationType, Callable[[Tensor], Tensor]] = {
    'relu': lambda x: F.relu(x, inplace=True),
    'sigmoid': lambda x: x.sigmoid(),
    'tanh': F.tanh,
    'softplus': F.softplus,
    'none': lambda x: x,
}

ActivationType: TypeAlias = Literal['relu', 'sigmoid', 'tanh', 'softplus', 'none']


@dataclass
class MLP(Module):

    """Multilayer perceptron"""

    layers: List[int] = ...
    skip_connections: List[int] = field(default_factory=list)
    activation: ActivationType = 'none'
    bias: bool = True
    initialization: Literal[
        'kaiming-uniform',
        'kaiming-normal',
        'normal',
        'trunc-normal',
        'xavier-uniform',
        'xavier-normal',
        'orthogonal',
        'default',
    ] = 'default'
    weight_norm: bool = False

    def __setup__(self) -> None:
        assert len(self.layers) > 1
        assert all([layer != -1 for layer in self.layers[1:]]), "Only the first layer can be set to auto size"
        assert 0 not in self.skip_connections, "Skip connection at layer 0 doesn't make sense."

        self.skip_connection_set = set(self.skip_connections)
        nn_layers = []
        for i, in_dim, out_dim in zip(range(len(self.layers) - 1), self.layers[:-1], self.layers[1:]):
            if i in self.skip_connection_set:
                in_dim = -1 if self.layers[0] == -1 else (in_dim + self.layers[0])
            nn_layers.append(
                nn.Linear(in_dim, out_dim, bias=self.bias)
                if in_dim != -1
                else nn.LazyLinear(out_dim, bias=self.bias)
            )
        self.nn_layers = nn.ModuleList(nn_layers)
        self._loaded = False

    def _load_from_state_dict(
        self,
        state_dict: Dict,
        prefix: str,
        local_metadata: Dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        self._loaded = True
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @lazy
    def initialize_weights(self, init_dim: Optional[int] = None) -> None:
        if self._loaded:
            return
        for in_dim, layer in zip(self.layers[:-1], self.nn_layers):
            if isinstance(layer, nn.LazyLinear):
                assert init_dim is not None
                in_dim = init_dim if in_dim == -1 else (in_dim + init_dim)
                layer.in_features = in_dim
                if layer.has_uninitialized_params():
                    layer.weight.materialize((layer.out_features, in_dim))
                    if layer.bias is not None:
                        layer.bias.materialize((layer.out_features, ))
            if self.initialization == 'kaiming-uniform':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif self.initialization == 'kaiming-normal':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif self.initialization == 'normal':
                nn.init.normal_(layer.weight, std=0.02)
            elif self.initialization == 'trunc-normal':
                nn.init.trunc_normal_(layer.weight, std=0.02)
            elif self.initialization == 'xavier-uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif self.initialization == 'xavier-normal':
                nn.init.xavier_normal_(layer.weight)
            elif self.initialization == 'orthogonal':
                nn.init.orthogonal_(layer.weight)
            elif self.initialization == 'default':
                layer.reset_parameters()
            else:
                raise ValueError(self.initialization)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        if self.weight_norm:
            self.nn_layers = nn.ModuleList([
                nn.utils.weight_norm(layer) if isinstance(layer, nn.Linear) else layer
                for layer in self.nn_layers
            ])

    def __call__(self, inputs: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """
        Process input with a multilayer perceptron.
        Args:
            inputs: Network input

        Returns:
            MLP network output
        """
        activation_fn = activation_dict[self.activation]
        x = inputs
        self.initialize_weights(x.shape[-1])
        for i, layer in enumerate(self.nn_layers):
            if i in self.skip_connection_set:
                x = torch.cat((inputs, x), dim=-1)
            x = layer(x)
            if i < len(self.nn_layers) - 1:
                x = F.relu(x)
            else:
                x = activation_fn(x)
        return x
