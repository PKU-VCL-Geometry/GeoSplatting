from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn

from rfstudio.nn import Module, ParameterModule


@dataclass
class PosEncoding(Module):

    """
    Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    num_frequencies: int
    min_freq_exp: float = 0.0
    max_freq_exp: float = 8.0
    include_input: bool = True

    def __call__(self, inputs: Float[Tensor, "... IN"]) -> Float[Tensor, "... OUT"]:
        """
        Calculates NeRF encoding

        Args:
            inputs: For best performance, the input tensor should be between -1 and 1.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = torch.pi * inputs                              # scale to [-pi, pi]
        freqs = 2 ** torch.linspace(self.min_freq_exp, self.max_freq_exp, self.num_frequencies, device=inputs.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs               # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1) # [..., "input_dim" * "num_scales"]
        base_inputs = [inputs] if self.include_input else []
        return torch.sin(torch.cat(base_inputs + [scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))


@dataclass
class SHEncoding(Module):

    '''
    Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode.

    '''

    num_levels: Literal[1, 2, 3, 4] = ...
    backend: Literal["tcnn", "torch"] = "tcnn"

    def __setup__(self) -> None:
        assert self.num_levels in [1, 2, 3, 4]
        if self.backend == 'torch':
            raise NotImplementedError
        elif self.backend == 'tcnn':
            import tinycudann as tcnn
            encoding_config = {
                "otype": "SphericalHarmonics",
                "degree": self.num_levels,
            }
            self.encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )
        else:
            raise ValueError(self.backend)

    def __call__(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        return self.encoder(in_tensor)


@dataclass
class HashEncoding(Module):

    """
    Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        backend: Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
    """

    mlp: Module = ...
    num_levels: int = 16
    min_res: int = 16
    max_res: int = 1024
    log2_hashmap_size: int = 19
    features_per_level: int = 2
    hash_init_scale: float = 0.001
    backend: Literal["tcnn", "torch"] = "tcnn"
    interpolation: Literal["nearest", "linear", "smoothstep"] = "linear"
    grad_scaling: Optional[float] = None

    def reset(self) -> None:
        device = self.device
        self.mlp = replace(self.mlp)
        self.mlp.__setup__()
        self.__setup__()
        self.to(device)
        if self.backend == "tcnn":
            import tinycudann as tcnn
            tcnn.free_temporary_memory()

    def __setup__(self) -> None:
        assert self.grad_scaling is None or self.grad_scaling > 0
        self.hash_table_size = 2 ** self.log2_hashmap_size

        levels = torch.arange(self.num_levels)
        self.growth_factor = (
            np.exp((np.log(self.max_res) - np.log(self.min_res)) / (self.num_levels - 1))
            if self.num_levels > 1
            else 1
        )
        self.scalings = torch.floor(self.min_res * self.growth_factor ** levels)

        self.hash_offset = levels * self.hash_table_size

        self.hash_table = None
        if self.backend == "torch":
            assert self.interpolation == "linear", (
                f"interpolation '{self.interpolation}' "
                "is not supported for torch encoding backend"
            )
            self.encoder = ParameterModule.from_tensor(
                (torch.rand(self.hash_table_size * self.num_levels, self.features_per_level) * 2 - 1) *
                self.hash_init_scale
            )
            self.hash_table = self.encoder.params
        elif self.backend == "tcnn":
            import tinycudann as tcnn
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.min_res,
                "per_level_scale": self.growth_factor,
                "interpolation": self.interpolation.capitalize(),
            }
            self.encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )
        else:
            raise ValueError('The argument `implementation` must be one of torch and tcnn.')

    def hash_fn(self, in_tensor: Int[Tensor, "*bs num_levels 3"]) -> Shaped[Tensor, "*bs num_levels"]:
        """
        Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :] * 0.5 + 0.5                     # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device) # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)      # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]        # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = torch.add(
            f0312 * offset[..., 2:3],
            f4756 * (1 - offset[..., 2:3]),
        )                                      # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def __call__(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.grad_scaling is not None:
            in_tensor = in_tensor * (1 / self.grad_scaling) + in_tensor.detach() * (1 - 1 / self.grad_scaling)
        if self.backend == 'tcnn':
            feats = self.encoder(in_tensor.flatten(end_dim=-2) * 0.5 + 0.5)
            feats = feats.view(*in_tensor.shape[:-1], -1).float()
        else:
            feats = self.pytorch_fwd(in_tensor)
        if self.grad_scaling is not None:
            feats = feats * self.grad_scaling + feats.detach() * (1 - self.grad_scaling)
        return self.mlp(feats)


@dataclass
class TriplaneEncoding(Module):

    """
    Learned triplane encoding

    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the
    three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].

    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing
    and symmetrical, unlike with VM decomposition where we needed one component with a vector along all the x, y, z
    directions for symmetry.

    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting
    at the origin, and the encoding being the element-wise product of the element at the projection of [i, j, k] on
    these planes.

    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of grid.
        num_components: The number of scalar triplanes to use (ie: output feature size)
        init_scale: The scale of the initial values of the planes
        product: Whether to use the element-wise product of the planes or the sum
    """

    resolution: int = 32
    num_components: int = 64
    init_scale: float = 0.1
    reduce: Literal["sum", "product"] = "sum"

    def __setup__(self) -> None:
        self.plane_coef = nn.Parameter(
            self.init_scale *
            torch.randn((3, self.num_components, self.resolution, self.resolution))
        ) # [3, num_components, resolution, resolution]

    def __call__(self, inputs: Float[Tensor, "... 3"]) -> Float[Tensor, "... num_components"]:
        """Sample features from this encoder. Expects inputs to be in range [-1, 1]"""

        original_shape = inputs.shape[:-1]     # ...

        plane_coord = torch.stack((
            inputs[..., [0, 1]],
            inputs[..., [0, 2]],
            inputs[..., [1, 2]],
        ), dim=0)                              # [3, ..., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2) # [3, flattened_bs, 1, 2]
        plane_features = F.grid_sample(
            self.plane_coef,                                 # [3, num_components, resolution, resolution]
            plane_coord,                                     # [3, flattened_bs, 1, 2]
            align_corners=True
        )                                                    # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(0).squeeze(-1).T # [flattened_bs, num_components]
        else:
            plane_features = plane_features.sum(0).squeeze(-1).T  # [flattened_bs, num_components]

        return plane_features.reshape(*original_shape, self.num_components)

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """
        Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=True,
        )

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution
