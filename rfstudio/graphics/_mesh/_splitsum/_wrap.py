# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

import numpy as np
import torch
import torch.utils.cpp_extension

#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None
def _get_plugin():
    # Return cached plugin if already loaded.
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/cubemap.cu',
        'c_src/common.cpp',
        'c_src/torch_bindings.cpp'
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory('rfstudio_render_utils', False), 'lock')
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='rfstudio_render_utils', sources=source_paths, extra_cflags=opts,
         extra_cuda_cflags=opts, extra_ldflags=ldflags, with_cuda=True, verbose=False)

    # Import, cache, and return the compiled module.
    import rfstudio_render_utils
    _cached_plugin = rfstudio_render_utils
    return _cached_plugin

#----------------------------------------------------------------------------
# cubemap filter with filtering across edges

class _diffuse_cubemap_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        out = _get_plugin().diffuse_cubemap_fwd(cubemap)
        ctx.save_for_backward(cubemap)
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, = ctx.saved_variables
        cubemap_grad = _get_plugin().diffuse_cubemap_bwd(cubemap, dout)
        return cubemap_grad, None

def diffuse_cubemap(cubemap: torch.Tensor, use_python=False) -> torch.Tensor:
    assert cubemap.is_cuda
    if use_python:
        assert False
    else:
        out = _diffuse_cubemap_func.apply(cubemap)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of diffuse_cubemap contains inf or NaN"
    return out

class _specular_cubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):
        out = _get_plugin().specular_cubemap_fwd(cubemap, bounds, roughness, costheta_cutoff)
        ctx.save_for_backward(cubemap, bounds)
        ctx.roughness, ctx.theta_cutoff = roughness, costheta_cutoff
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, bounds = ctx.saved_variables
        cubemap_grad = _get_plugin().specular_cubemap_bwd(cubemap, bounds, dout, ctx.roughness, ctx.theta_cutoff)
        return cubemap_grad, None, None, None

# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff, index):
    def ndfGGX(alphaSqr, costheta):
        costheta = np.clip(costheta, 0.0, 1.0)
        d = (costheta * alphaSqr - costheta) * costheta + 1.0
        return alphaSqr / (d * d * np.pi)

    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi/2.0, nSamples))
    D = np.cumsum(ndfGGX(roughness**4, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)

    # Brute force compute lookup table with bounds
    bounds = _get_plugin().specular_bounds(res, costheta[idx], index)

    return costheta[idx], bounds
__ndfBoundsDict = {}


def specular_cubemap(
    cubemap: torch.Tensor,
    roughness: float,
    cutoff: float = 0.99,
    use_python: bool = False,
) -> torch.Tensor:
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(cubemap.shape)
    assert cubemap.is_cuda

    if use_python:
        assert False
    else:
        key = (cubemap.shape[1], roughness, cutoff, cubemap.device.index)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        out = _specular_cubemap.apply(cubemap, roughness, *__ndfBoundsDict[key])
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of specular_cubemap contains inf or NaN"
    return out[..., 0:3] / out[..., 3:]
