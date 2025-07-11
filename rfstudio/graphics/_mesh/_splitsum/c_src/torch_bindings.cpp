/*
 * Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related 
 * documentation and any modifications thereto. Any use, reproduction, 
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or 
 * its affiliates is strictly prohibited.
 */

#ifdef _MSC_VER 
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <algorithm>
#include <string>

#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) { cudaError_t err = CUDA_CALL; AT_CUDA_CHECK(cudaGetLastError()); }
#define NVDR_CHECK_GL_ERROR(GL_CALL) { GL_CALL; GLenum err = glGetError(); TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]"); }
#define CHECK_TENSOR(X, DIMS, CHANNELS) \
    TORCH_CHECK(X.is_cuda(), #X " must be a cuda tensor") \
    TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kBFloat16, #X " must be fp32 or bf16") \
    TORCH_CHECK(X.dim() == DIMS, #X " must have " #DIMS " dimensions") \
    TORCH_CHECK(X.size(DIMS - 1) == CHANNELS, #X " must have " #CHANNELS " channels")

#include "common.h"
#include "cubemap.h"

#define BLOCK_X 8
#define BLOCK_Y 8

//------------------------------------------------------------------------
// cubemap.cu

void DiffuseCubemapFwdKernel(DiffuseCubemapKernelParams p);
void DiffuseCubemapBwdKernel(DiffuseCubemapKernelParams p);
void SpecularBoundsKernel(SpecularBoundsKernelParams p);
void SpecularCubemapFwdKernel(SpecularCubemapKernelParams p);
void SpecularCubemapBwdKernel(SpecularCubemapKernelParams p);

//------------------------------------------------------------------------
// Tensor helpers

void update_grid(dim3 &gridSize, torch::Tensor x)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
}

template<typename... Ts>
void update_grid(dim3& gridSize, torch::Tensor x, Ts&&... vs)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
    update_grid(gridSize, std::forward<Ts>(vs)...);
}

Tensor make_cuda_tensor(torch::Tensor val)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    return res;
}

Tensor make_cuda_tensor(torch::Tensor val, dim3 outDims, torch::Tensor* grad = nullptr)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    if (val.dim() == 4)
        res._dims[0] = outDims.z, res._dims[1] = outDims.y, res._dims[2] = outDims.x, res._dims[3] = val.size(3);
    else
        res._dims[0] = outDims.z, res._dims[1] = outDims.x, res._dims[2] = val.size(2), res._dims[3] = 1; // Add a trailing one for indexing math to work out

    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    if (grad != nullptr)
    {
        if (val.dim() == 4)
            *grad = torch::empty({ outDims.z, outDims.y, outDims.x, val.size(3) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA, val.device().index()));
        else // 3
            *grad = torch::empty({ outDims.z, outDims.x, val.size(2) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA, val.device().index()));

        res.d_val = res.fp16 ? (void*)grad->data_ptr<torch::BFloat16>() : (void*)grad->data_ptr<float>();
    }
    return res;
}

//------------------------------------------------------------------------
// filter_cubemap

torch::Tensor diffuse_cubemap_fwd(torch::Tensor cubemap)
{
    CHECK_TENSOR(cubemap, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    DiffuseCubemapKernelParams p;
    update_grid(p.gridSize, cubemap);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, cubemap.device().index());
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 3 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)DiffuseCubemapFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor diffuse_cubemap_bwd(torch::Tensor cubemap, torch::Tensor grad)
{
    CHECK_TENSOR(cubemap, 4, 3);
    CHECK_TENSOR(grad, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    DiffuseCubemapKernelParams p;
    update_grid(p.gridSize, cubemap);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    torch::Tensor cubemap_grad;
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.out = make_cuda_tensor(grad, p.gridSize);

    cubemap_grad = torch::zeros({ p.gridSize.z, p.gridSize.y, p.gridSize.x, cubemap.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, cubemap.device().index()));
    p.cubemap.d_val = (void*)cubemap_grad.data_ptr<float>();

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)DiffuseCubemapBwdKernel, gridSize, blockSize, args, 0, stream));

    return cubemap_grad;
}

torch::Tensor specular_bounds(int resolution, float costheta_cutoff, int index)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    SpecularBoundsKernelParams p;
    p.costheta_cutoff = costheta_cutoff;
    p.gridSize = dim3(resolution, resolution, 6);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, index);
    torch::Tensor out = torch::zeros({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 6*4 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)SpecularBoundsKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor specular_cubemap_fwd(torch::Tensor cubemap, torch::Tensor bounds, float roughness, float costheta_cutoff)
{
    CHECK_TENSOR(cubemap, 4, 3);
    CHECK_TENSOR(bounds, 4, 6*4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    SpecularCubemapKernelParams p;
    p.roughness = roughness;
    p.costheta_cutoff = costheta_cutoff;
    update_grid(p.gridSize, cubemap);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, cubemap.device().index());
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 4 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.bounds = make_cuda_tensor(bounds, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)SpecularCubemapFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor specular_cubemap_bwd(torch::Tensor cubemap, torch::Tensor bounds, torch::Tensor grad, float roughness, float costheta_cutoff)
{
    CHECK_TENSOR(cubemap, 4, 3);
    CHECK_TENSOR(bounds, 4, 6*4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    SpecularCubemapKernelParams p;
    p.roughness = roughness;
    p.costheta_cutoff = costheta_cutoff;
    update_grid(p.gridSize, cubemap);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    torch::Tensor cubemap_grad;
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.bounds = make_cuda_tensor(bounds, p.gridSize);
    p.out = make_cuda_tensor(grad, p.gridSize);

    cubemap_grad = torch::zeros({ p.gridSize.z, p.gridSize.y, p.gridSize.x, cubemap.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, cubemap.device().index()));
    p.cubemap.d_val = (void*)cubemap_grad.data_ptr<float>();

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)SpecularCubemapBwdKernel, gridSize, blockSize, args, 0, stream));

    return cubemap_grad;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diffuse_cubemap_fwd", &diffuse_cubemap_fwd, "diffuse_cubemap_fwd");
    m.def("diffuse_cubemap_bwd", &diffuse_cubemap_bwd, "diffuse_cubemap_bwd");
    m.def("specular_bounds", &specular_bounds, "specular_bounds");
    m.def("specular_cubemap_fwd", &specular_cubemap_fwd, "specular_cubemap_fwd");
    m.def("specular_cubemap_bwd", &specular_cubemap_bwd, "specular_cubemap_bwd");
}