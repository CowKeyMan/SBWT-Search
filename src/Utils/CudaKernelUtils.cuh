#ifndef CUDA_KERNEL_UTILS_CUH
#define CUDA_KERNEL_UTILS_CUH

/**
 * @file CudaKernelUtils.cuh
 * @brief Contains kernel device functions used commonly in some kernels
 * */

namespace gpu_utils {

__device__ auto get_idx() -> int {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

}  // namespace gpu_utils

#endif
