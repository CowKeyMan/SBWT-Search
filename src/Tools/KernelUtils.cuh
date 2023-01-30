#ifndef KERNEL_UTILS_CUH
#define KERNEL_UTILS_CUH

/**
 * @file KernelUtils.cuh
 * @brief Contains kernel device functions used commonly in some kernels
 */

namespace gpu_utils {

inline __device__ auto get_idx() -> int {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

}  // namespace gpu_utils

#endif
