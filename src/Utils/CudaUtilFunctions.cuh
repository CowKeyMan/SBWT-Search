#ifndef CUDA_UTIL_FUNCTIONS_CUH
#define CUDA_UTIL_FUNCTIONS_CUH

/**
 * @file CudaUtilFunctions.cuh
 * @brief Contains kernel device functions used commonly in some kernels
 * */

__device__ auto get_idx() -> int {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

#endif
