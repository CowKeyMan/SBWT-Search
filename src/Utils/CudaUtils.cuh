#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

__device__ auto get_idx() -> int {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

#endif
