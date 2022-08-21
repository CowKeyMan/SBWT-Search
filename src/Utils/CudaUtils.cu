#include "Utils/CudaUtils.cuh"

auto get_free_gpu_memory() -> size_t {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  return free;
}
