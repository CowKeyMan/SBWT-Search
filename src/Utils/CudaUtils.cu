#include "Utils/CudaUtils.cuh"

namespace gpu_utils {

auto get_free_gpu_memory() -> size_t {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  return free;
}

}  // namespace gpu_utils
