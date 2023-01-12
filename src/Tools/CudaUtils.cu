#include "Tools/GpuUtils.h"

namespace gpu_utils {

template <class Error_t>
auto getErrorString(Error_t code) -> const char * {
  return cudaGetErrorString(code);
}

template auto getErrorString(cudaError_t code) -> const char *;

auto get_free_gpu_memory() -> size_t {
  size_t free = 0;
  size_t total = 0;
  GPU_CHECK(cudaMemGetInfo(&free, &total));
  return free;
}

}  // namespace gpu_utils
