#include "Tools/GpuUtils.h"
#include "Tools/TypeDefinitions.h"

namespace gpu_utils {

template <class Error_t>
auto getErrorString(Error_t code) -> const char * {
  return cudaGetErrorString(code);
}

template auto getErrorString(cudaError_t code) -> const char *;

auto get_free_gpu_memory() -> u64 {
  u64 free = 0;
  u64 total = 0;
  GPU_CHECK(cudaMemGetInfo(&free, &total));
  return free;
}

}  // namespace gpu_utils
