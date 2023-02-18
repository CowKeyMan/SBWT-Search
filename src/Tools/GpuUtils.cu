#include "Tools/GpuUtils.h"
#include "Tools/TypeDefinitions.h"
#include "hip/hip_runtime.h"

namespace gpu_utils {

auto get_free_gpu_memory() -> u64 {
  u64 free = 0;
  u64 total = 0;
  GPU_CHECK(hipMemGetInfo(&free, &total));
  return free;
}

}  // namespace gpu_utils
