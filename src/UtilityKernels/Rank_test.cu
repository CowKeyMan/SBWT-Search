#include "Tools/GpuPointer.h"
#include "Tools/GpuUtils.h"
#include "UtilityKernels/Rank_test.cuh"
#include "UtilityKernels/Rank_test.h"
#include "hip/hip_runtime.h"

using gpu_utils::GpuPointer;

namespace sbwt_search {

auto get_rank(
  const GpuPointer<u64> &bit_vector,
  const GpuPointer<u64> &poppy_layer_0,
  const GpuPointer<u64> &poppy_layer_1_2,
  const u64 index
) -> u64 {
  GpuPointer<u64> d_result(1);
  hipLaunchKernelGGL(
    d_global_rank,
    1,
    1,
    0,
    nullptr,
    bit_vector.data(),
    poppy_layer_0.data(),
    poppy_layer_1_2.data(),
    index,
    d_result.data()
  );
  GPU_CHECK(hipPeekAtLastError());
  GPU_CHECK(hipDeviceSynchronize());
  u64 result = static_cast<u64>(-1);
  d_result.copy_to(&result);
  return result;
}

}  // namespace sbwt_search
