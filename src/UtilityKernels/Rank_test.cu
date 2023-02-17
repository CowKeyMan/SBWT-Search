#include "Tools/GpuPointer.h"
#include "Tools/GpuUtils.h"
#include "UtilityKernels/Rank_test.cuh"
#include "UtilityKernels/Rank_test.h"

using gpu_utils::GpuPointer;

namespace sbwt_search {

auto get_rank(
  const GpuPointer<u64> &bit_vector,
  const GpuPointer<u64> &poppy_layer_0,
  const GpuPointer<u64> &poppy_layer_1_2,
  const u64 index
) -> u64 {
  GpuPointer<u64> d_result(1);
  d_global_rank<<<1, 1>>>(
    bit_vector.get(),
    poppy_layer_0.get(),
    poppy_layer_1_2.get(),
    index,
    d_result.get()
  );
  GPU_CHECK(cudaPeekAtLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  u64 result = static_cast<u64>(-1);
  d_result.copy_to(&result);
  return result;
}

}  // namespace sbwt_search
