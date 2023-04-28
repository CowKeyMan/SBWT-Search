#include "Tools/GpuPointer.h"
#include "Tools/GpuUtils.h"
#include "UtilityKernels/GetBoolFromBitVector_test.cuh"
#include "UtilityKernels/Rank_test.h"
#include "hip/hip_runtime.h"

using gpu_utils::GpuPointer;

namespace sbwt_search {

auto get_bool_from_bit_vector(const GpuPointer<u64> &bit_vector, u64 index)
  -> bool {
  GpuPointer<u64> d_result(1);
  hipLaunchKernelGGL(
    d_global_get_bool_from_bit_vector,
    1,
    1,
    0,
    nullptr,
    bit_vector.data(),
    index,
    d_result.data()
  );
  GPU_CHECK(hipPeekAtLastError());
  GPU_CHECK(hipDeviceSynchronize());
  u64 result = static_cast<u64>(-1);
  d_result.copy_to(&result);
  return static_cast<bool>(result);
}

}  // namespace sbwt_search
