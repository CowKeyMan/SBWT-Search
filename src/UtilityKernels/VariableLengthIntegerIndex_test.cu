#include "UtilityKernels/VariableLengthIntegerIndex_test.cuh"
#include "UtilityKernels/VariableLengthIntegerIndex_test.h"

using gpu_utils::GpuPointer;

namespace sbwt_search {

auto get_variable_length_int_index(
  const GpuPointer<u64> &container, u64 width, u64 width_set_bits, u64 index
) -> u64 {
  GpuPointer<u64> d_result(1);
  hipLaunchKernelGGL(
    d_global_variable_length_int_index,
    1,
    1,
    0,
    nullptr,
    container.data(),
    width,
    width_set_bits,
    index,
    d_result.data()
  );
  u64 result = static_cast<u64>(-1);
  d_result.copy_to(&result);
  return result;
}

}  // namespace sbwt_search
